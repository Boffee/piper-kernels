"""Offline configuration search for development benchmarks."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from typing import Any, Generic, TypeVar

from triton.runtime.errors import OutOfResources

from .environment import EnvironmentInfo
from .providers import (
    BenchmarkProvider,
    DistributionTimer,
    ProviderPhase,
    provider_phase_launch,
)
from .quality import QualityMetrics
from .reporting import SCHEMA_VERSION, add_output_arguments
from .timing import Timing, synchronized_wall_benchmark, triton_benchmark

PreparedT = TypeVar("PreparedT")
OutputT = TypeVar("OutputT")


class TuningStatus(StrEnum):
    """Outcome of evaluating one candidate."""

    MEASURED = "measured"
    QUALITY_REJECTED = "quality_rejected"
    SKIPPED = "skipped"


class UnsupportedTuningCandidateError(RuntimeError):
    """Signal that a candidate is unsupported without aborting the tuning run."""


@dataclass(frozen=True, slots=True)
class TuningCandidate(Generic[PreparedT, OutputT]):
    """A named configuration and a factory for its benchmark provider."""

    name: str
    configuration: Mapping[str, Any]
    make_provider: Callable[[], BenchmarkProvider[PreparedT, OutputT]]


@dataclass(frozen=True, slots=True)
class TuningRecord:
    """One measured, rejected, or skipped tuning candidate."""

    tuning: str
    candidate: str
    shape: Mapping[str, Any]
    configuration: Mapping[str, Any]
    phase: ProviderPhase
    status: TuningStatus
    selected: bool
    warmup_ms: int
    measurement_time_ms: int
    environment: EnvironmentInfo
    timing: Timing | None = None
    quality: QualityMetrics | None = None
    reason: str | None = None
    schema_version: int = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Return a versioned machine-readable tuning record."""
        return {
            "schema_version": self.schema_version,
            "record_type": "tuning_candidate",
            "tuning": self.tuning,
            "candidate": self.candidate,
            "shape": dict(self.shape),
            "configuration": dict(self.configuration),
            "phase": self.phase.value,
            "status": self.status.value,
            "selected": self.selected,
            "warmup_ms": self.warmup_ms,
            "measurement_time_ms": self.measurement_time_ms,
            "timing": None if self.timing is None else self.timing.as_dict(),
            "quality": None if self.quality is None else self.quality.as_dict(),
            "reason": self.reason,
            "environment": self.environment.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TuningRun:
    """Complete results from one candidate search."""

    records: tuple[TuningRecord, ...]

    @property
    def winner(self) -> TuningRecord | None:
        """Return the selected record, or ``None`` when no candidate passed."""
        return next((record for record in self.records if record.selected), None)


@dataclass(frozen=True, slots=True)
class _TuningRunContext:
    tuning: str
    shape: Mapping[str, Any]
    phase: ProviderPhase
    warmup_ms: int
    measurement_time_ms: int
    environment: EnvironmentInfo

    def record(
        self,
        candidate_name: str,
        configuration: Mapping[str, Any],
        status: TuningStatus,
        *,
        timing: Timing | None = None,
        quality: QualityMetrics | None = None,
        reason: str | None = None,
    ) -> TuningRecord:
        """Create one result while keeping run-wide fields in one place."""
        return TuningRecord(
            tuning=self.tuning,
            candidate=candidate_name,
            shape=self.shape,
            configuration=configuration,
            phase=self.phase,
            status=status,
            selected=False,
            warmup_ms=self.warmup_ms,
            measurement_time_ms=self.measurement_time_ms,
            timing=timing,
            quality=quality,
            reason=reason,
            environment=self.environment,
        )


def add_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """Add controls shared by every offline tuning CLI."""
    parser.add_argument(
        "--phase",
        type=ProviderPhase,
        choices=tuple(ProviderPhase),
        default=ProviderPhase.PREPARED_EXECUTION,
    )
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measurement-time-ms", type=int, default=200)
    parser.add_argument("--minimum-sqnr-db", type=float, default=20.0)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    add_output_arguments(parser, record_name="tuning candidate")


def validate_tuning_arguments(arguments: argparse.Namespace) -> None:
    """Validate controls shared by every offline tuning CLI."""
    if arguments.warmup_ms < 0 or arguments.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    if not math.isfinite(arguments.minimum_sqnr_db):
        raise SystemExit("minimum SQNR must be finite")
    if arguments.max_candidates <= 0:
        raise SystemExit("maximum candidate count must be positive")


def tuning_axis[T](values: Sequence[T] | None, production_value: T) -> tuple[T, ...]:
    """Resolve one optional tuning axis and remove duplicate explicit values."""
    return (production_value,) if values is None else tuple(dict.fromkeys(values))


def boolean_tuning_axis(value: bool | None, production_value: bool) -> tuple[bool, ...]:
    """Resolve one Boolean optional action against its production value."""
    return (production_value if value is None else value,)


def parse_optional_integer(value: str) -> int | None:
    """Parse zero as an omitted integer value for a numeric tuning axis."""
    converted = int(value)
    return None if converted == 0 else converted


def validate_tuning_candidate_count(
    axes: Sequence[Sequence[object]],
    maximum_candidates: int,
) -> None:
    """Reject a Cartesian search that exceeds the configured candidate budget."""
    candidate_count = math.prod(len(axis) for axis in axes)
    if candidate_count > maximum_candidates:
        raise SystemExit(
            f"search expands to {candidate_count} candidates; narrow the axes or increase "
            "--max-candidates"
        )


def meets_minimum_sqnr(quality: QualityMetrics, minimum_sqnr_db: float) -> bool:
    """Apply the common finite-output and SQNR quality gate."""
    return quality.nonfinite_mismatch_count == 0 and quality.sqnr_db >= minimum_sqnr_db


def print_tuning_results(records: Sequence[TuningRecord]) -> None:
    """Print the common compact tuning-result table."""
    print("| candidate | status | selected | p50 (ms) | SQNR (dB) | reason |")
    print("|:---|:---|:---:|---:|---:|:---|")
    for record in records:
        timing = "-" if record.timing is None else f"{record.timing.median_ms:.3f}"
        quality = "-" if record.quality is None else f"{record.quality.sqnr_db:.2f}"
        print(
            f"| {record.candidate} | {record.status.value} | {record.selected} "
            f"| {timing} | {quality} | {record.reason or ''} |"
        )


def _reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _phase_timer(
    provider: BenchmarkProvider[Any, Any],
    phase: ProviderPhase,
    device_timer: DistributionTimer,
) -> DistributionTimer:
    """Select the timer matching a provider execution boundary."""
    if phase is ProviderPhase.PREPARED_EXECUTION:
        return device_timer
    return partial(
        synchronized_wall_benchmark,
        synchronize=provider.synchronize,
    )


def _validate_tuning_run[PreparedT, OutputT](
    candidates: Sequence[TuningCandidate[PreparedT, OutputT]],
    *,
    tuning: str,
    warmup_ms: int,
    measurement_time_ms: int,
    measure_candidate_quality: Callable[[OutputT], QualityMetrics] | None,
    quality_gate: Callable[[QualityMetrics], bool] | None,
) -> None:
    """Validate invariants required by the shared candidate loop."""
    if not candidates:
        raise ValueError("at least one tuning candidate is required")
    if not tuning:
        raise ValueError("tuning name cannot be empty")
    if warmup_ms < 0 or measurement_time_ms <= 0:
        raise ValueError("warmup must be non-negative and measurement time must be positive")
    if quality_gate is not None and measure_candidate_quality is None:
        raise ValueError("a quality gate requires a quality measurement function")
    names = [candidate.name for candidate in candidates]
    if any(not name for name in names):
        raise ValueError("tuning candidate names cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("tuning candidate names must be unique")


def tune_candidates(
    candidates: Sequence[TuningCandidate[PreparedT, OutputT]],
    *,
    tuning: str,
    shape: Mapping[str, Any],
    environment: EnvironmentInfo,
    phase: ProviderPhase = ProviderPhase.PREPARED_EXECUTION,
    warmup_ms: int = 50,
    measurement_time_ms: int = 200,
    measure_candidate_quality: Callable[[OutputT], QualityMetrics] | None = None,
    quality_gate: Callable[[QualityMetrics], bool] | None = None,
    device_timer: DistributionTimer = triton_benchmark,
) -> TuningRun:
    """Measure candidates and select the fastest one that passes quality checks.

    Candidate factories define kernel-specific legality and launch details. Raise
    :class:`UnsupportedTuningCandidateError` for an unsupported configuration. Triton
    out-of-resource failures are skipped automatically. Unexpected exceptions propagate
    so compiler and benchmark bugs remain visible.
    """
    _validate_tuning_run(
        candidates,
        tuning=tuning,
        warmup_ms=warmup_ms,
        measurement_time_ms=measurement_time_ms,
        measure_candidate_quality=measure_candidate_quality,
        quality_gate=quality_gate,
    )

    context = _TuningRunContext(
        tuning=tuning,
        shape=shape,
        phase=phase,
        warmup_ms=warmup_ms,
        measurement_time_ms=measurement_time_ms,
        environment=environment,
    )
    skipped_errors = (UnsupportedTuningCandidateError, OutOfResources)
    records: list[TuningRecord] = []
    for candidate in candidates:
        configuration = dict(candidate.configuration)
        try:
            provider = candidate.make_provider()
            configuration.update(provider.configuration)
            launch = provider_phase_launch(provider, phase)
            output = launch()
            provider.synchronize()
            timer = _phase_timer(provider, phase, device_timer)

            quality = (
                measure_candidate_quality(output) if measure_candidate_quality is not None else None
            )
            if quality is not None and quality_gate is not None and not quality_gate(quality):
                records.append(
                    context.record(
                        candidate.name,
                        configuration,
                        TuningStatus.QUALITY_REJECTED,
                        quality=quality,
                        reason="quality gate rejected candidate",
                    )
                )
                continue

            timing = timer(launch, warmup_ms, measurement_time_ms)
            records.append(
                context.record(
                    candidate.name,
                    configuration,
                    TuningStatus.MEASURED,
                    timing=timing,
                    quality=quality,
                )
            )
        except skipped_errors as error:
            records.append(
                context.record(
                    candidate.name,
                    configuration,
                    TuningStatus.SKIPPED,
                    reason=_reason(error),
                )
            )

    measured = [record for record in records if record.status is TuningStatus.MEASURED]
    if measured:
        winner = min(
            measured,
            key=lambda record: (
                record.timing.median_ms if record.timing is not None else float("inf")
            ),
        )
        records[records.index(winner)] = replace(winner, selected=True)
    return TuningRun(tuple(records))
