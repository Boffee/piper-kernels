"""Offline configuration search for development benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from typing import Any, Generic, TypeVar

from triton.runtime.errors import OutOfResources

from .environment import EnvironmentInfo
from .providers import BenchmarkProvider, DistributionTimer
from .quality import QualityMetrics
from .reporting import SCHEMA_VERSION
from .timing import Timing, synchronized_wall_benchmark, triton_benchmark

PreparedT = TypeVar("PreparedT")
OutputT = TypeVar("OutputT")


class TuningPhase(StrEnum):
    """Provider execution phase measured by an offline tuning run."""

    PREPARED_EXECUTION = "prepared_execution"
    OPERATOR_END_TO_END = "operator_end_to_end"


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
    phase: TuningPhase
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


def _reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def tune_candidates(
    candidates: Sequence[TuningCandidate[PreparedT, OutputT]],
    *,
    tuning: str,
    shape: Mapping[str, Any],
    environment: EnvironmentInfo,
    phase: TuningPhase = TuningPhase.PREPARED_EXECUTION,
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

    skipped_errors = (UnsupportedTuningCandidateError, OutOfResources)
    records: list[TuningRecord] = []
    for candidate in candidates:
        configuration = dict(candidate.configuration)
        try:
            provider = candidate.make_provider()
            configuration.update(provider.configuration)
            if phase is TuningPhase.PREPARED_EXECUTION:
                prepared = provider.prepare()

                launch = partial(provider.run, prepared)
                output = launch()
                provider.synchronize()
                timer = device_timer
            else:
                launch = provider.run_operator
                output = launch()
                provider.synchronize()

                timer = partial(
                    synchronized_wall_benchmark,
                    synchronize=provider.synchronize,
                )

            quality = (
                measure_candidate_quality(output) if measure_candidate_quality is not None else None
            )
            if quality is not None and quality_gate is not None and not quality_gate(quality):
                records.append(
                    TuningRecord(
                        tuning=tuning,
                        candidate=candidate.name,
                        shape=shape,
                        configuration=configuration,
                        phase=phase,
                        status=TuningStatus.QUALITY_REJECTED,
                        selected=False,
                        warmup_ms=warmup_ms,
                        measurement_time_ms=measurement_time_ms,
                        quality=quality,
                        reason="quality gate rejected candidate",
                        environment=environment,
                    )
                )
                continue

            timing = timer(launch, warmup_ms, measurement_time_ms)
            records.append(
                TuningRecord(
                    tuning=tuning,
                    candidate=candidate.name,
                    shape=shape,
                    configuration=configuration,
                    phase=phase,
                    status=TuningStatus.MEASURED,
                    selected=False,
                    warmup_ms=warmup_ms,
                    measurement_time_ms=measurement_time_ms,
                    timing=timing,
                    quality=quality,
                    environment=environment,
                )
            )
        except skipped_errors as error:
            records.append(
                TuningRecord(
                    tuning=tuning,
                    candidate=candidate.name,
                    shape=shape,
                    configuration=configuration,
                    phase=phase,
                    status=TuningStatus.SKIPPED,
                    selected=False,
                    warmup_ms=warmup_ms,
                    measurement_time_ms=measurement_time_ms,
                    reason=_reason(error),
                    environment=environment,
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
