"""Benchmark providers with explicit preparation and execution phases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import Any, Generic, Protocol, TypeVar

from .timing import (
    PhaseTimings,
    Timing,
    synchronized_wall_benchmark,
    time_first_call,
    triton_benchmark,
)

PreparedT = TypeVar("PreparedT")
OutputT = TypeVar("OutputT")


class ProviderPhase(StrEnum):
    """A provider execution boundary shared by timing, tuning, and profiling."""

    PREPARED_EXECUTION = "prepared_execution"
    OPERATOR_END_TO_END = "operator_end_to_end"


class DistributionTimer(Protocol):
    """A timing adapter for one benchmark phase."""

    def __call__(
        self,
        function: Callable[[], Any],
        warmup_ms: int,
        measurement_time_ms: int,
    ) -> Timing: ...


@dataclass(slots=True)
class BenchmarkProvider(Generic[PreparedT, OutputT]):
    """One implementation of an operation under benchmark.

    Preparation may quantize, pack, or transform inputs. Run receives the
    prepared value and should contain only the prepared execution path.
    ``triton_jit_functions`` maps stable report names to JIT functions launched by
    the provider so compiler inspection remains independent of the operation.
    """

    name: str
    prepare: Callable[[], PreparedT]
    run: Callable[[PreparedT], OutputT]
    synchronize: Callable[[], None] = lambda: None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    triton_jit_functions: Mapping[str, object] = field(default_factory=dict)

    def run_operator(self) -> OutputT:
        """Run preparation and execution as one end-to-end operator call."""
        return self.run(self.prepare())


def provider_phase_launch(
    provider: BenchmarkProvider[PreparedT, OutputT],
    phase: ProviderPhase,
) -> Callable[[], OutputT]:
    """Bind one provider phase to the exact launch callable used by every tool."""
    if phase is ProviderPhase.PREPARED_EXECUTION:
        return partial(provider.run, provider.prepare())
    if phase is ProviderPhase.OPERATOR_END_TO_END:
        return provider.run_operator
    raise ValueError(f"unsupported provider phase {phase!r}")


@dataclass(frozen=True, slots=True)
class ProviderMeasurement(Generic[OutputT]):
    """A provider's representative output and standardized phase timings."""

    provider: str
    output: OutputT
    timings: PhaseTimings
    configuration: Mapping[str, Any]


def measure_provider(
    provider: BenchmarkProvider[PreparedT, OutputT],
    *,
    warmup_ms: int,
    measurement_time_ms: int,
    device_timer: DistributionTimer = triton_benchmark,
    wall_timer: DistributionTimer | None = None,
    measure_first_call: bool = True,
    measure_preparation: bool = True,
    measure_operator_end_to_end: bool = True,
) -> ProviderMeasurement[OutputT]:
    """Measure all requested phases of one provider.

    The first call is intentionally performed before any warmed phase. The
    returned output is produced from the same prepared inputs used by the
    prepared-execution measurement so callers can compute quality afterward.
    """
    operator_launch = provider_phase_launch(provider, ProviderPhase.OPERATOR_END_TO_END)
    first_call_ms = None
    if measure_first_call:
        first_output, first_call_ms = time_first_call(operator_launch, provider.synchronize)
        del first_output

    def default_wall_timer(
        function: Callable[[], Any],
        phase_warmup_ms: int,
        phase_measurement_time_ms: int,
    ) -> Timing:
        return synchronized_wall_benchmark(
            function,
            phase_warmup_ms,
            phase_measurement_time_ms,
            synchronize=provider.synchronize,
        )

    resolved_wall_timer = wall_timer or default_wall_timer
    prepared_launch = provider_phase_launch(provider, ProviderPhase.PREPARED_EXECUTION)
    preparation = (
        resolved_wall_timer(provider.prepare, warmup_ms, measurement_time_ms)
        if measure_preparation
        else None
    )
    prepared_execution = device_timer(
        prepared_launch,
        warmup_ms,
        measurement_time_ms,
    )
    operator_end_to_end = (
        resolved_wall_timer(
            operator_launch,
            warmup_ms,
            measurement_time_ms,
        )
        if measure_operator_end_to_end
        else None
    )
    output = prepared_launch()
    provider.synchronize()

    return ProviderMeasurement(
        provider=provider.name,
        output=output,
        timings=PhaseTimings(
            warmup_ms=warmup_ms,
            measurement_time_ms=measurement_time_ms,
            first_call_ms=first_call_ms,
            preparation=preparation,
            prepared_execution=prepared_execution,
            operator_end_to_end=operator_end_to_end,
        ),
        configuration=dict(provider.configuration),
    )
