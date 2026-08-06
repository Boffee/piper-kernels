"""Callable benchmark providers with explicit prepare and run phases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from .timing import PhaseTimings, Timing, measure_first_call, triton_benchmark

PreparedT = TypeVar("PreparedT")
OutputT = TypeVar("OutputT")


class DistributionTimer(Protocol):
    """A timing adapter compatible with :func:`triton_benchmark`."""

    def __call__(
        self,
        function: Callable[[], Any],
        warmup_ms: int,
        repeat_ms: int,
    ) -> Timing: ...


@dataclass(slots=True)
class BenchmarkProvider(Generic[PreparedT, OutputT]):
    """One implementation of an operation under benchmark.

    Preparation may quantize, pack, or transform inputs. Run receives the
    prepared value and should contain only the hot kernel path.
    """

    name: str
    prepare: Callable[[], PreparedT]
    run: Callable[[PreparedT], OutputT]
    synchronize: Callable[[], None] = lambda: None
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def complete(self) -> OutputT:
        """Run preparation and the kernel as one complete operator call."""
        return self.run(self.prepare())


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
    repeat_ms: int,
    timer: DistributionTimer = triton_benchmark,
    measure_compilation: bool = True,
    measure_preparation: bool = True,
    measure_complete: bool = True,
) -> ProviderMeasurement[OutputT]:
    """Measure all requested phases of one provider.

    The cold call is intentionally performed before any warmed phase. The
    returned output is produced from the same prepared inputs used by the
    kernel-only measurement so callers can compute quality afterward.
    """
    compilation_ms = None
    if measure_compilation:
        _, compilation_ms = measure_first_call(provider.complete, provider.synchronize)

    prepared = provider.prepare()
    preparation = (
        timer(provider.prepare, warmup_ms, repeat_ms) if measure_preparation else None
    )
    kernel = timer(lambda: provider.run(prepared), warmup_ms, repeat_ms)
    complete = timer(provider.complete, warmup_ms, repeat_ms) if measure_complete else None
    output = provider.run(prepared)
    provider.synchronize()

    return ProviderMeasurement(
        provider=provider.name,
        output=output,
        timings=PhaseTimings(
            warmup_ms=warmup_ms,
            repeat_ms=repeat_ms,
            compilation_ms=compilation_ms,
            preparation=preparation,
            kernel=kernel,
            complete=complete,
        ),
        configuration=dict(provider.configuration),
    )
