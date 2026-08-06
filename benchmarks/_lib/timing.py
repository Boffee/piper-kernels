"""Consistent timing terminology and Triton timing integration."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast


class _TritonTesting(Protocol):
    def do_bench(
        self,
        function: Callable[[], Any],
        *,
        warmup: int,
        rep: int,
        quantiles: list[float],
    ) -> Sequence[float]: ...


class ClockDomain(StrEnum):
    """Clock domains used by benchmark timing implementations."""

    DEVICE_EVENT = "device_event"
    SYNCHRONIZED_WALL = "synchronized_wall"


@dataclass(frozen=True, slots=True)
class Timing:
    """Median latency with a central 60% interval, in milliseconds."""

    median_ms: float
    p20_ms: float
    p80_ms: float
    clock: ClockDomain

    def __post_init__(self) -> None:
        if min(self.median_ms, self.p20_ms, self.p80_ms) < 0:
            raise ValueError("latencies cannot be negative")
        if not self.p20_ms <= self.median_ms <= self.p80_ms:
            raise ValueError("timing quantiles must satisfy p20 <= median <= p80")

    def display(self, precision: int = 3) -> str:
        """Format p50 followed by the p20/p80 interval."""
        return (
            f"{self.median_ms:.{precision}f} "
            f"[{self.p20_ms:.{precision}f}, {self.p80_ms:.{precision}f}]"
        )

    def as_dict(self) -> dict[str, float | str]:
        """Return stable machine-readable field names."""
        return {
            "median_ms": self.median_ms,
            "p20_ms": self.p20_ms,
            "p80_ms": self.p80_ms,
            "clock": self.clock.value,
        }


@dataclass(frozen=True, slots=True)
class PhaseTimings:
    """Standard benchmark phases.

    ``first_call_ms`` is the synchronized first operator invocation, including
    any lazy compilation. ``preparation`` and ``operator_end_to_end`` use a
    synchronized wall clock so they include host work. ``prepared_execution``
    uses device events on already-prepared inputs.
    """

    warmup_ms: int
    measurement_time_ms: int
    first_call_ms: float | None
    preparation: Timing | None
    prepared_execution: Timing
    operator_end_to_end: Timing | None

    def __post_init__(self) -> None:
        if self.warmup_ms < 0 or self.measurement_time_ms <= 0:
            raise ValueError(
                "warmup must be non-negative and measurement time must be positive"
            )

    def as_dict(self) -> dict[str, float | str | dict[str, float | str] | None]:
        """Return stable machine-readable field names."""
        return {
            "warmup_ms": self.warmup_ms,
            "measurement_time_ms": self.measurement_time_ms,
            "first_call_ms": self.first_call_ms,
            "first_call_clock": (
                None
                if self.first_call_ms is None
                else ClockDomain.SYNCHRONIZED_WALL.value
            ),
            "preparation": None if self.preparation is None else self.preparation.as_dict(),
            "prepared_execution": self.prepared_execution.as_dict(),
            "operator_end_to_end": (
                None
                if self.operator_end_to_end is None
                else self.operator_end_to_end.as_dict()
            ),
        }


def time_first_call(
    function: Callable[[], Any],
    synchronize: Callable[[], None] | None = None,
) -> tuple[Any, float]:
    """Run and synchronize the first invocation, returning output and wall time."""
    sync = synchronize or (lambda: None)
    sync()
    started = time.perf_counter()
    output = function()
    sync()
    return output, (time.perf_counter() - started) * 1_000


def _linear_quantile(ordered_values: Sequence[float], quantile: float) -> float:
    """Interpolate one quantile from an ordered, non-empty sample."""
    position = (len(ordered_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = ordered_values[lower_index]
    upper = ordered_values[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def synchronized_wall_benchmark(
    function: Callable[[], Any],
    warmup_ms: int,
    measurement_time_ms: int,
    *,
    synchronize: Callable[[], None] | None = None,
) -> Timing:
    """Measure host and device latency with a synchronized wall clock."""
    if warmup_ms < 0 or measurement_time_ms <= 0:
        raise ValueError(
            "warmup must be non-negative and measurement time must be positive"
        )

    sync = synchronize or (lambda: None)
    sync()
    warmup_started = time.perf_counter()
    while (time.perf_counter() - warmup_started) * 1_000 < warmup_ms:
        function()
        sync()

    samples: list[float] = []
    measured_ms = 0.0
    while measured_ms < measurement_time_ms:
        started = time.perf_counter()
        function()
        sync()
        elapsed_ms = (time.perf_counter() - started) * 1_000
        samples.append(elapsed_ms)
        measured_ms += elapsed_ms

    samples.sort()
    return Timing(
        median_ms=_linear_quantile(samples, 0.5),
        p20_ms=_linear_quantile(samples, 0.2),
        p80_ms=_linear_quantile(samples, 0.8),
        clock=ClockDomain.SYNCHRONIZED_WALL,
    )


def triton_benchmark(
    function: Callable[[], Any],
    warmup_ms: int,
    measurement_time_ms: int,
) -> Timing:
    """Measure device-stream latency with Triton's GPU-event benchmark helper."""
    if warmup_ms < 0 or measurement_time_ms <= 0:
        raise ValueError(
            "warmup must be non-negative and measurement time must be positive"
        )

    testing = cast(_TritonTesting, importlib.import_module("triton.testing"))
    median, p20, p80 = testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=measurement_time_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return Timing(
        float(median),
        float(p20),
        float(p80),
        clock=ClockDomain.DEVICE_EVENT,
    )
