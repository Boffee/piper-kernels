"""Consistent timing terminology and Triton timing integration."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class Timing:
    """Median latency with a central 60% interval, in milliseconds."""

    median_ms: float
    p20_ms: float
    p80_ms: float

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

    def as_dict(self) -> dict[str, float]:
        """Return stable machine-readable field names."""
        return {
            "median_ms": self.median_ms,
            "p20_ms": self.p20_ms,
            "p80_ms": self.p80_ms,
        }


@dataclass(frozen=True, slots=True)
class PhaseTimings:
    """Standard benchmark phases.

    ``compilation_ms`` is the synchronized first complete invocation, including
    any lazy compilation. ``preparation`` contains preprocessing or packing,
    ``kernel`` runs on already-prepared inputs, and ``complete`` includes both.
    """

    warmup_ms: int
    repeat_ms: int
    compilation_ms: float | None
    preparation: Timing | None
    kernel: Timing
    complete: Timing | None

    def __post_init__(self) -> None:
        if self.warmup_ms < 0 or self.repeat_ms <= 0:
            raise ValueError("warmup must be non-negative and repeat must be positive")

    def as_dict(self) -> dict[str, float | dict[str, float] | None]:
        """Return stable machine-readable field names."""
        return {
            "warmup_ms": self.warmup_ms,
            "repeat_ms": self.repeat_ms,
            "compilation_ms": self.compilation_ms,
            "preparation": None if self.preparation is None else self.preparation.as_dict(),
            "kernel": self.kernel.as_dict(),
            "complete": None if self.complete is None else self.complete.as_dict(),
        }


def measure_first_call(
    function: Callable[[], Any],
    synchronize: Callable[[], None] | None = None,
) -> tuple[Any, float]:
    """Run and synchronize one cold invocation, returning output and wall time."""
    sync = synchronize or (lambda: None)
    sync()
    started = time.perf_counter()
    output = function()
    sync()
    return output, (time.perf_counter() - started) * 1_000


def triton_benchmark(
    function: Callable[[], Any],
    warmup_ms: int,
    repeat_ms: int,
) -> Timing:
    """Measure a callable with Triton's adaptive GPU benchmark helper."""
    if warmup_ms < 0 or repeat_ms <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")

    testing = cast(_TritonTesting, importlib.import_module("triton.testing"))
    median, p20, p80 = testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=repeat_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return Timing(float(median), float(p20), float(p80))
