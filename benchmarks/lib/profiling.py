"""Provider launch loops for external GPU profilers."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import torch

from .providers import BenchmarkProvider, ProviderPhase, provider_phase_launch

PreparedT = TypeVar("PreparedT")
OutputT = TypeVar("OutputT")


class CaptureController(Protocol):
    """Profiler capture and annotation operations used by the launch loop."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def push_range(self, name: str) -> None: ...

    def pop_range(self) -> None: ...


class CudaProfilerController:
    """Control CUDA profiler API capture and annotate it with NVTX."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiler capture requires an available NVIDIA GPU")
        if getattr(torch.version, "hip", None) is not None:
            raise RuntimeError(
                "CUDA profiler/NVTX capture is unavailable on ROCm; use a future "
                "ROCTracer/ROCTx capture controller with the generic launch loop"
            )
        if torch.version.cuda is None:
            raise RuntimeError("PyTorch was not built with NVIDIA CUDA support")

    def start(self) -> None:
        """Start a CUDA profiler API capture."""
        torch.cuda.profiler.start()

    def stop(self) -> None:
        """Stop a CUDA profiler API capture."""
        torch.cuda.profiler.stop()

    def push_range(self, name: str) -> None:
        """Push an NVTX range."""
        torch.cuda.nvtx.range_push(name)

    def pop_range(self) -> None:
        """Pop the current NVTX range."""
        torch.cuda.nvtx.range_pop()


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Summary of a completed provider capture."""

    provider: str
    phase: ProviderPhase
    iterations: int
    warmup_iterations: int
    range_name: str
    include_setup: bool


def _run_range(
    controller: CaptureController,
    name: str,
    function: Callable[[], object],
    iterations: int,
    synchronize: Callable[[], None],
) -> None:
    controller.push_range(name)
    try:
        for _ in range(iterations):
            function()
    finally:
        try:
            synchronize()
        finally:
            controller.pop_range()


def profile_provider(
    provider: BenchmarkProvider[PreparedT, OutputT],
    *,
    iterations: int,
    warmup_iterations: int = 5,
    phase: ProviderPhase = ProviderPhase.OPERATOR_END_TO_END,
    range_name: str = "profile",
    include_setup: bool = False,
    controller: CaptureController | None = None,
) -> ProviderProfile:
    """Launch a provider inside profiler and annotation capture ranges.

    By default, one compilation call and all warmup calls complete before the
    CUDA profiler starts.  ``include_setup=True`` explicitly places them in a
    separate ``<range_name>/setup`` range within the same capture.
    """
    if iterations <= 0:
        raise ValueError("profile iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("profile warmup iterations cannot be negative")
    if not range_name:
        raise ValueError("profile range name cannot be empty")

    launch = provider_phase_launch(provider, phase)

    capture = controller or CudaProfilerController()
    if not include_setup:
        launch()  # Compile and initialize lazy state outside the capture.
        for _ in range(warmup_iterations):
            launch()
        provider.synchronize()

    capture.start()
    try:
        if include_setup:
            _run_range(
                capture,
                f"{range_name}/setup",
                launch,
                warmup_iterations + 1,
                provider.synchronize,
            )
        _run_range(capture, range_name, launch, iterations, provider.synchronize)
    finally:
        capture.stop()

    return ProviderProfile(
        provider=provider.name,
        phase=phase,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        range_name=range_name,
        include_setup=include_setup,
    )


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional external-profiler capture arguments to a provider CLI."""
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run the provider in CUDA profiler API and NVTX capture ranges",
    )
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--profile-warmup-iterations", type=int, default=5)
    parser.add_argument(
        "--profile-phase",
        type=ProviderPhase,
        choices=tuple(ProviderPhase),
        default=ProviderPhase.OPERATOR_END_TO_END,
    )
    parser.add_argument("--profile-range-name", default="profile")
    parser.add_argument(
        "--profile-include-setup",
        action="store_true",
        help="include the compilation call and warmup in a separate setup range",
    )
