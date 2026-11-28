"""AMD RDNA4 launch schedules and accumulator policy for ConvRot INT8 Conv3D."""

from __future__ import annotations

import sys

from piper_kernels._triton.targets import (
    GFX120X_ARCHITECTURES,
    RDNA4_MIN_FULL_GRID_TILES,
    AcceleratorTarget,
    rdna4_int8_accumulator_seed,
)

from .._schedule import (
    ConvolutionExecutionPlan,
    ConvolutionSchedule,
    PreparationSchedule,
    tile_count,
)


def supports_target(target: AcceleratorTarget) -> bool:
    """gfx1201 has runtime coverage; gfx1200 also has offline compilation coverage."""
    return sys.platform in ("linux", "win32") and target.is_hip_architecture(*GFX120X_ARCHITECTURES)


def _convolution_schedule(outputs: int, rows: int) -> ConvolutionSchedule:
    """Return the measured RDNA4 Conv3D convolution schedule for ``outputs`` and ``rows``."""
    # Fill at least two workgroups per WGP (64 tiles) across 32-WGP RDNA4 while
    # requiring 256-column outputs to be divisible so BN=256 never wastes a half tile.
    if (
        rows >= 128
        and outputs >= 256
        and outputs % 256 == 0
        and tile_count(rows, outputs, 128, 256) >= RDNA4_MIN_FULL_GRID_TILES
    ):
        return ConvolutionSchedule(128, 256, 64, 8, 2)
    if (
        rows >= 128
        and outputs >= 128
        and tile_count(rows, outputs, 128, 128) >= RDNA4_MIN_FULL_GRID_TILES
    ):
        return ConvolutionSchedule(128, 128, 64, 8, 2)
    if (
        rows >= 64
        and outputs >= 128
        and tile_count(rows, outputs, 64, 128) >= RDNA4_MIN_FULL_GRID_TILES
    ):
        return ConvolutionSchedule(64, 128, 128, 4, 2)
    if outputs >= 64 and rows >= 128:
        return ConvolutionSchedule(64, 64, 64, 4, 2)
    return ConvolutionSchedule(32, 64, 64, 4, 2)


convolution_schedule = _convolution_schedule


def preparation_schedule(channels: int) -> PreparationSchedule:
    """Bound the FP32 rotation tile independently of the convolution output width."""
    return PreparationSchedule(max(1, min(16, 4096 // channels)), 4)


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    channels: int,
    outputs: int,
    output_rows: int,
    convolution_schedule: ConvolutionSchedule | None = None,
) -> ConvolutionExecutionPlan:
    """Resolve AMD RDNA4 preparation, convolution, and accumulator-seed execution choices."""
    schedule = (
        _convolution_schedule(outputs, output_rows)
        if convolution_schedule is None
        else convolution_schedule
    )
    return ConvolutionExecutionPlan(
        preparation=preparation_schedule(channels),
        convolution=schedule,
        use_weight_descriptor=False,
        lds_only_fences=True,
        accumulator_seed=rdna4_int8_accumulator_seed(
            target,
            27 * channels,
            tile_count(output_rows, outputs, schedule.block_m, schedule.block_n),
        ),
    )
