"""NVIDIA SM120 and SM8x launch schedules and descriptor policy for ConvRot INT8 Conv3D."""

from __future__ import annotations

from piper_kernels._triton.targets import AcceleratorTarget

from .._schedule import (
    ConvolutionExecutionPlan,
    ConvolutionSchedule,
    PreparationSchedule,
    tile_count,
)

# Fill SM120 with 128-row tiles before falling back to 64-row tiles for small grids.
_SM120_MIN_FULL_GRID_TILES = 128
_PREPARE_C256_MIN_ROWS = 200_000
_PREPARE_C512_MIN_ROWS = 5_000

# SM89 measurements favor narrower tiles below 32 row tiles of 64.
_SM8X_WIDE_MIN_ROWS = 2048

_BASE_CONVOLUTION = ConvolutionSchedule(64, 128, 128, 4, 3)
_SM120_K64 = ConvolutionSchedule(128, 128, 64, 4, 3)
_SM120_NARROW_WIDE_M = ConvolutionSchedule(128, 64, 64, 4, 3)
_SM120_NARROW_OUTPUT = ConvolutionSchedule(64, 64, 64, 4, 3)
_SM8X_SHORT = ConvolutionSchedule(64, 64, 128, 4, 3)
_SM8X_NARROW_OUTPUT = ConvolutionSchedule(32, 32, 128, 2, 3)

_PREPARE_DEFAULT = PreparationSchedule(8, 8)
_PREPARE_WIDE = PreparationSchedule(64, 4)
_PREPARE_MEDIUM = PreparationSchedule(32, 4)
_PREPARE_SHORT = PreparationSchedule(16, 4)
_PREPARE_MEDIUM_8_WARPS = PreparationSchedule(32, 8)
_PREPARE_SHORT_8_WARPS = PreparationSchedule(16, 8)


def supports_target(target: AcceleratorTarget) -> bool:
    """Enable exact SM120 and the SM8x family (SM80, SM86, SM87, SM89)."""
    return target.is_cuda_capability(12, 0) or target.is_cuda_capability(8)


def _sm120_use_weight_descriptor(
    channels: int, outputs: int, height: int, block_n: int, *, aligned: bool
) -> bool:
    # TMA saves weight-addressing overhead on C=128/256 shorter K-loops; C=64 and
    # C>=512 prefer pointer loads.
    return (
        aligned
        and channels in (128, 256)
        and outputs % block_n == 0
        and (height >= 256 or outputs > 128)
    )


def _sm120_convolution_schedule(outputs: int, rows: int) -> ConvolutionSchedule:
    if outputs <= 64:
        return (
            _SM120_NARROW_WIDE_M
            if rows >= 128 and tile_count(rows, outputs, 128, 64) >= _SM120_MIN_FULL_GRID_TILES
            else _SM120_NARROW_OUTPUT
        )
    return (
        _SM120_K64
        if rows >= 128 and tile_count(rows, outputs, 128, 128) >= _SM120_MIN_FULL_GRID_TILES
        else _BASE_CONVOLUTION
    )


def _sm8x_convolution_schedule(outputs: int, rows: int) -> ConvolutionSchedule:
    if outputs <= 64:
        return _SM8X_NARROW_OUTPUT
    return _BASE_CONVOLUTION if rows >= _SM8X_WIDE_MIN_ROWS else _SM8X_SHORT


def _preparation_schedule(channels: int, rows: int, *, group_norm: bool) -> PreparationSchedule:
    # SM120 and SM8x share bandwidth-bound preparation schedules; C=64 and C=512 were
    # updated from SM120 measurements and verified spill-free on sm_89 offline.
    if channels <= 128:
        return _PREPARE_WIDE
    if channels == 256:
        if group_norm:
            return _PREPARE_MEDIUM if rows >= _PREPARE_C256_MIN_ROWS else _PREPARE_SHORT
        return _PREPARE_WIDE if rows >= _PREPARE_C256_MIN_ROWS else _PREPARE_MEDIUM_8_WARPS
    if channels == 512 and rows >= _PREPARE_C512_MIN_ROWS:
        return _PREPARE_SHORT_8_WARPS
    return _PREPARE_DEFAULT


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    channels: int,
    outputs: int,
    input_rows: int,
    output_rows: int,
    output_height: int,
    weight_aligned: bool,
    group_norm: bool,
    convolution_schedule: ConvolutionSchedule | None = None,
) -> ConvolutionExecutionPlan:
    """Resolve NVIDIA preparation, convolution, and weight-descriptor execution choices."""
    sm8x = target.is_cuda_capability(8)
    if convolution_schedule is None:
        convolution_schedule = (
            _sm8x_convolution_schedule(outputs, output_rows)
            if sm8x
            else _sm120_convolution_schedule(outputs, output_rows)
        )
    return ConvolutionExecutionPlan(
        preparation=_preparation_schedule(channels, input_rows, group_norm=group_norm),
        convolution=convolution_schedule,
        use_weight_descriptor=not sm8x
        and _sm120_use_weight_descriptor(
            channels,
            outputs,
            output_height,
            convolution_schedule.block_n,
            aligned=weight_aligned,
        ),
    )
