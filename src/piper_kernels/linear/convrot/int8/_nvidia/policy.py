"""Measured NVIDIA ConvRot INT8 preparation and GEMM schedules.

SM120 and SM8x select GEMM schedules independently of preparation. SM8x thresholds
are measured on SM89; other supported NVIDIA targets retain the baseline schedule.
"""

from dataclasses import replace

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks

from ._plan import MatmulSchedule, NvidiaExecutionPlan

# Measured crossovers in useful tiles, not exact model shapes or sequence lengths.
_SM120_SMALL_MAX_TILES = 128
_SM120_LARGE_MIN_TILES = 72
_SM8X_SMALL_MAX_TILES = 96
_SM8X_LARGE_MIN_TILES = 64
_SM8X_GLUON_MIN_TILES = 48

# Complete schedules: M, N, K, warps, stages, grouping, and implementation options.
_BASE_MATMUL = MatmulSchedule(128, 256, 128, 8, 3, 16)
_SM120_SMALL = MatmulSchedule(32, 64, 128, 8, 4, 0)
_SM120_MEDIUM = MatmulSchedule(64, 64, 128, 4, 3, 0)
_SM8X_SMALL = MatmulSchedule(
    16, 64, 128, 4, 4, 0, triton_specialize_m=False, triton_explicit_bias_fma=True
)
_SM8X_MEDIUM = MatmulSchedule(
    64, 64, 128, 4, 4, 0, triton_specialize_m=False, triton_explicit_bias_fma=True
)
_GROUPED_TRITON = MatmulSchedule(
    128, 64, 128, 4, 3, 16, triton_specialize_m=False, triton_explicit_bias_fma=True
)
_SM8X_GLUON_MEDIUM = MatmulSchedule(
    128,
    128,
    64,
    4,
    3,
    8,
    "gluon_async_copy",
    triton_specialize_m=False,
    triton_explicit_bias_fma=True,
)
_SM8X_GLUON_LARGE = MatmulSchedule(
    256,
    128,
    64,
    8,
    4,
    8,
    "gluon_async_copy",
    triton_specialize_m=False,
    triton_explicit_bias_fma=True,
)


def supports_target(target: AcceleratorTarget) -> bool:
    """Return whether the NVIDIA INT8 linear and update implementation is supported."""
    return target.cuda_capability_at_least(7, 5)


def supports_preparation_target(target: AcceleratorTarget) -> bool:
    """Return whether rotation and quantization can use the NVIDIA implementation."""
    return target.is_nvidia_cuda


def _preparation_schedule(in_features: int, *, sm8x: bool) -> tuple[bool, int]:
    """Select fusion and warp count without consulting rows or output width."""
    fused_chunks = fused_preparation_chunks(in_features)
    fused_num_warps = 4
    if sm8x and in_features <= 1024:
        # One warp holds a short row in registers without cross-warp reductions.
        fused_num_warps = 1
    elif fused_chunks is not None:
        chunk_count, chunk_size = fused_chunks
        if chunk_count > 1 and chunk_size <= 2048:
            fused_num_warps = 2
        elif chunk_size == 16384:
            fused_num_warps = 8
    return fused_chunks is not None, fused_num_warps


def _sm120_matmul_schedule(rows: int | None, out_features: int | None) -> MatmulSchedule:
    """Select one of three measured tiles using useful work and available parallelism."""
    if not rows or not out_features:
        return _BASE_MATMUL
    small_tiles = ((rows + 31) // 32) * ((out_features + 63) // 64)
    if rows <= 32 or small_tiles <= _SM120_SMALL_MAX_TILES:
        return _SM120_SMALL
    large_columns = (out_features + 255) // 256
    # A one-row tail is not a full tile; wide tiles cannot reuse a single 64-column tile.
    if out_features <= 64 or rows * large_columns < 128 * _SM120_LARGE_MIN_TILES:
        return _SM120_MEDIUM
    return _BASE_MATMUL


def _sm8x_matmul_schedule(rows: int | None, out_features: int | None) -> MatmulSchedule:
    """Select Triton for short/narrow work and Gluon when enough wide tiles are useful.

    Each output width uses at most three schedules as rows increase. All SM8x
    schedules use dynamic M, bounding the number of compiled GEMMs per layer.
    """
    if not rows or not out_features:
        return _SM8X_GLUON_LARGE
    column_tiles = (out_features + 63) // 64
    small_tiles = ((rows + 15) // 16) * column_tiles
    if rows <= 16 or small_tiles <= _SM8X_SMALL_MAX_TILES:
        return _SM8X_SMALL
    if out_features > 128:
        # Up to 1024 columns, 256-row tiles leave too few CTAs; use 128-row tiles.
        gluon = _SM8X_GLUON_MEDIUM if out_features <= 1024 else _SM8X_GLUON_LARGE
        wide_columns = (out_features + 127) // 128
        if rows >= 256 and rows * wide_columns >= gluon.matmul_block_m * _SM8X_GLUON_MIN_TILES:
            return gluon
    elif rows * column_tiles >= 128 * _SM8X_LARGE_MIN_TILES:
        return _GROUPED_TRITON
    return _SM8X_MEDIUM


def _execution_plan(
    in_features: int, matmul: MatmulSchedule, *, sm8x: bool = False
) -> NvidiaExecutionPlan:
    """Combine independently selected preparation and GEMM choices once."""
    fused, fused_num_warps = _preparation_schedule(in_features, sm8x=sm8x)
    return NvidiaExecutionPlan(
        fuse_rotation_quantization=fused,
        fused_num_warps=fused_num_warps,
        rotation_num_warps=4,
        quantization_num_warps=8,
        **matmul._asdict(),
    )


def baseline_execution_plan(*, in_features: int) -> NvidiaExecutionPlan:
    """Build the historical fixed schedule for explicit benchmark comparisons."""
    return _execution_plan(in_features, _BASE_MATMUL)


def grouped_triton_plan(plan: NvidiaExecutionPlan) -> NvidiaExecutionPlan:
    """Keep preparation while selecting the rounding-compatible alignment fallback."""
    return replace(plan, **_GROUPED_TRITON._asdict())


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    in_features: int,
    rows: int | None = None,
    out_features: int | None = None,
) -> NvidiaExecutionPlan:
    """Select preparation and a concrete GEMM schedule, then build one execution plan."""
    if not supports_target(target):
        raise ValueError(f"ConvRot INT8 execution has no optimized policy for {target}")
    sm8x = target.is_cuda_capability(8)
    if sm8x:
        matmul = _sm8x_matmul_schedule(rows, out_features)
    elif target.is_architecture("sm120"):
        matmul = _sm120_matmul_schedule(rows, out_features)
    else:
        matmul = _BASE_MATMUL
    return _execution_plan(in_features, matmul, sm8x=sm8x)
