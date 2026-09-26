"""Support and measured launch policy for the NVIDIA ConvRot INT8 implementation.

These schedules are measured on SM120 and SM89; the SM89 schedule covers the SM8x
family. Other supported NVIDIA targets retain the existing defaults. Hardware support
does not imply per-target tuning.
"""

from dataclasses import dataclass, replace
from typing import ClassVar

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks

from .._plan import LinearExecutionPlan

_FUSED_NUM_WARPS_VALUES = (2, 4, 8, 16)
_SM8X_FUSED_NUM_WARPS_VALUES = (1, *_FUSED_NUM_WARPS_VALUES)
_ROTATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_QUANTIZATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_MATMUL_BLOCK_M_VALUES = (16, 32, 64, 128)
_SM8X_MATMUL_BLOCK_M_VALUES = (*_MATMUL_BLOCK_M_VALUES, 256)
_MATMUL_BLOCK_N_VALUES = (16, 32, 64, 128, 256)
_MATMUL_BLOCK_K_VALUES = (32, 64, 128)
_MATMUL_NUM_WARPS_VALUES = (2, 4, 8)
_MATMUL_NUM_STAGES_VALUES = (1, 2, 3, 4)
_MATMUL_GROUP_M_VALUES = (0, 8, 16)
# SM8x runs exactly these tiles (block_m, block_n, block_k, warps) with the Gluon GEMM in
# ``gluon.py``: 64x64 warp tiles over 64-column K tiles. Every other tile runs Triton.
_GLUON_TILES = ((128, 128, 64, 4), (256, 128, 64, 8))
_GLUON_NUM_STAGES_VALUES = (3, 4)


def _choices(values: tuple[int, ...]) -> str:
    *leading, last = values
    return f"{', '.join(map(str, leading))}, or {last}"


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan(LinearExecutionPlan):
    """Launch choices accepted by the existing NVIDIA kernels and tuner."""

    fused_num_warps_values: ClassVar[tuple[int, ...]] = _FUSED_NUM_WARPS_VALUES
    matmul_block_m_values: ClassVar[tuple[int, ...]] = _MATMUL_BLOCK_M_VALUES

    def __post_init__(self) -> None:
        LinearExecutionPlan.__post_init__(self)
        if self.fused_num_warps not in self.fused_num_warps_values:
            choices = _choices(self.fused_num_warps_values)
            raise ValueError(f"ConvRot fused preparation num_warps must be {choices}")
        if self.rotation_num_warps not in _ROTATION_NUM_WARPS_VALUES:
            raise ValueError("ConvRot split rotation num_warps must be 1, 2, 4, or 8")
        if self.quantization_num_warps not in _QUANTIZATION_NUM_WARPS_VALUES:
            raise ValueError("ConvRot split quantization num_warps must be 1, 2, 4, or 8")
        if self.matmul_block_m not in self.matmul_block_m_values:
            raise ValueError(
                f"ConvRot matmul block_m must be {_choices(self.matmul_block_m_values)}"
            )
        if self.matmul_block_n not in _MATMUL_BLOCK_N_VALUES:
            raise ValueError("ConvRot matmul block_n must be 16, 32, 64, 128, or 256")
        if self.matmul_block_k not in _MATMUL_BLOCK_K_VALUES:
            raise ValueError("ConvRot matmul block_k must be 32, 64, or 128")
        if self.matmul_num_warps not in _MATMUL_NUM_WARPS_VALUES:
            raise ValueError("ConvRot matmul num_warps must be 2, 4, or 8")
        if self.matmul_num_stages not in _MATMUL_NUM_STAGES_VALUES:
            raise ValueError("ConvRot matmul num_stages must be 1, 2, 3, or 4")


@dataclass(frozen=True, slots=True)
class Sm8xExecutionPlan(NvidiaExecutionPlan):
    """SM8x launch choices with explicit GEMM grouping.

    The base 128x256 tile spills on SM8x. The 128x128x64 and 256x128x64 tiles run the
    Gluon GEMM in ``gluon.py``; other tiles run the shared Triton kernel, so replacing a
    plan's tile also selects its kernel. Other plans derive grouping from the 128x256 tile,
    which SM120 schedules keep unchanged. SM8x launches also write bias adds as explicit
    FMAs. One-warp fused preparation is measured only on SM8x.
    """

    fused_num_warps_values: ClassVar[tuple[int, ...]] = _SM8X_FUSED_NUM_WARPS_VALUES
    matmul_block_m_values: ClassVar[tuple[int, ...]] = _SM8X_MATMUL_BLOCK_M_VALUES
    matmul_group_m: int = 0

    def __post_init__(self) -> None:
        NvidiaExecutionPlan.__post_init__(self)
        if self.matmul_group_m not in _MATMUL_GROUP_M_VALUES:
            raise ValueError("ConvRot matmul group_m must be 0, 8, or 16")
        if self.matmul_kernel == "gluon":
            if self.matmul_num_stages not in _GLUON_NUM_STAGES_VALUES:
                raise ValueError("ConvRot SM8x Gluon tiles use 3 or 4 stages")
        elif self.matmul_block_m not in _MATMUL_BLOCK_M_VALUES:
            raise ValueError(
                "ConvRot SM8x 256-row tiles must be 256x128x64 Gluon tiles with 8 warps"
            )

    @property
    def matmul_kernel(self) -> str:
        """Return ``gluon`` for the Gluon GEMM tiles and ``triton`` otherwise."""
        tile = (
            self.matmul_block_m,
            self.matmul_block_n,
            self.matmul_block_k,
            self.matmul_num_warps,
        )
        return "gluon" if tile in _GLUON_TILES else "triton"


_FUSED_MAX_CHUNK_SIZE = 16_384
_TWO_WARP_MAX_CHUNK_SIZE = 2_048
_DEFAULT_ROTATION_NUM_WARPS = 4
_DEFAULT_QUANTIZATION_NUM_WARPS = 8
_SM120_SMALL_TILE_LIMIT = 128
_SM120_LARGE_TILE_THRESHOLD = 72
_SM8X_SMALL_ROWS = 16
_SM8X_SMALL_TILE_LIMIT = 96
_SM8X_LARGE_TILE_THRESHOLD = 64
# Gluon tiles need at least two 128-row tiles and more than one 128-column tile.
_SM8X_GLUON_MIN_ROWS = 256
_SM8X_GLUON_MIN_COLUMNS = 129
_SM8X_GLUON_MEDIUM_THRESHOLD = 48
_SM8X_GLUON_LARGE_THRESHOLD = 128
_SM8X_ONE_WARP_MAX_CHUNK_SIZE = 1_024


def supports_target(target: AcceleratorTarget) -> bool:
    """Return whether the NVIDIA INT8 linear and update implementation is supported."""
    return target.cuda_capability_at_least(7, 5)


def supports_preparation_target(target: AcceleratorTarget) -> bool:
    """Return whether rotation and quantization can use the NVIDIA implementation."""
    return target.is_nvidia_cuda


def _base_execution_plan(*, in_features: int) -> NvidiaExecutionPlan:
    """Build shared NVIDIA preparation and fixed GEMM defaults."""
    fused_chunks = fused_preparation_chunks(in_features)
    fused_num_warps = 4
    if fused_chunks is not None:
        chunk_count, chunk_size = fused_chunks
        if chunk_count > 1 and chunk_size <= _TWO_WARP_MAX_CHUNK_SIZE:
            fused_num_warps = 2
        elif chunk_size == _FUSED_MAX_CHUNK_SIZE:
            fused_num_warps = 8
    return NvidiaExecutionPlan(
        # Prepared inputs may feed weights with different output widths.
        # Keep every preparation choice independent of output width.
        fuse_rotation_quantization=fused_chunks is not None,
        fused_num_warps=fused_num_warps,
        rotation_num_warps=_DEFAULT_ROTATION_NUM_WARPS,
        quantization_num_warps=_DEFAULT_QUANTIZATION_NUM_WARPS,
        matmul_block_m=128,
        matmul_block_n=256,
        matmul_block_k=128,
        matmul_num_warps=8,
        matmul_num_stages=3,
    )


def _sm120_execution_plan(
    *,
    in_features: int,
    rows: int | None,
    out_features: int | None,
) -> NvidiaExecutionPlan:
    """Apply measured SM120 policy to the full base plan."""
    plan = _base_execution_plan(in_features=in_features)
    if not rows or not out_features:
        return plan
    small_tiles = ((rows + 31) // 32) * ((out_features + 63) // 64)
    if rows <= 32 or small_tiles <= _SM120_SMALL_TILE_LIMIT:
        return replace(plan, matmul_block_m=32, matmul_block_n=64, matmul_num_stages=4)
    large_columns = (out_features + 255) // 256
    # Count useful 128-row tiles so a one-row tail is not a full tile.
    # When N fits one 64-column tile, wider tiles add no input reuse.
    if out_features <= 64 or rows * large_columns < 128 * _SM120_LARGE_TILE_THRESHOLD:
        return replace(plan, matmul_block_m=64, matmul_block_n=64, matmul_num_warps=4)
    return plan


def _sm8x_execution_plan(
    *,
    in_features: int,
    rows: int | None,
    out_features: int | None,
) -> Sm8xExecutionPlan:
    """Apply measured SM8x policy to the shared preparation plan.

    Short and narrow projections use 64-column Triton tiles, where the base 128x256
    tile would spill. Wider projections with enough rows use the Gluon GEMM, whose
    128x128 and 256x128 tiles keep 64x64 warp tiles and a four-stage copy pipeline.
    """
    base = _base_execution_plan(in_features=in_features)
    fused_num_warps = base.fused_num_warps
    fused_chunks = fused_preparation_chunks(in_features)
    if fused_chunks is not None:
        chunk_count, chunk_size = fused_chunks
        # One warp keeps a short row in registers without cross-warp reductions. GELU and
        # SwiGLU codes may differ from wider launches by one INT8 code; plain inputs do not.
        if chunk_count == 1 and chunk_size <= _SM8X_ONE_WARP_MAX_CHUNK_SIZE:
            fused_num_warps = 1
    plan = Sm8xExecutionPlan(
        fuse_rotation_quantization=base.fuse_rotation_quantization,
        fused_num_warps=fused_num_warps,
        rotation_num_warps=base.rotation_num_warps,
        quantization_num_warps=base.quantization_num_warps,
        matmul_block_m=256,
        matmul_block_n=128,
        matmul_block_k=64,
        matmul_num_warps=8,
        matmul_num_stages=4,
        matmul_group_m=8,
    )
    if not rows or not out_features:
        return plan
    column_tiles = (out_features + 63) // 64
    small_tiles = ((rows + _SM8X_SMALL_ROWS - 1) // _SM8X_SMALL_ROWS) * column_tiles
    if rows <= _SM8X_SMALL_ROWS or small_tiles <= _SM8X_SMALL_TILE_LIMIT:
        return _sm8x_triton_plan(plan, block_m=16, num_stages=4, group_m=0)
    # Count useful tiles so a one-row tail is not a full tile.
    if out_features >= _SM8X_GLUON_MIN_COLUMNS and rows >= _SM8X_GLUON_MIN_ROWS:
        wide_columns = (out_features + 127) // 128
        if rows * wide_columns >= 256 * _SM8X_GLUON_LARGE_THRESHOLD:
            return plan
        if rows * wide_columns >= 128 * _SM8X_GLUON_MEDIUM_THRESHOLD:
            return replace(plan, matmul_block_m=128, matmul_num_warps=4, matmul_num_stages=3)
    if rows >= 128 and rows * column_tiles >= 128 * _SM8X_LARGE_TILE_THRESHOLD:
        return _sm8x_triton_plan(plan, block_m=128, num_stages=3, group_m=16)
    return _sm8x_triton_plan(plan, block_m=64, num_stages=4, group_m=0)


def sm8x_triton_fallback(plan: Sm8xExecutionPlan) -> Sm8xExecutionPlan:
    """Return the grouped 128x64 Triton tile used when Gluon operands are unaligned."""
    return _sm8x_triton_plan(plan, block_m=128, num_stages=3, group_m=16)


def _sm8x_triton_plan(
    plan: Sm8xExecutionPlan, *, block_m: int, num_stages: int, group_m: int
) -> Sm8xExecutionPlan:
    """Return a 64-column, 4-warp Triton tile with the plan's preparation."""
    return replace(
        plan,
        matmul_block_m=block_m,
        matmul_block_n=64,
        matmul_block_k=128,
        matmul_num_warps=4,
        matmul_num_stages=num_stages,
        matmul_group_m=group_m,
    )


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    in_features: int,
    rows: int | None = None,
    out_features: int | None = None,
) -> NvidiaExecutionPlan:
    """Dispatch full architecture policies, retaining the base plan for other targets."""
    if not supports_target(target):
        raise ValueError(f"ConvRot INT8 execution has no optimized policy for {target}")
    if target.is_architecture("sm120"):
        return _sm120_execution_plan(in_features=in_features, rows=rows, out_features=out_features)
    if target.is_cuda_capability(8):
        return _sm8x_execution_plan(in_features=in_features, rows=rows, out_features=out_features)
    return _base_execution_plan(in_features=in_features)
