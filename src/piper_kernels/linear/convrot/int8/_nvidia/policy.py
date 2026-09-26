"""Support and measured launch policy for the NVIDIA ConvRot INT8 implementation.

These schedules are measured on SM120 and SM89; the SM89 schedule covers the SM8x
family. Other supported NVIDIA targets retain the existing defaults. Hardware support
does not imply per-target tuning.
"""

from dataclasses import dataclass, replace
from typing import Literal

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks

from .._plan import LinearExecutionPlan

_FUSED_NUM_WARPS_VALUES = (1, 2, 4, 8, 16)
_ROTATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_QUANTIZATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_MATMUL_BLOCK_M_VALUES = (16, 32, 64, 128, 256)
_MATMUL_BLOCK_N_VALUES = (16, 32, 64, 128, 256)
_MATMUL_BLOCK_K_VALUES = (32, 64, 128)
_MATMUL_NUM_WARPS_VALUES = (2, 4, 8)
_MATMUL_NUM_STAGES_VALUES = (1, 2, 3, 4)
_MATMUL_GROUP_M_VALUES = (0, 8, 16)
_MATMUL_KERNEL_VALUES = ("triton", "gluon_async_copy")


def _choices(values: tuple[int, ...]) -> str:
    *leading, last = values
    return f"{', '.join(map(str, leading))}, or {last}"


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan(LinearExecutionPlan):
    """Explicit implementation and schedule, independent of the selecting architecture.

    Triton may specialize on M and contract bias implicitly, or branch per tile and
    use explicit bias FMAs. Gluon's async-copy implementation always uses dynamic M
    and explicit FMAs, including in its unaligned-operand fallback.
    """

    matmul_kernel: Literal["triton", "gluon_async_copy"] = "triton"
    matmul_group_m: int = 0
    matmul_specialize_m: bool = True
    matmul_explicit_bias_fma: bool = False

    def __post_init__(self) -> None:
        LinearExecutionPlan.__post_init__(self)
        if self.fused_num_warps not in _FUSED_NUM_WARPS_VALUES:
            raise ValueError(
                f"ConvRot fused preparation num_warps must be {_choices(_FUSED_NUM_WARPS_VALUES)}"
            )
        if self.rotation_num_warps not in _ROTATION_NUM_WARPS_VALUES:
            raise ValueError(
                f"ConvRot split rotation num_warps must be {_choices(_ROTATION_NUM_WARPS_VALUES)}"
            )
        if self.quantization_num_warps not in _QUANTIZATION_NUM_WARPS_VALUES:
            raise ValueError(
                "ConvRot split quantization num_warps must be "
                f"{_choices(_QUANTIZATION_NUM_WARPS_VALUES)}"
            )
        if self.matmul_block_m not in _MATMUL_BLOCK_M_VALUES:
            raise ValueError(f"ConvRot matmul block_m must be {_choices(_MATMUL_BLOCK_M_VALUES)}")
        if self.matmul_block_n not in _MATMUL_BLOCK_N_VALUES:
            raise ValueError(f"ConvRot matmul block_n must be {_choices(_MATMUL_BLOCK_N_VALUES)}")
        if self.matmul_block_k not in _MATMUL_BLOCK_K_VALUES:
            raise ValueError(f"ConvRot matmul block_k must be {_choices(_MATMUL_BLOCK_K_VALUES)}")
        if self.matmul_num_warps not in _MATMUL_NUM_WARPS_VALUES:
            raise ValueError(
                f"ConvRot matmul num_warps must be {_choices(_MATMUL_NUM_WARPS_VALUES)}"
            )
        if self.matmul_num_stages not in _MATMUL_NUM_STAGES_VALUES:
            raise ValueError(
                f"ConvRot matmul num_stages must be {_choices(_MATMUL_NUM_STAGES_VALUES)}"
            )
        self._validate_matmul()

    def _validate_matmul(self) -> None:
        """Validate implementation capabilities independently of measured target defaults."""
        if self.matmul_kernel not in _MATMUL_KERNEL_VALUES:
            raise ValueError("ConvRot matmul kernel must be triton or gluon_async_copy")
        for name in ("matmul_specialize_m", "matmul_explicit_bias_fma"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"ConvRot {name} must be boolean")
        group_m = self.matmul_group_m
        if type(group_m) is not int or group_m not in _MATMUL_GROUP_M_VALUES:
            raise ValueError(f"ConvRot matmul group_m must be {_choices(_MATMUL_GROUP_M_VALUES)}")
        if self.matmul_kernel == "gluon_async_copy":
            # Two 64-column warp tiles and two/four 64-row warp tiles per CTA.
            if (
                self.matmul_block_m not in (128, 256)
                or self.matmul_block_n != 128
                or self.matmul_block_k != 64
                or self.matmul_num_warps != self.matmul_block_m // 32
                or self.matmul_num_stages not in (3, 4)
            ):
                raise ValueError(
                    "ConvRot async-copy GEMM requires 128x128x64/4-warps or "
                    "256x128x64/8-warps, with 3 or 4 stages"
                )
        elif self.matmul_block_m == 256:
            raise ValueError("ConvRot Triton matmul block_m must be 16, 32, 64, or 128")


_FUSED_MAX_CHUNK_SIZE = 16_384
_TWO_WARP_MAX_CHUNK_SIZE = 2_048
_DEFAULT_ROTATION_NUM_WARPS = 4
_DEFAULT_QUANTIZATION_NUM_WARPS = 8
_SM120_SMALL_TILE_LIMIT = 128
_SM120_LARGE_TILE_THRESHOLD = 72
_SM8X_SMALL_ROWS = 16
_SM8X_SMALL_TILE_LIMIT = 96
_SM8X_LARGE_TILE_THRESHOLD = 64
# Gluon tiles need at least 256 rows and more than one 128-column tile. Up to 1,024
# columns, 256-row tiles leave too few CTAs, so those outputs use 128-row Gluon tiles.
_SM8X_GLUON_MIN_ROWS = 256
_SM8X_GLUON_MIN_COLUMNS = 129
_SM8X_GLUON_MEDIUM_MAX_COLUMNS = 1_024
_SM8X_GLUON_THRESHOLD = 48
_SM8X_ONE_WARP_MAX_COLUMNS = 1_024


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
        matmul_group_m=16,
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
        return replace(
            plan, matmul_block_m=32, matmul_block_n=64, matmul_num_stages=4, matmul_group_m=0
        )
    large_columns = (out_features + 255) // 256
    # Count useful 128-row tiles so a one-row tail is not a full tile.
    # When N fits one 64-column tile, wider tiles add no input reuse.
    if out_features <= 64 or rows * large_columns < 128 * _SM120_LARGE_TILE_THRESHOLD:
        return replace(
            plan, matmul_block_m=64, matmul_block_n=64, matmul_num_warps=4, matmul_group_m=0
        )
    return plan


def _sm8x_execution_plan(
    *,
    in_features: int,
    rows: int | None,
    out_features: int | None,
) -> NvidiaExecutionPlan:
    """Apply measured SM8x policy to the shared preparation plan.

    Short and narrow projections use 64-column Triton tiles, where the base 128x256
    tile would spill. Wider projections with enough rows use the Gluon GEMM: 128x128
    tiles up to 1,024 output columns and 256x128 tiles beyond. Each output width uses at
    most three tiles as the row count changes, and SM8x launches do not specialize on M,
    so a layer compiles at most three GEMMs.
    """
    base = _base_execution_plan(in_features=in_features)
    fused_num_warps = base.fused_num_warps
    # One warp keeps a short row, always one fused chunk, in registers without cross-warp
    # reductions. GELU and SwiGLU codes may differ from wider launches by one INT8 code;
    # plain inputs do not.
    if in_features <= _SM8X_ONE_WARP_MAX_COLUMNS:
        fused_num_warps = 1
    plan = replace(
        base,
        fused_num_warps=fused_num_warps,
        matmul_kernel="gluon_async_copy",
        matmul_block_m=256,
        matmul_block_n=128,
        matmul_block_k=64,
        matmul_num_warps=8,
        matmul_num_stages=4,
        matmul_group_m=8,
        matmul_specialize_m=False,
        matmul_explicit_bias_fma=True,
    )
    if not rows or not out_features:
        return plan
    column_tiles = (out_features + 63) // 64
    small_tiles = ((rows + _SM8X_SMALL_ROWS - 1) // _SM8X_SMALL_ROWS) * column_tiles
    if rows <= _SM8X_SMALL_ROWS or small_tiles <= _SM8X_SMALL_TILE_LIMIT:
        return _narrow_triton_plan(plan, block_m=16, num_stages=4, group_m=0)
    # Count useful tiles so a one-row tail is not a full tile. Wide outputs step from the
    # 64-row Triton tile to one Gluon tile; narrow outputs to the 128-row Triton tile.
    if out_features >= _SM8X_GLUON_MIN_COLUMNS:
        gluon = plan
        if out_features <= _SM8X_GLUON_MEDIUM_MAX_COLUMNS:
            gluon = replace(plan, matmul_block_m=128, matmul_num_warps=4, matmul_num_stages=3)
        wide_columns = (out_features + 127) // 128
        gluon_tiles = gluon.matmul_block_m * _SM8X_GLUON_THRESHOLD
        if rows >= _SM8X_GLUON_MIN_ROWS and rows * wide_columns >= gluon_tiles:
            return gluon
    elif rows * column_tiles >= 128 * _SM8X_LARGE_TILE_THRESHOLD:
        return async_copy_fallback_plan(plan)
    return _narrow_triton_plan(plan, block_m=64, num_stages=4, group_m=0)


def async_copy_fallback_plan(plan: NvidiaExecutionPlan) -> NvidiaExecutionPlan:
    """Return the grouped 128x64 Triton tile, also used when Gluon operands are unaligned."""
    return _narrow_triton_plan(plan, block_m=128, num_stages=3, group_m=16)


def _narrow_triton_plan(
    plan: NvidiaExecutionPlan, *, block_m: int, num_stages: int, group_m: int
) -> NvidiaExecutionPlan:
    """Return a dynamic-M, explicit-FMA Triton tile with the plan's preparation."""
    return replace(
        plan,
        matmul_kernel="triton",
        matmul_block_m=block_m,
        matmul_block_n=64,
        matmul_block_k=128,
        matmul_num_warps=4,
        matmul_num_stages=num_stages,
        matmul_group_m=group_m,
        matmul_specialize_m=False,
        matmul_explicit_bias_fma=True,
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
