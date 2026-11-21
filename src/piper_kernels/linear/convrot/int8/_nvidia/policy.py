"""Measured NVIDIA ConvRot INT8 plans, schedules, and validation policy.

Preparation depends on target family and input width, independently of output
width so projections can share it. GEMM selection returns one complete
:class:`MatmulSchedule`; both choices form one :class:`NvidiaExecutionPlan`.

``matmul_kernel`` explicitly selects ``triton``, ``gluon_async_copy``, or
``gluon_tma``; replacing tiles via :func:`dataclasses.replace` never changes the
implementation. SM120 selects ``gluon_tma`` (``128x256x64``, 8 warps, 3 stages,
``group_m=16``) for large tiles while keeping Triton for small/medium tiles (and
a grouped ``128x64`` Triton fallback when operands do not satisfy Gluon
alignment); SM8x selects measured SM89 Triton/``gluon_async_copy`` schedules;
other supported NVIDIA targets (SM75+) retain the baseline Triton schedule.
"""

from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks

from .._plan import LinearExecutionPlan

_FUSED_NUM_WARPS_VALUES = (1, 2, 4, 8, 16)
_ROTATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_QUANTIZATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_MATMUL_BLOCK_M_VALUES = (16, 32, 64, 128)
_MATMUL_BLOCK_N_VALUES = (16, 32, 64, 128, 256)
_MATMUL_BLOCK_K_VALUES = (32, 64, 128)
_MATMUL_NUM_WARPS_VALUES = (2, 4, 8)
_MATMUL_NUM_STAGES_VALUES = (1, 2, 3, 4)
_MATMUL_GROUP_M_VALUES = (0, 8, 16)
_MATMUL_KERNEL_VALUES = ("triton", "gluon_async_copy", "gluon_tma")


def _choices(values: tuple[int, ...]) -> str:
    *leading, last = values
    return f"{', '.join(map(str, leading))}, or {last}"


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan(LinearExecutionPlan):
    """Explicit implementation and schedule, independent of the selecting architecture.

    ``matmul_group_m`` controls cache grouping (zero means ungrouped). The
    ``triton_*`` options apply only to Triton: ``triton_specialize_m`` decides
    whether row alignment enters the JIT cache key (dynamic-M launches branch per
    tile), and ``triton_explicit_bias_fma`` preserves scale/bias rounding
    independently of tail scheduling. The read-only ``matmul_specialize_m`` and
    ``matmul_explicit_bias_fma`` properties and ``as_dict()`` report the selected
    implementation's effective behavior; Gluon async-copy and TMA always use
    dynamic M and explicit bias FMAs, including in their unaligned fallbacks.
    """

    matmul_kernel: Literal["triton", "gluon_async_copy", "gluon_tma"] = "triton"
    matmul_group_m: int = 0
    triton_specialize_m: bool = True
    triton_explicit_bias_fma: bool = False

    @property
    def matmul_specialize_m(self) -> bool:
        """Return whether the selected implementation specializes on the row count."""
        return self.matmul_kernel == "triton" and self.triton_specialize_m

    @property
    def matmul_explicit_bias_fma(self) -> bool:
        """Return whether the selected implementation uses explicit bias FMAs."""
        return (
            self.matmul_kernel in ("gluon_async_copy", "gluon_tma") or self.triton_explicit_bias_fma
        )

    def as_dict(self) -> dict[str, int | bool | str]:
        """Report effective execution choices, omitting inactive Triton options."""
        choices = LinearExecutionPlan.as_dict(self)
        del choices["triton_specialize_m"], choices["triton_explicit_bias_fma"]
        return choices | {
            "matmul_specialize_m": self.matmul_specialize_m,
            "matmul_explicit_bias_fma": self.matmul_explicit_bias_fma,
        }

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
        self._validate_matmul()

    def _validate_matmul(self) -> None:
        """Validate implementation capabilities independently of measured target defaults."""
        if self.matmul_kernel not in _MATMUL_KERNEL_VALUES:
            raise ValueError("ConvRot matmul kernel must be triton, gluon_async_copy, or gluon_tma")
        for name in ("triton_specialize_m", "triton_explicit_bias_fma"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"ConvRot {name} must be boolean")
        group_m = self.matmul_group_m
        if type(group_m) is not int or group_m not in _MATMUL_GROUP_M_VALUES:
            raise ValueError(f"ConvRot matmul group_m must be {_choices(_MATMUL_GROUP_M_VALUES)}")
        if self.matmul_kernel == "gluon_tma":
            # Two 64-row warp rows and four 64-column warp columns per CTA.
            if (
                self.matmul_block_m != 128
                or self.matmul_block_n != 256
                or self.matmul_block_k != 64
                or self.matmul_num_warps != 8
                or self.matmul_num_stages != 3
            ):
                raise ValueError("ConvRot TMA GEMM requires 128x256x64/8-warps with 3 stages")
            return
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
            return
        self._validate_triton_matmul()

    def _validate_triton_matmul(self) -> None:
        if self.matmul_block_m not in _MATMUL_BLOCK_M_VALUES:
            raise ValueError(
                f"ConvRot Triton matmul block_m must be {_choices(_MATMUL_BLOCK_M_VALUES)}"
            )
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


class MatmulSchedule(NamedTuple):
    """Complete immutable GEMM choices, independent of input preparation."""

    matmul_block_m: int
    matmul_block_n: int
    matmul_block_k: int
    matmul_num_warps: int
    matmul_num_stages: int
    matmul_group_m: int
    matmul_kernel: Literal["triton", "gluon_async_copy", "gluon_tma"] = "triton"
    triton_specialize_m: bool = True
    triton_explicit_bias_fma: bool = False


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
_SM120_GLUON_TMA = MatmulSchedule(
    128,
    256,
    64,
    8,
    3,
    16,
    "gluon_tma",
    triton_specialize_m=False,
    triton_explicit_bias_fma=True,
)
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


def supports_residual_epilogue(
    target: AcceleratorTarget, plan: NvidiaExecutionPlan, logical_dtype: torch.dtype
) -> bool:
    """Use the measured SM120 epilogue when the tile can pack to 16-bit values.

    Both SM120 Triton and ``gluon_tma`` round projections to FP16/BF16 before
    FP32 residual arithmetic to preserve the materialized rounding boundary and
    stay spill-free: RTX 5090 measurements found retaining FP32 in the SM120
    128x256 Triton tile spilled registers and regressed attention blocks up to
    ~1.6% and FFNs 1-5% (and FP32 reusable-update epilogues also spill).
    """
    return (
        target.is_cuda_capability(12, 0)
        and plan.matmul_kernel in ("triton", "gluon_tma")
        and logical_dtype in (torch.float16, torch.bfloat16)
    )


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
        return _SM120_GLUON_TMA
    small_tiles = ((rows + 31) // 32) * ((out_features + 63) // 64)
    if rows <= 32 or small_tiles <= _SM120_SMALL_MAX_TILES:
        return _SM120_SMALL
    large_columns = (out_features + 255) // 256
    # A one-row tail is not a full tile; wide tiles cannot reuse a single 64-column tile.
    if out_features <= 64 or rows * large_columns < 128 * _SM120_LARGE_MIN_TILES:
        return _SM120_MEDIUM
    return _SM120_GLUON_TMA


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
    elif target.is_cuda_capability(12, 0):
        matmul = _sm120_matmul_schedule(rows, out_features)
    else:
        matmul = _BASE_MATMUL
    return _execution_plan(in_features, matmul, sm8x=sm8x)
