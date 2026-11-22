"""Conservative AMD support, execution plans, and measured RDNA4 launch policy.

Supports HIP architectures covered by offline compilation tests on Linux and
Windows (``gfx942``, ``gfx1100``, ``gfx1151``, ``gfx1200``, ``gfx1201``), with
hardware correctness and performance measurements on RX 9070 XT (``gfx1201``).
Unknown targets fall back to the portable reference path.

Preparation chooses up to three group-aligned power-of-two segments, fusing
rotation and quantization through 16,384 input features before falling back to a
split path. On ``gfx1201``, short batches with known ``out_features`` use
single-row-tile schedules (``(32, 128, 64)`` / 4 warps for ``1..32`` rows and
``(64, 128, 64)`` / 4 warps for ``33..64`` rows) so uncached weights stream once
while doubling N parallelism. Above 64 rows, GEMMs use ``(128, 256, 64)`` tiles
(8 warps, 2 stages, ``group_m=8``), switching to ``(128, 128, 128)`` when
``in_features`` is a multiple of 8,192 and the launch fills at least two
workgroups per WGP (64 tiles).
"""

import sys
from dataclasses import dataclass

import torch

from piper_kernels._triton.targets import (
    RDNA4_MIN_FULL_GRID_TILES,
    AcceleratorTarget,
    rdna4_int8_accumulator_seed,
)

from .._plan import LinearExecutionPlan

# gfx1201 has hardware correctness/performance coverage; the remaining targets
# have offline compilation coverage. Unknown architectures use the reference.
_SUPPORTED_ARCHITECTURES = ("gfx942", "gfx1100", "gfx1151", "gfx1200", "gfx1201")

# Maximum feature width fused into a single rotation-and-quantization kernel.
_MAX_FUSED_PREPARATION_WIDTH = 16384

_MIN_GFX1201_MATMUL_TILES = RDNA4_MIN_FULL_GRID_TILES


@dataclass(frozen=True, slots=True)
class AmdExecutionPlan(LinearExecutionPlan):
    """AMD-owned launch constraints, including wave32 preparation with 32 warps."""

    def __post_init__(self) -> None:
        LinearExecutionPlan.__post_init__(self)
        for name in ("fused_num_warps", "rotation_num_warps", "quantization_num_warps"):
            if getattr(self, name) not in (1, 2, 4, 8, 16, 32):
                raise ValueError(f"AMD {name} must be a power of two from 1 through 32")
        if self.matmul_num_warps not in (4, 8):
            raise ValueError("AMD matmul_num_warps must be 4 or 8")
        for name in ("matmul_block_m", "matmul_block_n", "matmul_block_k"):
            if getattr(self, name) not in (16, 32, 64, 128, 256):
                raise ValueError(f"AMD {name} must be a power of two from 16 through 256")
        if self.matmul_num_stages not in (1, 2, 3, 4):
            raise ValueError("AMD matmul_num_stages must be 1, 2, 3, or 4")


def supports_target(target: AcceleratorTarget) -> bool:
    """Select HIP architectures covered by the AMD compile tests on Linux and Windows."""
    return (
        sys.platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture(*_SUPPORTED_ARCHITECTURES)
    )


def supports_fused_preparation(in_features: int) -> bool:
    """Return whether AMD fuses rotation and quantization for ``in_features``."""
    return 0 < in_features <= _MAX_FUSED_PREPARATION_WIDTH


def preparation_blocks(row_width: int) -> tuple[int, int, int]:
    """Choose up to three group-aligned power-of-two row segments."""
    if row_width <= 0:
        raise ValueError("ConvRot preparation row width must be positive")
    if (row_width & (row_width - 1)) == 0:
        return row_width, 0, 0

    leading_block = 1 << (row_width.bit_length() - 1)
    tail_width = row_width - leading_block
    tail_block = 1 << (tail_width - 1).bit_length()
    leading_tail = (leading_block, tail_block, 0)

    equal_block_width = (row_width + 2) // 3
    equal_block = 1 << (equal_block_width - 1).bit_length()
    equal_count = (row_width + equal_block - 1) // equal_block
    if equal_count in (2, 3) and equal_count * equal_block <= leading_block + tail_block:
        return (equal_block, equal_block, equal_block if equal_count == 3 else 0)
    return leading_tail


def matmul_group_m(plan: LinearExecutionPlan) -> int:
    """Reuse each weight tile across eight row tiles in the large GEMM schedules."""
    if plan.matmul_block_m == 128 and (
        plan.matmul_block_n == 256 or (plan.matmul_block_n, plan.matmul_block_k) == (128, 128)
    ):
        return 8
    return 0


def matmul_accumulator_seed(target: AcceleratorTarget, in_features: int, output_tiles: int) -> int:
    """Return the ``0x155555`` zero-crossing seed for long INT8 reductions on gfx1201."""
    return rdna4_int8_accumulator_seed(target, in_features, output_tiles)


def _supports_rdna4_128m_fast_path(
    target: AcceleratorTarget,
    plan: LinearExecutionPlan,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    output_row_bytes: int,
) -> bool:
    """Check shared gfx1201 128-row schedule and signed-INT32 SRD byte bounds."""
    padded_rows = (rows + 127) // 128 * 128
    return (
        target.is_architecture("gfx1201")
        and plan.matmul_block_m == 128
        and plan.matmul_num_warps == 8
        and plan.matmul_num_stages == 2
        and rows > 64
        # Keep input, weight, and output byte/element offsets within signed INT32
        # so AMDGPU lowers memory ops to SRD buffer instructions.
        and max(in_features * out_features, padded_rows * max(in_features, output_row_bytes))
        < 2**31
    )


def supports_paired_matmul(
    target: AcceleratorTarget,
    plan: LinearExecutionPlan,
    logical_dtype: torch.dtype,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    output_row_stride: int,
) -> bool:
    """Select the shared-activation dual-accumulator GEMM on gfx1201."""
    return (
        (plan.matmul_block_n, plan.matmul_block_k) == (256, 64)
        and out_features % 128 == 0
        and in_features % 64 == 0
        and _supports_rdna4_128m_fast_path(
            target,
            plan,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            output_row_bytes=logical_dtype.itemsize * output_row_stride,
        )
    )


def supports_residual_epilogue(
    target: AcceleratorTarget,
    plan: LinearExecutionPlan,
    logical_dtype: torch.dtype,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> bool:
    """Return whether aligned RDNA4 16-bit GEMMs fuse the residual epilogue.

    FP32 outputs, short batches (``rows <= 64``), unaligned dimensions, and
    offsets exceeding signed 32-bit SRD range fall back to a materialized
    projection plus standalone ``apply_residual_add``.
    """
    return (
        logical_dtype in (torch.float16, torch.bfloat16)
        and (plan.matmul_block_n, plan.matmul_block_k) in ((128, 128), (256, 64))
        and out_features % plan.matmul_block_n == 0
        and in_features % plan.matmul_block_k == 0
        and _supports_rdna4_128m_fast_path(
            target,
            plan,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            output_row_bytes=logical_dtype.itemsize * out_features,
        )
    )


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    in_features: int,
    rows: int | None = None,
    out_features: int | None = None,
) -> AmdExecutionPlan:
    """Select preparation by input width and GEMM tiling by operand shape."""
    if not supports_target(target):
        raise ValueError(f"ConvRot INT8 execution has no optimized policy for {target}")
    if in_features <= 0:
        raise ValueError("ConvRot preparation row width must be positive")
    block_size = max(128, 1 << (in_features - 1).bit_length())
    rdna4 = target.is_architecture("gfx1200", "gfx1201")
    fused_num_warps = 4
    if rdna4:
        if in_features & (in_features - 1):
            fused_num_warps = 8 if block_size > 8192 else 4
        elif block_size > 8192:
            fused_num_warps = 16
        elif block_size > 2048:
            fused_num_warps = 8
        else:
            fused_num_warps = 4
    # At 8 KiB row-stride multiples, fetching 128 adjacent K bytes improves
    # throughput on gfx1201. Halve N to keep the accumulator spill-free.
    # Keep the existing tile for underfilled grids: below two workgroups per
    # WGP, the final wave can dominate the benefit of wider loads.
    wide_k = (
        target.is_architecture("gfx1201")
        and in_features % 8192 == 0
        and rows is not None
        and out_features is not None
        and ((rows + 127) // 128) * ((out_features + 255) // 256) >= _MIN_GFX1201_MATMUL_TILES
    )
    block_m = 128 if rdna4 else 64
    block_n, block_k = (256, 64) if rdna4 else (64, 64)
    matmul_warps = 8 if rdna4 else 4
    if wide_k:
        block_n, block_k = 128, 128
    # Keep short batches (rows <= 64) on a single M tile with 128-column blocks
    # so weights exceeding the 64 MiB Infinity Cache stream once while doubling
    # N parallelism across the 32 WGPs. Above 64 rows, keep 128-row plans so
    # paired and fused-residual fast paths apply uniformly.
    if target.is_architecture("gfx1201") and rows is not None and out_features is not None:
        if 1 <= rows <= 32:
            block_m, block_n, block_k, matmul_warps = 32, 128, 64, 4
        elif 33 <= rows <= 64:
            block_m, block_n, block_k, matmul_warps = 64, 128, 64, 4
    return AmdExecutionPlan(
        fuse_rotation_quantization=supports_fused_preparation(in_features),
        fused_num_warps=fused_num_warps,
        rotation_num_warps=4,
        quantization_num_warps=4,
        matmul_block_m=block_m,
        matmul_block_n=block_n,
        matmul_block_k=block_k,
        matmul_num_warps=matmul_warps,
        matmul_num_stages=2 if rdna4 else 1,
    )
