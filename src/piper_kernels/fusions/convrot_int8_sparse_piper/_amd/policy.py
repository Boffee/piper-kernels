"""Validated ROCm fused sparse projection and output integrations on RDNA4."""

import sys
from dataclasses import dataclass
from typing import Literal

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .._layout import TILE_ROWS

type ProjectionOperation = Literal["query", "key", "value"]

GLUON_BLOCK_M = 2 * TILE_ROWS
GLUON_BLOCK_K = 64
GLUON_NUM_WARPS = 8
GLUON_GROUP_M = 8
# Avoid signed zero-crossing WMMA clock drops across >=4096-feature reductions (linear/_amd/README).
GLUON_ACCUMULATOR_SEED = 0x155555
GLUON_ACCUMULATOR_SEED_MIN_K = 4096
GLUON_ROTARY_DIMS: dict[int, tuple[int, ...]] = {
    64: (0, 32, 48, 64),
    128: (0, 32, 64, 96, 128),
}


@dataclass(frozen=True, slots=True)
class AmdExecutionPlan(ProjectionExecutionPlan):
    """Complete implementation and launch choices for one RDNA4 projection."""

    kernel: Literal["gluon", "triton"] = "triton"


def supports_head_dim(head_dim: int) -> bool:
    """Head widths handled by the RDNA4 projection schedules."""
    return head_dim in (64, 128)


def supports_target(target: AcceleratorTarget) -> bool:
    return (
        sys.platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    )


_QK_EXECUTION_PLAN = AmdExecutionPlan(
    block_m=TILE_ROWS,
    block_k=64,
    heads_per_program=1,
    num_warps=4,
    num_stages=2,
    group_m=8,
    lds_only_fences=True,
    kernel="triton",
)
_D64_VALUE_EXECUTION_PLAN = AmdExecutionPlan(
    block_m=2 * TILE_ROWS,
    block_k=64,
    heads_per_program=2,
    num_warps=8,
    num_stages=2,
    group_m=8,
    lds_only_fences=True,
    kernel="triton",
)
_D128_VALUE_EXECUTION_PLAN = AmdExecutionPlan(
    block_m=2 * TILE_ROWS,
    block_k=64,
    heads_per_program=1,
    num_warps=8,
    num_stages=2,
    group_m=8,
    lds_only_fences=True,
    kernel="triton",
)
_D64_GLUON_EXECUTION_PLAN = AmdExecutionPlan(
    block_m=GLUON_BLOCK_M,
    block_k=GLUON_BLOCK_K,
    heads_per_program=2,
    num_warps=GLUON_NUM_WARPS,
    num_stages=2,
    group_m=GLUON_GROUP_M,
    lds_only_fences=True,
    kernel="gluon",
)
_D128_GLUON_EXECUTION_PLAN = AmdExecutionPlan(
    block_m=GLUON_BLOCK_M,
    block_k=GLUON_BLOCK_K,
    heads_per_program=1,
    num_warps=GLUON_NUM_WARPS,
    num_stages=2,
    group_m=GLUON_GROUP_M,
    lds_only_fences=True,
    kernel="gluon",
)


def supports_gluon(
    operation: ProjectionOperation,
    *,
    head_dim: int,
    input_features: int,
    heads: int,
    rotary_dim: int = 0,
    operands_aligned: bool = True,
    value_format: ValueFormat = ValueFormat.AMD_WMMA,
) -> bool:
    """Return whether RDNA4's 128x128 4x2 Gluon kernels cover these operands."""
    if not operands_aligned or head_dim not in (64, 128):
        return False
    if input_features < GLUON_BLOCK_K or input_features % GLUON_BLOCK_K != 0:
        return False
    heads_per_program = 128 // head_dim
    if heads <= 0 or heads % heads_per_program != 0:
        return False
    if operation == "value":
        return value_format is ValueFormat.AMD_WMMA
    return rotary_dim in GLUON_ROTARY_DIMS[head_dim]


def select_execution_plan(
    operation: ProjectionOperation,
    *,
    head_dim: int,
    input_features: int | None = None,
    heads: int | None = None,
    rotary_dim: int = 0,
    operands_aligned: bool = False,
    value_format: ValueFormat = ValueFormat.AMD_WMMA,
) -> AmdExecutionPlan:
    """Select RDNA4 Gluon kernels when supported, falling back to Triton schedules otherwise."""
    if (
        input_features is not None
        and heads is not None
        and supports_gluon(
            operation,
            head_dim=head_dim,
            input_features=input_features,
            heads=heads,
            rotary_dim=rotary_dim,
            operands_aligned=operands_aligned,
            value_format=value_format,
        )
    ):
        return _D64_GLUON_EXECUTION_PLAN if head_dim == 64 else _D128_GLUON_EXECUTION_PLAN
    if operation != "value":
        return _QK_EXECUTION_PLAN
    return _D64_VALUE_EXECUTION_PLAN if head_dim == 64 else _D128_VALUE_EXECUTION_PLAN
