"""Projection choices shared by the fused ConvRot INT8 dense and sparse Piper fusions on RDNA4."""

import sys
from dataclasses import dataclass
from typing import Literal

from piper_kernels._triton.targets import (
    GFX120X_ARCHITECTURES,
    RDNA4_INT8_ACCUMULATOR_SEED,
    RDNA4_INT8_ACCUMULATOR_SEED_MIN_REDUCTION,
    AcceleratorTarget,
)
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan
from piper_kernels.fusions.convrot_int8_projection._plan import (
    ProjectionOperation as ProjectionOperation,  # noqa: PLC0414
)

TILE_ROWS = 64
QUERY_SCALE_ROWS = 32
GLUON_BLOCK_M = 2 * TILE_ROWS
GLUON_BLOCK_K = 64
GLUON_NUM_WARPS = 8
GLUON_GROUP_M = 8
GLUON_ACCUMULATOR_SEED = RDNA4_INT8_ACCUMULATOR_SEED
GLUON_ACCUMULATOR_SEED_MIN_K = RDNA4_INT8_ACCUMULATOR_SEED_MIN_REDUCTION
GLUON_ROTARY_DIMS: dict[int, tuple[int, ...]] = {
    64: (0, 32, 48, 64),
    128: (0, 32, 64, 96, 128),
}


@dataclass(frozen=True, slots=True)
class AmdExecutionPlan(ProjectionExecutionPlan):
    """Complete implementation and launch choices for one RDNA4 projection."""

    kernel: Literal["gluon", "triton"] = "triton"

    @property
    def uses_gluon(self) -> bool:
        return self.kernel == "gluon"


QK_TRITON_EXECUTION_PLAN = AmdExecutionPlan(
    block_m=TILE_ROWS,
    block_k=64,
    heads_per_program=1,
    num_warps=4,
    num_stages=2,
    group_m=8,
    lds_only_fences=True,
    kernel="triton",
)
D64_GLUON_EXECUTION_PLAN = AmdExecutionPlan(
    block_m=GLUON_BLOCK_M,
    block_k=GLUON_BLOCK_K,
    heads_per_program=2,
    num_warps=GLUON_NUM_WARPS,
    num_stages=2,
    group_m=GLUON_GROUP_M,
    lds_only_fences=True,
    kernel="gluon",
)
D128_GLUON_EXECUTION_PLAN = AmdExecutionPlan(
    block_m=GLUON_BLOCK_M,
    block_k=GLUON_BLOCK_K,
    heads_per_program=1,
    num_warps=GLUON_NUM_WARPS,
    num_stages=2,
    group_m=GLUON_GROUP_M,
    lds_only_fences=True,
    kernel="gluon",
)


def supports_head_dim(head_dim: int) -> bool:
    """Head widths handled by the RDNA4 projection schedules."""
    return head_dim in (64, 128)


def supports_target(target: AcceleratorTarget) -> bool:
    """Match validated RDNA4 HIP targets on Linux and Windows."""
    return sys.platform in ("linux", "win32") and target.is_hip_architecture(*GFX120X_ARCHITECTURES)


def supports_gluon(
    operation: ProjectionOperation,
    *,
    head_dim: int,
    input_features: int,
    heads: int,
    rotary_dim: int = 0,
    operands_aligned: bool = True,
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
        return True
    return rotary_dim in GLUON_ROTARY_DIMS[head_dim]


def select_execution_plan(
    operation: ProjectionOperation,
    *,
    head_dim: int,
    input_features: int | None = None,
    heads: int | None = None,
    rotary_dim: int = 0,
    operands_aligned: bool = False,
    triton_execution_plan: AmdExecutionPlan = QK_TRITON_EXECUTION_PLAN,
) -> AmdExecutionPlan:
    """Select RDNA4 Gluon kernels when supported, else fall back to ``triton_execution_plan``."""
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
        )
    ):
        return D64_GLUON_EXECUTION_PLAN if head_dim == 64 else D128_GLUON_EXECUTION_PLAN
    return triton_execution_plan
