"""Validated ROCm fused sparse projection and output integrations on RDNA4."""

from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.fusions.convrot_int8_projection._amd import policy as projection_policy
from piper_kernels.fusions.convrot_int8_projection._amd.policy import (
    D64_GLUON_EXECUTION_PLAN as _D64_GLUON_EXECUTION_PLAN,
)
from piper_kernels.fusions.convrot_int8_projection._amd.policy import (
    D128_GLUON_EXECUTION_PLAN as _D128_GLUON_EXECUTION_PLAN,
)
from piper_kernels.fusions.convrot_int8_projection._amd.policy import (
    GLUON_ACCUMULATOR_SEED,
    GLUON_ACCUMULATOR_SEED_MIN_K,
    GLUON_BLOCK_K,
    GLUON_BLOCK_M,
    GLUON_GROUP_M,
    GLUON_NUM_WARPS,
    GLUON_ROTARY_DIMS,
    TILE_ROWS,
    AmdExecutionPlan,
    ProjectionOperation,
    supports_head_dim,
    supports_target,
)
from piper_kernels.fusions.convrot_int8_projection._amd.policy import (
    QK_TRITON_EXECUTION_PLAN as _QK_EXECUTION_PLAN,
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
    """Return whether RDNA4's 128x128 4x2 Gluon kernels cover these sparse operands."""
    if operation == "value" and value_format is not ValueFormat.AMD_WMMA:
        return False
    return projection_policy.supports_gluon(
        operation,
        head_dim=head_dim,
        input_features=input_features,
        heads=heads,
        rotary_dim=rotary_dim,
        operands_aligned=operands_aligned,
    )


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
    if operation != "value":
        triton_plan = _QK_EXECUTION_PLAN
    else:
        triton_plan = _D64_VALUE_EXECUTION_PLAN if head_dim == 64 else _D128_VALUE_EXECUTION_PLAN
    gluon_supported = (
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
    )
    return projection_policy.select_execution_plan(
        operation,
        head_dim=head_dim,
        input_features=input_features,
        heads=heads,
        rotary_dim=rotary_dim,
        operands_aligned=gluon_supported,
        triton_execution_plan=triton_plan,
    )


__all__ = [
    "GLUON_ACCUMULATOR_SEED",
    "GLUON_ACCUMULATOR_SEED_MIN_K",
    "GLUON_BLOCK_K",
    "GLUON_BLOCK_M",
    "GLUON_GROUP_M",
    "GLUON_NUM_WARPS",
    "GLUON_ROTARY_DIMS",
    "_D64_GLUON_EXECUTION_PLAN",
    "_D64_VALUE_EXECUTION_PLAN",
    "_D128_GLUON_EXECUTION_PLAN",
    "_D128_VALUE_EXECUTION_PLAN",
    "_QK_EXECUTION_PLAN",
    "AmdExecutionPlan",
    "ProjectionOperation",
    "select_execution_plan",
    "supports_gluon",
    "supports_head_dim",
    "supports_target",
]
