"""Projection choices shared by the fused ConvRot INT8 dense and sparse Piper fusions.

The Gluon kernels run the SM8x ConvRot INT8 GEMM's ``cp.async`` pipeline. SM8x plans
follow measurements on SM89, which represents the family; SM120 runs the same kernels.
"""

from ._plan import (
    GLUON_BLOCK_K,
    GLUON_BLOCK_M,
    GLUON_NUM_STAGES,
    GLUON_NUM_WARPS,
    NvidiaExecutionPlan,
    ProjectionOperation,
)

# SM120's two-head tiles exceed the 99 KiB of shared memory on SM86 and SM89, and its V
# accumulator spills there. One head per 64-row tile keeps four warps within SM8x's
# register budget, and grouping eight row blocks reuses input rows across heads.
SM8X_TRITON_EXECUTION_PLAN = NvidiaExecutionPlan(
    block_m=64,
    block_k=128,
    heads_per_program=1,
    num_warps=4,
    num_stages=3,
    group_m=8,
    round_rsqrt_to_nearest=True,
)
_GLUON_EXECUTION_PLAN = NvidiaExecutionPlan(
    block_m=GLUON_BLOCK_M,
    block_k=GLUON_BLOCK_K,
    heads_per_program=1,
    num_warps=GLUON_NUM_WARPS,
    num_stages=GLUON_NUM_STAGES,
    round_rsqrt_to_nearest=True,
    kernel="gluon_async_copy",
)


def select_execution_plan(
    *,
    operation: ProjectionOperation,
    input_features: int,
    head_dim: int,
    rotary_dim: int,
    operands_aligned: bool,
    triton_execution_plan: NvidiaExecutionPlan,
) -> NvidiaExecutionPlan:
    """Select the Gluon kernels when they cover the operands, else the target's Triton plan.

    Gluon's Q/K epilogue pairs RoPE features within registers, so Q/K need rotary widths
    in multiples of 32; V has no RoPE. Whole K64 slices and 16-byte-aligned operands keep
    its ``cp.async`` copies valid.
    """
    if (
        head_dim == 128
        and input_features % GLUON_BLOCK_K == 0
        and (operation == "value" or rotary_dim % 32 == 0)
        and operands_aligned
    ):
        return _GLUON_EXECUTION_PLAN
    return triton_execution_plan
