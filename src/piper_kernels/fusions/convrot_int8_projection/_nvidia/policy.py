"""SM89 projection choices shared by the fused ConvRot INT8 dense and sparse Piper fusions."""

from ._plan import (
    GLUON_BLOCK_K,
    GLUON_BLOCK_M,
    GLUON_NUM_STAGES,
    GLUON_NUM_WARPS,
    NvidiaExecutionPlan,
    ProjectionOperation,
)

# SM120's two-head tiles exceed SM89's 99 KiB of shared memory, and its V accumulator
# spills. One head per 64-row tile keeps four warps within SM89's register budget, and
# grouping eight row blocks reuses input rows across heads.
_SM89_TRITON_EXECUTION_PLAN = NvidiaExecutionPlan(
    block_m=64,
    block_k=128,
    heads_per_program=1,
    num_warps=4,
    num_stages=3,
    group_m=8,
    round_rsqrt_to_nearest=True,
)
_SM89_GLUON_EXECUTION_PLAN = NvidiaExecutionPlan(
    block_m=GLUON_BLOCK_M,
    block_k=GLUON_BLOCK_K,
    heads_per_program=1,
    num_warps=GLUON_NUM_WARPS,
    num_stages=GLUON_NUM_STAGES,
    round_rsqrt_to_nearest=True,
    kernel="gluon_async_copy",
)


def select_sm89_execution_plan(
    *,
    operation: ProjectionOperation,
    input_features: int,
    head_dim: int,
    rotary_dim: int,
    operands_aligned: bool,
) -> NvidiaExecutionPlan:
    """Select the Gluon projection kernels when they cover the operands, else Triton tiles.

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
        return _SM89_GLUON_EXECUTION_PLAN
    return _SM89_TRITON_EXECUTION_PLAN
