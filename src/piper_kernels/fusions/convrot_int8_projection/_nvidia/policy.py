"""Projection choices shared by the fused ConvRot INT8 dense and sparse Piper fusions.

The Gluon kernels run the SM8x ``cp.async`` pipeline and use SM120 TMA operand
loads when targeting exact SM120. SM8x plans follow measurements on SM89, which
represents the family.
"""

from dataclasses import replace

from piper_kernels._triton.targets import AcceleratorTarget

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
    # The compact mean projection needs more CTAs than the full matrix projection.
    value_mean_block_n=32,
)
_GLUON_ASYNC_COPY_EXECUTION_PLAN = NvidiaExecutionPlan(
    block_m=GLUON_BLOCK_M,
    block_k=GLUON_BLOCK_K,
    heads_per_program=1,
    num_warps=GLUON_NUM_WARPS,
    num_stages=GLUON_NUM_STAGES,
    round_rsqrt_to_nearest=True,
    value_mean_block_n=32,
    kernel="gluon_async_copy",
)
_GLUON_TMA_EXECUTION_PLAN = replace(_GLUON_ASYNC_COPY_EXECUTION_PLAN, kernel="gluon_tma")


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    operation: ProjectionOperation,
    input_features: int,
    head_dim: int,
    rotary_dim: int,
    operands_aligned: bool,
    triton_execution_plan: NvidiaExecutionPlan,
) -> NvidiaExecutionPlan:
    """Select SM120 TMA or SM8x async-copy Gluon kernels when supported, else Triton.

    Gluon's Q/K epilogue pairs RoPE features within registers, so Q/K need rotary widths
    in multiples of 32; V has no RoPE. Whole K64 slices and 16-byte-aligned contiguous
    operands keep ``cp.async`` and TMA loads valid.
    """
    if (
        head_dim == 128
        and input_features % GLUON_BLOCK_K == 0
        and (operation == "value" or rotary_dim % 32 == 0)
        and operands_aligned
    ):
        return (
            _GLUON_TMA_EXECUTION_PLAN
            if target.is_cuda_capability(12, 0)
            else _GLUON_ASYNC_COPY_EXECUTION_PLAN
        )
    return triton_execution_plan
