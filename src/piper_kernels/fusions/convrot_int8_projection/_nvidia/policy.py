"""Projection choices shared by the fused ConvRot INT8 dense and sparse Piper fusions.

The Gluon kernels run the SM8x ``cp.async`` pipeline and use SM120 TMA operand
loads when targeting exact SM120. SM8x plans follow measurements on SM89, which
represents the family.
"""

from dataclasses import dataclass, replace
from typing import Literal

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan
from piper_kernels.fusions.convrot_int8_projection._plan import (
    ProjectionOperation as ProjectionOperation,  # noqa: PLC0414
)

# The Gluon feature layout and register epilogues require this fixed tile. The shared
# policy builds the one Gluon plan from these constants, which the fragments compile with.
GLUON_BLOCK_M = 128
GLUON_BLOCK_K = 64
GLUON_NUM_WARPS = 4
GLUON_NUM_STAGES = 3


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan(ProjectionExecutionPlan):
    """Complete implementation and launch choices for one NVIDIA projection."""

    kernel: Literal["gluon_async_copy", "gluon_tma", "triton"] = "triton"


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
