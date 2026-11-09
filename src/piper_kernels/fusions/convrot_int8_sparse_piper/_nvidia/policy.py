"""NVIDIA target support and resolved sparse Q/K/V projection choices."""

from dataclasses import replace

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._nvidia import policy as attention_policy
from piper_kernels.fusions.convrot_int8_projection._nvidia import policy as projection_policy
from piper_kernels.fusions.convrot_int8_projection._nvidia._plan import (
    NvidiaExecutionPlan,
    ProjectionOperation,
)

from .._layout import TILE_ROWS


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native sparse Piper attention targets: NVIDIA SM8x and exact SM120."""
    return attention_policy.supports_target(target)


_SM120_QUERY_EXECUTION_PLAN = NvidiaExecutionPlan(
    block_m=TILE_ROWS,
    block_k=128,
    heads_per_program=2,
    num_warps=8,
    num_stages=3,
    round_rsqrt_to_nearest=True,
)
_SM120_CONTEXT_EXECUTION_PLAN = replace(_SM120_QUERY_EXECUTION_PLAN, block_m=2 * TILE_ROWS)


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    operation: ProjectionOperation,
    input_features: int,
    head_dim: int,
    rotary_dim: int = 0,
    operands_aligned: bool,
) -> NvidiaExecutionPlan:
    """Resolve target, operation, and operand metadata into complete launch choices.

    Every supported target projects the operands that the Gluon kernels cover with
    them, and other operands with its own Triton schedule.
    """
    if not supports_target(target):
        raise ValueError(f"Sparse projections have no NVIDIA policy for {target}")
    # On an RTX 5090 at 1K-32K rows and 8-56 heads, Gluon projects Q 1.3-1.4x and V
    # 1.0-1.3x faster than SM120's Triton plans. K ranges from 5% slower, for 12 heads
    # at 2.8K rows, to 1.2x faster.
    if target.is_cuda_capability(12, 0):
        triton_execution_plan = (
            _SM120_QUERY_EXECUTION_PLAN if operation == "query" else _SM120_CONTEXT_EXECUTION_PLAN
        )
    else:
        triton_execution_plan = projection_policy.SM8X_TRITON_EXECUTION_PLAN
    return projection_policy.select_execution_plan(
        operation=operation,
        input_features=input_features,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        operands_aligned=operands_aligned,
        triton_execution_plan=triton_execution_plan,
    )
