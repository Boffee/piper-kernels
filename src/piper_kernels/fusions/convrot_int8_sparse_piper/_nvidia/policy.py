"""NVIDIA target support and resolved sparse Q/K/V projection choices."""

from dataclasses import replace

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._nvidia._plan import (
    NvidiaExecutionPlan,
    ProjectionOperation,
)
from piper_kernels.fusions.convrot_int8_projection._nvidia.policy import (
    select_sm89_execution_plan,
)

from .._layout import TILE_ROWS


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native sparse Piper attention targets: exact SM120 and SM89."""
    return target.is_cuda_capability(12, 0) or target.is_cuda_capability(8, 9)


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
    """Resolve target, operation, and operand metadata into complete launch choices."""
    if not supports_target(target):
        raise ValueError(f"Sparse projections have no NVIDIA policy for {target}")
    if operation not in ("query", "key", "value"):
        raise ValueError("projection operation must be query, key, or value")
    if target.is_cuda_capability(12, 0):
        return (
            _SM120_QUERY_EXECUTION_PLAN if operation == "query" else _SM120_CONTEXT_EXECUTION_PLAN
        )
    return select_sm89_execution_plan(
        operation=operation,
        input_features=input_features,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        operands_aligned=operands_aligned,
    )
