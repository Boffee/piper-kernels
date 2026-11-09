"""NVIDIA target support and resolved dense Q/K/V projection choices."""

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._nvidia import policy as projection_policy
from piper_kernels.fusions.convrot_int8_projection._nvidia._plan import (
    NvidiaExecutionPlan,
    ProjectionOperation,
)

_SM120_EXECUTION_PLAN = NvidiaExecutionPlan(
    block_k=128,
    heads_per_program=2,
    num_warps=8,
    num_stages=3,
    round_rsqrt_to_nearest=True,
)

PACKED_VALUE = False
# The compact mean projection needs more CTAs than the full matrix projection.
VALUE_MEAN_BLOCK_N = 32


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native quantized dense Piper targets: exact SM120 and SM89."""
    return target.is_cuda_capability(12, 0) or target.is_cuda_capability(8, 9)


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
        raise ValueError(f"Dense projections have no NVIDIA policy for {target}")
    if operation not in ("query", "key", "value"):
        raise ValueError("projection operation must be query, key, or value")
    if target.is_cuda_capability(12, 0):
        return _SM120_EXECUTION_PLAN
    return projection_policy.select_execution_plan(
        operation=operation,
        input_features=input_features,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        operands_aligned=operands_aligned,
        triton_execution_plan=projection_policy.SM8X_TRITON_EXECUTION_PLAN,
    )
