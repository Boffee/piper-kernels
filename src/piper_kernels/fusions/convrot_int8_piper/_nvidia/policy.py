"""NVIDIA target support and resolved dense Q/K/V projection choices."""

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

_SM120_EXECUTION_PLAN = ProjectionExecutionPlan(
    block_k=128,
    heads_per_program=2,
    num_warps=8,
    num_stages=3,
    round_rsqrt_to_nearest=True,
)
# SM120's two-head 64x256 tiles need more than SM89's 99 KiB of shared memory.
# One head per 64-row tile keeps four warps within the register budget, and
# grouping eight row blocks reuses input rows across heads.
_SM89_EXECUTION_PLAN = ProjectionExecutionPlan(
    block_k=128,
    heads_per_program=1,
    num_warps=4,
    num_stages=3,
    group_m=8,
    round_rsqrt_to_nearest=True,
)

PACKED_VALUE = False
# The compact mean projection needs more CTAs than the full matrix projection.
VALUE_MEAN_BLOCK_N = 32


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native quantized dense Piper targets: exact SM120 and SM89."""
    return target.is_cuda_capability(12, 0) or target.is_cuda_capability(8, 9)


def select_execution_plan(target: AcceleratorTarget) -> ProjectionExecutionPlan:
    """Resolve the target into complete dense projection launch choices."""
    if not supports_target(target):
        raise ValueError(f"Dense projections have no NVIDIA policy for {target}")
    return _SM120_EXECUTION_PLAN if target.is_cuda_capability(12, 0) else _SM89_EXECUTION_PLAN
