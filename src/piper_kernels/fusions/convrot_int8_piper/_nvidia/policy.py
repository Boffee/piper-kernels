"""NVIDIA target support and resolved dense Q/K/V projection choices."""

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._nvidia._plan import (
    GLUON_BLOCK_K,
    GLUON_BLOCK_M,
    GLUON_NUM_STAGES,
    GLUON_NUM_WARPS,
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
# SM120's two-head 64x256 tiles need more than SM89's 99 KiB of shared memory.
# One head per 64-row tile keeps four warps within the register budget, and
# grouping eight row blocks reuses input rows across heads.
_SM89_TRITON_EXECUTION_PLAN = NvidiaExecutionPlan(
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
    # Gluon's Q/K epilogue pairs RoPE features within registers; V has no RoPE.
    # Whole K64 slices and aligned pointers keep its 16-byte copies valid.
    if (
        head_dim == 128
        and input_features % GLUON_BLOCK_K == 0
        and (operation == "value" or rotary_dim % 32 == 0)
        and operands_aligned
    ):
        return _SM89_GLUON_EXECUTION_PLAN
    return _SM89_TRITON_EXECUTION_PLAN
