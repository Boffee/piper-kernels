"""Backend-independent execution planning for SageAttention2++."""

from dataclasses import asdict, dataclass, replace

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)


@dataclass(frozen=True, slots=True)
class SageAttention2ppExecutionPlan:
    """Host-side specialization choices for one SageAttention2++ invocation."""

    block_m: int
    grouped_qk: bool
    fuse_query_quantization: bool
    use_tensor_descriptors: bool
    use_packed_probability_conversion: bool = True
    num_warps: int = 4
    num_stages: int = 3
    reverse_causal_blocks: bool = False
    loop_num_stages: int | None = None
    loop_licm: bool = False

    def __post_init__(self) -> None:
        if self.block_m not in BLOCK_M_VALUES:
            raise ValueError("SageAttention2++ block_m must be 64 or 128")
        if self.num_warps not in NUM_WARPS_VALUES:
            raise ValueError("SageAttention2++ num_warps must be 2, 4, or 8")
        if self.num_stages not in NUM_STAGES_VALUES:
            raise ValueError("SageAttention2++ num_stages must be 1, 2, 3, or 4")
        if self.loop_num_stages not in LOOP_NUM_STAGES_VALUES:
            raise ValueError("SageAttention2++ loop_num_stages must be None, 1, 2, 3, or 4")
        if self.fuse_query_quantization and not self.grouped_qk:
            raise ValueError("fused Q quantization requires grouped Q/K scales")

    def as_dict(self) -> dict[str, object]:
        """Return the execution-plan fields as plain metadata."""
        return asdict(self)


def _generic_execution_plan(
    target: AcceleratorTarget,
    *,
    candidate_block_m: int,
    is_causal: bool,
) -> SageAttention2ppExecutionPlan:
    """Build capability-based defaults before exact-target tuning is applied."""
    return SageAttention2ppExecutionPlan(
        block_m=min(candidate_block_m, 64) if is_causal else candidate_block_m,
        grouped_qk=target.is_cuda_capability(12),
        fuse_query_quantization=False,
        use_tensor_descriptors=False,
    )


def _apply_sm89_policy(
    plan: SageAttention2ppExecutionPlan,
    *,
    candidate_block_m: int,
    head_dim: int,
    is_causal: bool,
) -> SageAttention2ppExecutionPlan:
    """Apply schedules measured on exact SM89 D128 shapes."""
    if is_causal and head_dim == 128:
        return replace(
            plan,
            block_m=candidate_block_m,
            num_stages=2,
            reverse_causal_blocks=True,
        )
    if not is_causal and head_dim == 128:
        return replace(
            plan,
            loop_num_stages=3,
            loop_licm=True,
        )
    return plan


def _apply_sm120_policy(
    plan: SageAttention2ppExecutionPlan,
    *,
    head_dim: int,
) -> SageAttention2ppExecutionPlan:
    """Apply schedules and preprocessing choices measured on exact SM120."""
    use_tensor_descriptors = head_dim == 128
    return replace(
        plan,
        block_m=128,
        fuse_query_quantization=False,
        use_tensor_descriptors=use_tensor_descriptors,
        use_packed_probability_conversion=False,
    )


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    candidate_block_m: int,
    head_dim: int,
    is_causal: bool,
) -> SageAttention2ppExecutionPlan:
    """Combine portable capability defaults with exact-target measured policy."""
    plan = _generic_execution_plan(
        target,
        candidate_block_m=candidate_block_m,
        is_causal=is_causal,
    )
    if target.is_cuda_capability(8, 9):
        return _apply_sm89_policy(
            plan,
            candidate_block_m=candidate_block_m,
            head_dim=head_dim,
            is_causal=is_causal,
        )
    if target.is_cuda_capability(12, 0):
        return _apply_sm120_policy(
            plan,
            head_dim=head_dim,
        )
    return plan
