"""Backend-independent execution planning for SageAttention2++."""

from dataclasses import replace

from piper_kernels._triton.targets import AcceleratorTarget

from ._plan import SageAttention2ppExecutionPlan


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
        use_tensor_descriptors=False,
    )


def _apply_sm89_policy(
    plan: SageAttention2ppExecutionPlan,
    *,
    candidate_block_m: int,
    head_dim: int,
    is_causal: bool,
) -> SageAttention2ppExecutionPlan:
    """Apply schedules measured on exact SM89 D128 shapes, plus causal LICM."""
    if is_causal:
        # Since Triton 3.9, disabling LICM keeps the Q tile's shared-memory copy
        # inside the K loop. Timed on SM120 as a proxy, not on SM89: causal
        # schedules without LICM were 20% (D128) and 3% (D64) slower on Triton 3.9
        # than on 3.8, and within 2% of 3.8 with it. Non-causal D64 is unaffected.
        plan = replace(plan, loop_licm=True)
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
        use_tensor_descriptors=use_tensor_descriptors,
        use_packed_probability_conversion=False,
        # Since Triton 3.9, disabling LICM also keeps the Q tile's shared-memory
        # copy inside the K loop, so D128 spills and runs up to 36% slower.
        # Enabling LICM brings Triton 3.9 within 2% of Triton 3.8, and changes
        # Triton 3.8 by 1% slower (non-causal D128) to 4% faster (causal D128).
        loop_licm=True,
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
