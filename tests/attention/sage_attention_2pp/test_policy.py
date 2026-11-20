"""Host-side specialization policy tests for SageAttention2++."""

from dataclasses import replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage_attention_2pp._policy import select_execution_plan
from piper_kernels.attention.sage_attention_2pp.triton import (
    _prepare_sage_attention_2pp,
    default_execution_plan,
)

_SM89 = AcceleratorTarget(backend="cuda", architecture="sm89")
_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")
_SM121 = AcceleratorTarget(backend="cuda", architecture="sm121")


def test_default_execution_plan_is_sequence_length_invariant() -> None:
    plans = [
        default_execution_plan(
            torch.empty((1, 8, sequence, 128), device="meta"), False, target=_SM120
        )
        for sequence in (1, 8192, 131073)
    ]
    assert all(plan == plans[0] for plan in plans)
    assert plans[0].grouped_qk is True
    assert plans[0].use_tensor_descriptors is True


@pytest.mark.parametrize("target", [_SM89, _SM120, _SM121])
@pytest.mark.parametrize("candidate_block_m", [64, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_selected_plans_satisfy_capability_and_safety_invariants(
    target: AcceleratorTarget,
    candidate_block_m: int,
    head_dim: int,
    is_causal: bool,
) -> None:
    plan = select_execution_plan(
        target,
        candidate_block_m=candidate_block_m,
        head_dim=head_dim,
        is_causal=is_causal,
    )
    assert plan.block_m in (64, 128)
    assert plan.num_warps in (2, 4, 8)
    assert 1 <= plan.num_stages <= 4
    assert plan.grouped_qk is target.is_cuda_capability(12)
    if not is_causal:
        assert not plan.reverse_causal_blocks
    # Tensor descriptors require exact SM120 with D128 and M128.
    assert plan.use_tensor_descriptors is (target == _SM120 and head_dim == 128)
    if plan.use_tensor_descriptors:
        assert plan.block_m == 128
    # Unmeasured SM12x targets do not inherit exact-SM120 specializations.
    if target == _SM121:
        assert not plan.use_tensor_descriptors
        assert not plan.loop_licm
        assert not plan.reverse_causal_blocks
        if is_causal:
            assert plan.block_m <= 64


def test_alternate_plan_can_disable_tensor_descriptors() -> None:
    default_plan = select_execution_plan(
        _SM120,
        candidate_block_m=128,
        head_dim=128,
        is_causal=False,
    )
    pointer_plan = replace(default_plan, use_tensor_descriptors=False)

    assert default_plan.use_tensor_descriptors
    assert not pointer_plan.use_tensor_descriptors


def test_execution_plan_rejects_reverse_order_for_noncausal_invocation() -> None:
    query = torch.empty((1, 1, 64, 64), device="meta")
    plan = replace(
        select_execution_plan(
            _SM89,
            candidate_block_m=64,
            head_dim=64,
            is_causal=False,
        ),
        reverse_causal_blocks=True,
    )

    with pytest.raises(ValueError, match="requires causal attention"):
        _prepare_sage_attention_2pp(
            query,
            query,
            query,
            0.125,
            False,
            execution_plan=plan,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"block_m": 16},
        {"num_warps": 1},
        {"num_stages": 5},
        {"loop_num_stages": 5},
    ],
)
def test_execution_plan_rejects_inconsistent_specializations(
    changes: dict[str, object],
) -> None:
    plan = select_execution_plan(
        _SM120,
        candidate_block_m=128,
        head_dim=128,
        is_causal=False,
    )

    with pytest.raises(ValueError, match=r"must be|requires"):
        replace(plan, **changes)
