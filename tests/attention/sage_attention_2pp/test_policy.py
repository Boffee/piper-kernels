"""Host-side specialization policy tests for SageAttention2++."""

from dataclasses import replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage_attention_2pp._policy import (
    SageAttention2ppExecutionPlan,
    select_execution_plan,
)
from piper_kernels.attention.sage_attention_2pp.triton import (
    _default_sage_attention_2pp_execution_plan,
    _prepare_sage_attention_2pp,
)

_SM89 = AcceleratorTarget(backend="cuda", architecture="sm89")
_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")
_SM121 = AcceleratorTarget(backend="cuda", architecture="sm121")


@pytest.mark.parametrize("sequence", [1, 8192, 131073])
def test_default_execution_plan_is_sequence_length_invariant(sequence: int) -> None:
    query = torch.empty((1, 8, sequence, 128), device="meta")

    plan = _default_sage_attention_2pp_execution_plan(
        query,
        False,
        target=_SM120,
    )

    assert plan.block_m == 128
    assert plan.grouped_qk
    assert plan.use_tensor_descriptors


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            _SM89,
            SageAttention2ppExecutionPlan(
                block_m=128,
                grouped_qk=False,
                fuse_query_quantization=False,
                use_tensor_descriptors=False,
                loop_num_stages=3,
                loop_licm=True,
            ),
        ),
        (
            _SM120,
            SageAttention2ppExecutionPlan(
                block_m=128,
                grouped_qk=True,
                fuse_query_quantization=False,
                use_tensor_descriptors=True,
                use_packed_probability_conversion=False,
            ),
        ),
        (
            _SM121,
            SageAttention2ppExecutionPlan(
                block_m=128,
                grouped_qk=True,
                fuse_query_quantization=False,
                use_tensor_descriptors=False,
            ),
        ),
    ],
)
def test_execution_plan_separates_architecture_facts_from_exact_target_tuning(
    target: AcceleratorTarget,
    expected: SageAttention2ppExecutionPlan,
) -> None:
    plan = select_execution_plan(
        target,
        candidate_block_m=128,
        head_dim=128,
        is_causal=False,
    )

    assert plan == expected


@pytest.mark.parametrize(
    ("target", "head_dim", "expected_block_m"),
    [
        (_SM121, 128, 64),
        (_SM89, 64, 64),
        (_SM89, 128, 128),
    ],
)
def test_causal_block_schedule_uses_architecture_specific_tuning(
    target: AcceleratorTarget,
    head_dim: int,
    expected_block_m: int,
) -> None:
    plan = select_execution_plan(
        target,
        candidate_block_m=128,
        head_dim=head_dim,
        is_causal=True,
    )

    assert plan.block_m == expected_block_m


@pytest.mark.parametrize("candidate_block_m", [64, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_uses_uniform_128_row_query_tiles(
    candidate_block_m: int,
    head_dim: int,
    is_causal: bool,
) -> None:
    plan = select_execution_plan(
        _SM120,
        candidate_block_m=candidate_block_m,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    assert plan.block_m == 128
    assert plan.use_tensor_descriptors is (head_dim == 128)


def test_sm89_d128_causal_schedule_uses_measured_launch_policy() -> None:
    plan = select_execution_plan(
        _SM89,
        candidate_block_m=128,
        head_dim=128,
        is_causal=True,
    )

    assert plan.num_warps == 4
    assert plan.num_stages == 2
    assert plan.reverse_causal_blocks


def test_sm89_d128_noncausal_schedule_enables_licm_and_loop_pipeline() -> None:
    plan = select_execution_plan(
        _SM89,
        candidate_block_m=128,
        head_dim=128,
        is_causal=False,
    )

    assert plan.loop_num_stages == 3
    assert plan.loop_licm


@pytest.mark.parametrize(
    ("head_dim", "is_causal"),
    [
        (64, False),
        (64, True),
        (128, False),
        (128, True),
    ],
)
def test_sm120_uses_standalone_query_quantization(
    head_dim: int,
    is_causal: bool,
) -> None:
    plan = select_execution_plan(
        _SM120,
        candidate_block_m=64,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    assert plan.grouped_qk
    assert not plan.fuse_query_quantization


@pytest.mark.parametrize(
    ("target", "head_dim", "is_causal", "expected"),
    [
        (_SM89, 128, True, True),
        (_SM120, 64, True, False),
        (_SM120, 128, False, False),
        (_SM120, 128, True, False),
    ],
)
def test_probability_conversion_policy_is_specialized_by_target(
    target: AcceleratorTarget,
    head_dim: int,
    is_causal: bool,
    expected: bool,
) -> None:
    plan = select_execution_plan(
        target,
        candidate_block_m=128,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    assert plan.use_packed_probability_conversion is expected


@pytest.mark.parametrize("is_causal", [False, True])
def test_other_sm12x_targets_do_not_inherit_sm120_specializations(
    is_causal: bool,
) -> None:
    plan = select_execution_plan(
        _SM121,
        candidate_block_m=128,
        head_dim=128,
        is_causal=is_causal,
    )

    assert plan.grouped_qk
    assert not plan.fuse_query_quantization
    assert not plan.use_tensor_descriptors


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
        {"grouped_qk": False, "fuse_query_quantization": True},
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
