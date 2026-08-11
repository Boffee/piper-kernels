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


def test_default_execution_plan_supports_meta_tensors_with_resolved_target() -> None:
    query = torch.empty((1, 8, 8192, 128), device="meta")
    key = torch.empty_like(query)

    plan = _default_sage_attention_2pp_execution_plan(
        query,
        key,
        False,
        target=_SM120,
    )

    assert plan.block_m == 64
    assert plan.grouped_qk
    assert plan.fuse_kv_quantization


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            _SM89,
            SageAttention2ppExecutionPlan(
                block_m=128,
                grouped_qk=False,
                fuse_kv_quantization=False,
                fuse_query_quantization=False,
                use_unscaled_score_recurrence=False,
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
                fuse_kv_quantization=True,
                fuse_query_quantization=True,
                use_unscaled_score_recurrence=False,
                use_tensor_descriptors=True,
            ),
        ),
        (
            _SM121,
            SageAttention2ppExecutionPlan(
                block_m=128,
                grouped_qk=True,
                fuse_kv_quantization=False,
                fuse_query_quantization=False,
                use_unscaled_score_recurrence=False,
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
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=False,
    )

    assert plan == expected


@pytest.mark.parametrize(
    ("target", "query_length", "expected_block_m"),
    [
        (_SM120, 4096, 64),
        (_SM120, 4097, 128),
        (_SM121, 8192, 64),
        (_SM89, 8191, 64),
        (_SM89, 8192, 128),
    ],
)
def test_causal_block_schedule_uses_architecture_specific_tuning(
    target: AcceleratorTarget,
    query_length: int,
    expected_block_m: int,
) -> None:
    plan = select_execution_plan(
        target,
        candidate_block_m=128,
        query_length=query_length,
        key_length=query_length,
        head_dim=128,
        is_causal=True,
    )

    assert plan.block_m == expected_block_m


def test_long_sm89_d128_causal_schedule_uses_measured_launch_policy() -> None:
    plan = select_execution_plan(
        _SM89,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=True,
    )

    assert plan.num_warps == 4
    assert plan.num_stages == 2
    assert plan.reverse_causal_blocks


def test_long_sm89_d128_noncausal_schedule_enables_licm_and_loop_pipeline() -> None:
    plan = select_execution_plan(
        _SM89,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=False,
    )

    assert plan.loop_num_stages == 3
    assert plan.loop_licm


@pytest.mark.parametrize(
    ("is_causal", "key_length", "fuse_query", "unscaled_recurrence"),
    [
        (True, 32 * 1024 - 1, False, False),
        (True, 32 * 1024, True, True),
        (False, 128 * 1024 - 1, True, False),
        (False, 128 * 1024, True, True),
    ],
)
def test_sm120_query_fusion_and_recurrence_thresholds(
    is_causal: bool,
    key_length: int,
    fuse_query: bool,
    unscaled_recurrence: bool,
) -> None:
    plan = select_execution_plan(
        _SM120,
        candidate_block_m=64,
        query_length=key_length,
        key_length=key_length,
        head_dim=128,
        is_causal=is_causal,
    )

    assert plan.fuse_query_quantization is fuse_query
    assert plan.use_unscaled_score_recurrence is unscaled_recurrence


@pytest.mark.parametrize(
    ("is_causal", "fuse_query"),
    [(False, True), (True, False)],
)
def test_sm120_d64_preserves_query_quantization_policy(
    is_causal: bool,
    fuse_query: bool,
) -> None:
    plan = select_execution_plan(
        _SM120,
        candidate_block_m=64,
        query_length=128 * 1024,
        key_length=128 * 1024,
        head_dim=64,
        is_causal=is_causal,
    )

    assert plan.grouped_qk
    assert plan.fuse_kv_quantization
    assert plan.fuse_query_quantization is fuse_query
    assert not plan.use_unscaled_score_recurrence


@pytest.mark.parametrize(
    ("target", "head_dim", "is_causal", "expected"),
    [
        (_SM89, 128, True, True),
        (_SM120, 64, True, True),
        (_SM120, 128, False, True),
        (_SM120, 128, True, False),
    ],
)
def test_probability_conversion_policy_is_specialized_by_target_and_shape(
    target: AcceleratorTarget,
    head_dim: int,
    is_causal: bool,
    expected: bool,
) -> None:
    plan = select_execution_plan(
        target,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    assert plan.use_packed_probability_conversion is expected


@pytest.mark.parametrize(
    ("is_causal", "key_length"),
    [(True, 32 * 1024), (False, 128 * 1024)],
)
def test_other_sm12x_targets_do_not_inherit_sm120_crossovers(
    is_causal: bool,
    key_length: int,
) -> None:
    plan = select_execution_plan(
        _SM121,
        candidate_block_m=128,
        query_length=key_length,
        key_length=key_length,
        head_dim=128,
        is_causal=is_causal,
    )

    assert plan.grouped_qk
    assert not plan.fuse_kv_quantization
    assert not plan.fuse_query_quantization
    assert not plan.use_unscaled_score_recurrence
    assert not plan.use_tensor_descriptors


def test_alternate_plan_can_disable_tensor_descriptors() -> None:
    default_plan = select_execution_plan(
        _SM120,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
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
            query_length=64,
            key_length=64,
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
        {"grouped_qk": False, "fuse_kv_quantization": True},
        {"grouped_qk": False, "fuse_query_quantization": True},
        {"fuse_query_quantization": False, "use_unscaled_score_recurrence": True},
    ],
)
def test_execution_plan_rejects_inconsistent_specializations(
    changes: dict[str, object],
) -> None:
    plan = select_execution_plan(
        _SM120,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=False,
    )

    with pytest.raises(ValueError, match=r"must be|requires"):
        replace(plan, **changes)
