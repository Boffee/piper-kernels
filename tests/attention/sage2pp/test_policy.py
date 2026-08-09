"""Host-side specialization policy tests for SageAttention2++."""

import pytest

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage2pp.triton import (
    _Sage2ppExecutionPlan,
    _select_sage2pp_execution_plan,
)

_SM89 = AcceleratorTarget(backend="cuda", architecture="sm89")
_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")
_SM121 = AcceleratorTarget(backend="cuda", architecture="sm121")


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            _SM89,
            _Sage2ppExecutionPlan(
                block_m=128,
                grouped_qk=False,
                fuse_kv_quantization=False,
                fuse_query_quantization=False,
                use_unscaled_score_recurrence=False,
                use_tensor_descriptors=False,
            ),
        ),
        (
            _SM120,
            _Sage2ppExecutionPlan(
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
            _Sage2ppExecutionPlan(
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
def test_execution_plan_separates_architecture_facts_from_sm120_tuning(
    target: AcceleratorTarget,
    expected: _Sage2ppExecutionPlan,
) -> None:
    plan = _select_sage2pp_execution_plan(
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
        (_SM89, 8192, 64),
    ],
)
def test_causal_block_schedule_is_tuned_only_for_sm120(
    target: AcceleratorTarget,
    query_length: int,
    expected_block_m: int,
) -> None:
    plan = _select_sage2pp_execution_plan(
        target,
        candidate_block_m=128,
        query_length=query_length,
        key_length=query_length,
        head_dim=128,
        is_causal=True,
    )

    assert plan.block_m == expected_block_m


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
    plan = _select_sage2pp_execution_plan(
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
    plan = _select_sage2pp_execution_plan(
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
    ("is_causal", "key_length"),
    [(True, 32 * 1024), (False, 128 * 1024)],
)
def test_other_sm12x_targets_do_not_inherit_sm120_crossovers(
    is_causal: bool,
    key_length: int,
) -> None:
    plan = _select_sage2pp_execution_plan(
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


def test_tensor_descriptor_override_wins_over_sm120_default() -> None:
    default_plan = _select_sage2pp_execution_plan(
        _SM120,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=False,
    )
    pointer_plan = _select_sage2pp_execution_plan(
        _SM120,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=False,
        use_tensor_descriptors=False,
    )

    assert default_plan.use_tensor_descriptors
    assert not pointer_plan.use_tensor_descriptors
