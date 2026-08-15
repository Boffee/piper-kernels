"""Host-side specialization policy tests for Piper Attention."""

from dataclasses import replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._policy import (
    PiperAttentionExecutionPlan,
    select_execution_plan,
)
from piper_kernels.attention.piper_attention.triton import (
    _default_piper_attention_execution_plan,
    _prepare_piper_attention,
)

_SM80 = AcceleratorTarget(backend="cuda", architecture="sm80")
_SM89 = AcceleratorTarget(backend="cuda", architecture="sm89")
_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")
_SM121 = AcceleratorTarget(backend="cuda", architecture="sm121")


def _select(
    target: AcceleratorTarget,
    *,
    head_dim: int = 128,
    is_causal: bool = False,
) -> PiperAttentionExecutionPlan:
    return select_execution_plan(
        target,
        head_dim=head_dim,
        is_causal=is_causal,
    )


@pytest.mark.parametrize("sequence", [1, 8192, 131073])
def test_default_execution_plan_is_sequence_length_invariant(sequence: int) -> None:
    query = torch.empty((1, 8, sequence, 128), device="meta")

    plan = _default_piper_attention_execution_plan(
        query,
        False,
        target=_SM120,
    )

    assert plan.block_m == 128
    assert plan.grouped_qk
    assert plan.split_pv_head_dim
    assert plan.derive_value_log_bound
    assert plan.use_packed_probability_conversion


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            _SM80,
            PiperAttentionExecutionPlan(
                block_m=128,
                grouped_qk=False,
                split_pv_head_dim=False,
                use_tensor_descriptors=False,
            ),
        ),
        (
            _SM89,
            PiperAttentionExecutionPlan(
                block_m=128,
                grouped_qk=False,
                split_pv_head_dim=True,
                use_tensor_descriptors=False,
                num_stages=1,
                loop_num_stages=3,
                loop_licm=True,
                use_packed_probability_conversion=True,
            ),
        ),
        (
            _SM120,
            PiperAttentionExecutionPlan(
                block_m=128,
                grouped_qk=True,
                split_pv_head_dim=True,
                use_tensor_descriptors=True,
                derive_value_log_bound=True,
                num_stages=2,
                use_packed_probability_conversion=True,
            ),
        ),
        (
            _SM121,
            PiperAttentionExecutionPlan(
                block_m=64,
                grouped_qk=True,
                split_pv_head_dim=True,
                use_tensor_descriptors=False,
            ),
        ),
    ],
)
def test_execution_plan_separates_architecture_facts_from_exact_target_tuning(
    target: AcceleratorTarget,
    expected: PiperAttentionExecutionPlan,
) -> None:
    plan = _select(target)

    assert plan == expected


@pytest.mark.parametrize(
    ("head_dim", "is_causal", "selected"),
    [
        (128, False, True),
        (64, False, False),
        (128, True, False),
    ],
)
def test_sm89_noncausal_d128_policy_is_dimension_and_mode_specific(
    head_dim: int,
    is_causal: bool,
    selected: bool,
) -> None:
    plan = _select(
        _SM89,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    assert plan.split_pv_head_dim is selected
    assert plan.use_packed_probability_conversion is selected
    assert plan.loop_licm is selected
    assert plan.loop_num_stages == (3 if selected else None)
    assert plan.num_stages == (1 if selected else 3)
    assert plan.block_m == (64 if is_causal else 128)
    assert not plan.use_tensor_descriptors


@pytest.mark.parametrize(
    ("head_dim", "is_causal", "expected"),
    [
        (64, False, True),
        (64, True, True),
        (128, False, True),
        (128, True, False),
    ],
)
def test_sm120_probability_conversion_policy(
    head_dim: int,
    is_causal: bool,
    expected: bool,
) -> None:
    plan = _select(_SM120, head_dim=head_dim, is_causal=is_causal)

    assert plan.use_packed_probability_conversion is expected


@pytest.mark.parametrize(
    ("head_dim", "is_causal", "expected"),
    [
        (64, False, True),
        (128, False, True),
        (64, True, False),
        (128, True, False),
    ],
)
def test_sm120_derived_value_log_policy_is_mode_specific(
    head_dim: int,
    is_causal: bool,
    expected: bool,
) -> None:
    plan = _select(
        _SM120,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    assert plan.derive_value_log_bound is expected


@pytest.mark.parametrize(
    (
        "head_dim",
        "is_causal",
        "block_m",
        "split_pv",
        "descriptors",
    ),
    [
        (64, False, 128, False, False),
        (128, False, 128, True, True),
        (64, True, 64, False, False),
        (128, True, 64, True, False),
    ],
)
def test_sm120_tiling_is_dimension_and_mode_specific(
    head_dim: int,
    is_causal: bool,
    block_m: int,
    split_pv: bool,
    descriptors: bool,
) -> None:
    plan = _select(
        _SM120,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    assert plan.block_m == block_m
    assert plan.split_pv_head_dim is split_pv
    assert plan.use_tensor_descriptors is descriptors
    assert plan.num_stages == (2 if descriptors else 3)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    ("is_causal", "expected"),
    [
        (True, True),
        (False, False),
    ],
)
def test_sm120_optimized_causal_traversal_policy_is_mode_specific(
    head_dim: int,
    is_causal: bool,
    expected: bool,
) -> None:
    plan = _select(
        _SM120,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    assert plan.optimize_causal_traversal is expected


def test_sm89_does_not_inherit_sage_attention_schedule() -> None:
    plan = _select(_SM89, is_causal=True)

    assert plan.block_m == 64
    assert not plan.optimize_causal_traversal
    assert plan.loop_num_stages is None
    assert not plan.loop_licm


def test_unmeasured_sm12x_target_does_not_inherit_sm120_causal_policy() -> None:
    plan = _select(_SM121, is_causal=True)

    assert not plan.optimize_causal_traversal


def test_alternate_plan_can_disable_tensor_descriptors() -> None:
    descriptor_plan = _select(_SM120)
    pointer_plan = replace(
        descriptor_plan,
        use_tensor_descriptors=False,
        num_stages=3,
    )

    assert descriptor_plan.use_tensor_descriptors
    assert descriptor_plan.num_stages == 2
    assert not pointer_plan.use_tensor_descriptors
    assert pointer_plan.num_stages == 3


def test_execution_plan_serializes_all_launch_choices() -> None:
    plan = replace(
        _select(_SM120, is_causal=True),
        optimize_causal_traversal=True,
        loop_num_stages=2,
        loop_licm=True,
    )

    assert plan.as_dict() == {
        "block_m": 64,
        "grouped_qk": True,
        "split_pv_head_dim": True,
        "use_tensor_descriptors": False,
        "derive_value_log_bound": False,
        "optimize_causal_traversal": True,
        "num_warps": 4,
        "num_stages": 3,
        "loop_num_stages": 2,
        "loop_licm": True,
        "use_packed_probability_conversion": False,
    }


def test_execution_plan_rejects_optimized_traversal_for_noncausal_invocation() -> None:
    query = torch.empty((1, 1, 64, 64), device="meta")
    plan = replace(
        _select(_SM80, head_dim=64),
        optimize_causal_traversal=True,
    )

    with pytest.raises(ValueError, match="requires causal attention"):
        _prepare_piper_attention(
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
def test_execution_plan_rejects_invalid_launch_choices(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"must be|requires"):
        replace(_select(_SM120), **changes)
