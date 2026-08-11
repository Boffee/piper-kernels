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
    query_length: int = 8192,
    key_length: int = 8192,
    head_dim: int = 128,
    is_causal: bool = False,
) -> PiperAttentionExecutionPlan:
    return select_execution_plan(
        target,
        candidate_block_m=128,
        query_length=query_length,
        key_length=key_length,
        head_dim=head_dim,
        is_causal=is_causal,
    )


def test_default_execution_plan_supports_meta_tensors_with_resolved_target() -> None:
    query = torch.empty((1, 8, 8192, 128), device="meta")

    plan = _default_piper_attention_execution_plan(
        query,
        query,
        False,
        target=_SM120,
    )

    assert plan.block_m == 128
    assert plan.grouped_qk
    assert plan.native_uint8
    assert plan.split_pv_head_dim
    assert plan.use_packed_probability_conversion


@pytest.mark.parametrize(
    ("target", "grouped_qk", "split_pv", "descriptors", "packed_probability"),
    [
        (_SM80, False, False, False, False),
        (_SM89, False, False, False, False),
        (_SM120, True, True, True, True),
        (_SM121, True, True, True, False),
    ],
)
def test_execution_plan_preserves_existing_architecture_policy(
    target: AcceleratorTarget,
    grouped_qk: bool,
    split_pv: bool,
    descriptors: bool,
    packed_probability: bool,
) -> None:
    plan = _select(target)

    assert plan.grouped_qk is grouped_qk
    assert plan.split_pv_head_dim is split_pv
    assert plan.scaled_fp16_numerator is split_pv
    assert plan.use_tensor_descriptors is descriptors
    assert plan.use_packed_probability_conversion is packed_probability
    assert plan.num_stages == (2 if descriptors else 3)


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


def test_sm89_does_not_inherit_sage_attention_schedule() -> None:
    plan = _select(_SM89, is_causal=True)

    assert plan.block_m == 64
    assert not plan.reverse_causal_blocks
    assert plan.loop_num_stages is None
    assert not plan.loop_licm


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
        reverse_causal_blocks=True,
        loop_num_stages=2,
        loop_licm=True,
    )

    assert plan.as_dict() == {
        "block_m": 64,
        "grouped_qk": True,
        "native_uint8": True,
        "split_pv_head_dim": False,
        "scaled_fp16_numerator": False,
        "use_tensor_descriptors": False,
        "num_warps": 4,
        "num_stages": 3,
        "reverse_causal_blocks": True,
        "loop_num_stages": 2,
        "loop_licm": True,
        "use_packed_probability_conversion": False,
    }


def test_execution_plan_rejects_reverse_order_for_noncausal_invocation() -> None:
    query = torch.empty((1, 1, 64, 64), device="meta")
    plan = replace(
        _select(_SM80, query_length=64, key_length=64, head_dim=64),
        reverse_causal_blocks=True,
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
        {"split_pv_head_dim": False, "scaled_fp16_numerator": True},
        {"native_uint8": False, "use_packed_probability_conversion": True},
    ],
)
def test_execution_plan_rejects_inconsistent_specializations(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"must be|requires"):
        replace(_select(_SM120), **changes)
