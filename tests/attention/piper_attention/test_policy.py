"""Host-side specialization policy tests for Piper Attention."""

from dataclasses import fields, replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._nvidia._plan import PiperAttentionExecutionPlan
from piper_kernels.attention.piper_attention._nvidia.policy import (
    groups_qk_scales,
    select_execution_plan,
)
from piper_kernels.attention.piper_attention._nvidia.triton import (
    _prepare_piper_attention,
    default_execution_plan,
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
    query_length: int = 128,
    key_length: int | None = None,
    quantized_query: bool = False,
) -> PiperAttentionExecutionPlan:
    return select_execution_plan(
        target,
        head_dim=head_dim,
        is_causal=is_causal,
        query_length=query_length,
        key_length=key_length,
        quantized_query=quantized_query,
    )


def test_noncausal_d128_execution_plan_is_sequence_length_invariant() -> None:
    plans = [
        default_execution_plan(
            torch.empty((1, 8, sequence, 128), device="meta"), False, target=_SM120
        )
        for sequence in (1, 8192, 131073)
    ]
    assert all(plan == plans[0] for plan in plans)
    plans[0].validate_kernel()
    assert plans[0].grouped_qk is True
    assert plans[0].optimize_causal_traversal is False


@pytest.mark.parametrize(("batch", "heads"), [(1, 1), (4, 48)])
def test_sm120_causal_d128_descriptors_switch_monotonically_with_query_length(
    batch: int,
    heads: int,
) -> None:
    sequences = (128, 512, 1023, 1024, 4096, 131073)
    plans = [
        default_execution_plan(
            torch.empty((batch, heads, sequence, 128), device="meta"), True, target=_SM120
        )
        for sequence in sequences
    ]
    descriptor_flags = [plan.use_tensor_descriptors for plan in plans]
    assert descriptor_flags[0] is False
    assert descriptor_flags[-1] is True
    assert descriptor_flags == sorted(descriptor_flags)
    # Only descriptor usage changes across the crossover; other launch choices stay consistent.
    for plan in plans:
        assert replace(plan, use_tensor_descriptors=False) == replace(
            plans[0], use_tensor_descriptors=False
        )


@pytest.mark.parametrize("target", [_SM80, _SM89, _SM121])
@pytest.mark.parametrize("sequence", [1024, 131073])
def test_other_targets_retain_causal_d128_load_schedule(
    target: AcceleratorTarget,
    sequence: int,
) -> None:
    query = torch.empty((1, 16, sequence, 128), device="meta")

    plan = default_execution_plan(query, True, target=target)

    assert plan == _select(target, is_causal=True)
    assert not plan.use_tensor_descriptors


@pytest.mark.parametrize("target", [_SM80, _SM89, _SM120, _SM121])
@pytest.mark.parametrize("query_length", [65, 8193])
def test_d64_descriptors_depend_on_key_traversal_not_query_length(target, query_length):
    key_lengths = (64, 1024, 4095, 4096, 131073)
    plans = [
        default_execution_plan(
            torch.empty((2, 6, query_length, 64), device="meta"),
            False,
            target=target,
            key_length=key_length,
        )
        for key_length in key_lengths
    ]
    for plan in plans:
        plan.validate_kernel()
    descriptor_flags = [plan.use_tensor_descriptors for plan in plans]
    if target == _SM120:
        assert descriptor_flags[0] is False
        assert descriptor_flags[-1] is True
        assert descriptor_flags == sorted(descriptor_flags)
        # Same key_length yields the same descriptor decision regardless of query_length.
        other = default_execution_plan(
            torch.empty((2, 6, 128, 64), device="meta"),
            False,
            target=target,
            key_length=key_lengths[-1],
        )
        assert other.use_tensor_descriptors is plans[-1].use_tensor_descriptors
    else:
        assert not any(descriptor_flags)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sequence", [128, 4096, 131073])
def test_d64_self_attention_default_key_length_matches_explicit_metadata(causal, sequence):
    plan = _select(_SM120, head_dim=64, is_causal=causal, query_length=sequence)
    assert plan == _select(
        _SM120, head_dim=64, is_causal=causal, query_length=sequence, key_length=sequence
    )
    plan.validate_kernel()


@pytest.mark.parametrize("target", [_SM80, _SM89, _SM120, _SM121])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("sequence", [64, 65, 8192, 8193])
def test_all_selected_plans_satisfy_capability_and_safety_invariants(
    target: AcceleratorTarget,
    head_dim: int,
    is_causal: bool,
    sequence: int,
) -> None:
    plan = default_execution_plan(
        torch.empty((1, 8, sequence, head_dim), device="meta"), is_causal, target=target
    )
    plan.validate_kernel()
    assert plan.block_m in (64, 128)
    assert plan.num_warps in (2, 4, 8)
    assert 1 <= plan.num_stages <= 4
    assert plan.grouped_qk is groups_qk_scales(target)
    if not is_causal:
        assert not plan.optimize_causal_traversal
    if head_dim == 64:
        assert not plan.split_pv_head_dim
    # Exact-SM120 tuning knobs and tensor descriptors never leak to other targets.
    if target != _SM120:
        assert not plan.use_tensor_descriptors
        assert not plan.optimize_causal_traversal
        assert not plan.loop_licm
        assert not plan.retain_query_tail_for_strided_output
        assert plan.output_ctas_per_sm == 0
        assert plan.strided_output_query_group == 0
        assert plan.ragged_strided_output_maxnreg is None
    # SM89 uses the Gluon async-copy kernel; other targets use Triton.
    assert plan.attention_kernel == ("gluon_async_copy" if target == _SM89 else "triton")
    if target == _SM120:
        assert plan.optimize_causal_traversal is is_causal
        assert plan.loop_licm is (head_dim == 64 and is_causal and sequence % plan.block_m != 0)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm89_quantized_query_disables_fused_quantization(head_dim: int, is_causal: bool) -> None:
    floating_plan = _select(_SM89, head_dim=head_dim, is_causal=is_causal, quantized_query=False)
    quantized_plan = _select(_SM89, head_dim=head_dim, is_causal=is_causal, quantized_query=True)

    floating_plan.validate_kernel()
    quantized_plan.validate_kernel()
    assert floating_plan.attention_kernel == quantized_plan.attention_kernel == "gluon_async_copy"
    assert not quantized_plan.fuse_query_quantization
    assert quantized_plan.block_m <= floating_plan.block_m


@pytest.mark.parametrize("target", [_SM80, _SM120, _SM121])
@pytest.mark.parametrize("is_causal", [False, True])
def test_quantized_query_leaves_other_targets_unchanged(
    target: AcceleratorTarget, is_causal: bool
) -> None:
    for head_dim in (64, 128):
        assert _select(
            target, head_dim=head_dim, is_causal=is_causal, quantized_query=True
        ) == _select(target, head_dim=head_dim, is_causal=is_causal)


@pytest.mark.parametrize(
    "architecture", ["sm80", "sm86", "sm89", "sm90", "sm100", "sm120", "sm121"]
)
def test_qk_scale_grouping_depends_only_on_target(architecture: str) -> None:
    target = AcceleratorTarget("cuda", architecture)
    for head_dim in (64, 128):
        for is_causal in (False, True):
            for query_length in (1, 1023, 1024, 65537):
                for quantized_query in (False, True):
                    plan = select_execution_plan(
                        target,
                        head_dim=head_dim,
                        is_causal=is_causal,
                        query_length=query_length,
                        quantized_query=quantized_query,
                    )
                    assert plan.grouped_qk is groups_qk_scales(target)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fuse_query_quantization": True}, "fused query quantization"),
        ({"max_registers": 232}, "register cap"),
    ],
)
def test_gluon_only_choices_require_the_gluon_kernel(
    changes: dict[str, object],
    message: str,
) -> None:
    query = torch.empty((1, 1, 64, 64), device="meta")
    triton_plan = replace(
        _select(_SM89, head_dim=64),
        attention_kernel="triton",
        max_registers=None,
        fuse_query_quantization=False,
    )
    plan = replace(triton_plan, **changes)

    with pytest.raises(ValueError, match=f"{message} requires the Gluon kernel"):
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
        {"num_warps": 8},
        {"grouped_qk": True},
        {"split_pv_head_dim": True},
        {"derive_value_log_bound": False},
        {"retain_query_tail_for_strided_output": True},
        {"strided_output_query_group": 8},
        {"ragged_strided_output_maxnreg": 168},
    ],
)
def test_gluon_plan_rejects_unsupported_recurrence_choices(
    changes: dict[str, object],
) -> None:
    query = torch.empty((1, 1, 64, 128), device="meta")
    plan = replace(_select(_SM89), **changes)

    with pytest.raises(ValueError, match="Gluon kernel requires"):
        _prepare_piper_attention(
            query,
            query,
            query,
            0.125,
            False,
            execution_plan=plan,
        )


def test_alternate_plan_can_disable_tensor_descriptors() -> None:
    descriptor_plan = _select(_SM120)
    pointer_plan = replace(
        descriptor_plan,
        use_tensor_descriptors=False,
        num_stages=3,
    )

    assert descriptor_plan.use_tensor_descriptors
    assert not pointer_plan.use_tensor_descriptors
    pointer_plan.validate_kernel()


def test_execution_plan_serializes_all_launch_choices() -> None:
    plan = replace(
        _select(_SM120, is_causal=True),
        optimize_causal_traversal=True,
        loop_num_stages=2,
        loop_licm=True,
    )
    serialized = plan.as_dict()
    expected_keys = {field_info.name for field_info in fields(PiperAttentionExecutionPlan)}
    assert set(serialized) == expected_keys
    for key in expected_keys:
        assert serialized[key] == getattr(plan, key)


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
