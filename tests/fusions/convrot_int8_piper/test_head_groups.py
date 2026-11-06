"""Head-grouped dense attention adds the residual update while holding one group's K and V."""

import pytest
import torch

from piper_kernels.fusions.convrot_int8_piper import _head_groups, key, output, value

from .test_query import _NATIVE, _operands

_HEAD_DIM = 64
_HEADS = 4
_GROUP_SIZE = 64
_EPSILON = 1e-6


def _attention(device, *, sequence, causal, output_features=96):
    query = _operands(device, sequence=sequence, heads=_HEADS, head_dim=_HEAD_DIM)
    context = _operands(
        device,
        sequence=sequence if causal else sequence + 17,
        heads=_HEADS,
        head_dim=_HEAD_DIM,
    )
    torch.manual_seed(907)
    weight = torch.randint(
        -100, 100, (output_features, _HEADS * _HEAD_DIM), device=device, dtype=torch.int8
    )
    scales = torch.full((output_features, 1), 0.001, device=device)
    bias = torch.randn(output_features, device=device, dtype=torch.bfloat16)
    base = torch.randn(2, sequence, output_features, device=device, dtype=torch.bfloat16)
    gate_storage = torch.randn(5, 3 * output_features, device=device, dtype=torch.bfloat16)
    gate = gate_storage[:, output_features : 2 * output_features]
    gate_indices = torch.randint(-5, 5, (2 * sequence,), device=device)
    return query, context, (weight, scales, bias), (base, gate, gate_indices)


def _add_ungrouped(query, context, projection, residual, *, causal):
    group_key = key._project_key_op(*context, _EPSILON, head_dim=_HEAD_DIM)
    group_value = value._project_value_op(*context[:4], head_dim=_HEAD_DIM, is_causal=causal)
    output._projected_query_attention_residual_add_op(
        *residual,
        True,
        *query,
        _EPSILON,
        _HEAD_DIM**-0.5,
        None,
        *group_key,
        *group_value,
        context[0].shape[1],
        causal,
        *projection,
        _GROUP_SIZE,
        None,
        head_dim=_HEAD_DIM,
        query_chunk_rows=128,
    )


def _add_grouped(query, context, projection, residual, *, causal, head_groups, fake=False):
    operator = (
        _head_groups._head_grouped_attention_residual_add_op_fake
        if fake
        else _head_groups._head_grouped_attention_residual_add_op
    )
    operator(
        *residual,
        True,
        *query,
        _EPSILON,
        _HEAD_DIM**-0.5,
        None,
        *context,
        _EPSILON,
        None,
        *context[:4],
        None,
        causal,
        *projection,
        _GROUP_SIZE,
        None,
        head_dim=_HEAD_DIM,
        head_groups=head_groups,
        query_chunk_rows=128,
    )


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("causal", [False, True])
def test_one_head_group_matches_separate_producers_exactly(causal):
    query, context, projection, residual = _attention("cuda", sequence=193, causal=causal)
    expected = residual[0].clone()
    _add_ungrouped(query, context, projection, (expected, *residual[1:]), causal=causal)

    _add_grouped(query, context, projection, residual, causal=causal, head_groups=1)

    torch.testing.assert_close(residual[0], expected, rtol=0, atol=0)


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sequence", [65, 385])
def test_head_groups_add_the_same_attention_update(causal, sequence):
    query, context, projection, residual = _attention("cuda", sequence=sequence, causal=causal)
    original = residual[0].clone()
    expected = residual[0].clone()
    _add_ungrouped(query, context, projection, (expected, *residual[1:]), causal=causal)

    _add_grouped(query, context, projection, residual, causal=causal, head_groups=2)

    # Each group quantizes its own output-projection input per token, and the base
    # rounds once per group. Two independent INT8 roundings of the same activations
    # differ by about 0.7% on RDNA4; adding a group or the bias twice would differ by
    # about 50%.
    update = residual[0].float() - original.float()
    expected_update = expected.float() - original.float()
    error = torch.linalg.vector_norm(update - expected_update)
    assert error <= 1.5e-2 * torch.linalg.vector_norm(expected_update)


@pytest.mark.gpu
@_NATIVE
def test_head_groups_hold_one_groups_context_at_a_time():
    query, context, projection, residual = _attention("cuda", sequence=8192, causal=False)
    torch.cuda.synchronize()

    def peak_extra(add):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        add()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - start

    ungrouped = peak_extra(
        lambda: _add_ungrouped(query, context, projection, residual, causal=False)
    )
    grouped = peak_extra(
        lambda: _add_grouped(query, context, projection, residual, causal=False, head_groups=_HEADS)
    )

    # K, V, and K's BF16 workspace dominate; four groups hold a quarter of each.
    assert grouped <= 0.6 * ungrouped


@pytest.mark.parametrize(
    ("head_groups", "group_size", "message"),
    [
        (3, _GROUP_SIZE, "must divide 4 query and 4 key heads"),
        (4, 128, "multiple of the output projection's ConvRot group size 128"),
        (0, _GROUP_SIZE, "must be positive"),
    ],
)
def test_head_groups_reject_a_split_that_does_not_fit_from_metadata(
    head_groups, group_size, message
):
    query, context, projection, _ = _attention("meta", sequence=65, causal=False)
    with pytest.raises(ValueError, match=message):
        _head_groups.validate_head_groups(
            query[2],
            context[2],
            context[2],
            projection[0],
            group_size,
            head_dim=_HEAD_DIM,
            head_groups=head_groups,
        )


def test_head_groups_validate_group_operands_from_metadata():
    # Causal attention needs equal query and key lengths; these keys are 17 rows longer.
    query, context, projection, residual = _attention("meta", sequence=65, causal=False)
    with pytest.raises(ValueError, match="square causal attention"):
        _add_grouped(query, context, projection, residual, causal=True, head_groups=2, fake=True)
