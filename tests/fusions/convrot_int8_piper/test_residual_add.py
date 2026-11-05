"""Dense attention adds its gated projection into the residual without a full output."""

import pytest
import torch

from piper_kernels.fusions.convrot_int8_piper import output

from .test_output import _arguments
from .test_query import _NATIVE

_OUTPUT_DTYPE_INDEX = 18


def _residual_operands(arguments, device):
    batch, sequence = arguments[0].shape[:2]
    output_features = arguments[19].shape[0]
    dtype = arguments[_OUTPUT_DTYPE_INDEX]
    torch.manual_seed(911)
    base = torch.randn(batch, sequence, output_features, device=device, dtype=dtype)
    gate_storage = torch.randn(5, 3 * output_features, device=device, dtype=dtype)
    gate = gate_storage[:, output_features : 2 * output_features]
    gate_indices = torch.randint(-5, 5, (batch * sequence,), device=device)
    return base, gate, gate_indices


def _attention_operands(arguments):
    """Drop the output dtype, which the residual add takes from its base."""
    return (*arguments[:_OUTPUT_DTYPE_INDEX], *arguments[_OUTPUT_DTYPE_INDEX + 1 :])


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sequence", [65, 385])
@pytest.mark.parametrize("output_expansion", [None, 64])
def test_residual_add_matches_materialized_attention_output(
    head_dim, causal, sequence, output_expansion
):
    arguments = _arguments(
        head_dim=head_dim,
        sequence=sequence,
        causal=causal,
        output_features=80 if output_expansion is None else 3 * head_dim + output_expansion,
    )
    base, gate, gate_indices = _residual_operands(arguments, "cuda")
    projected = output._projected_query_attention_output_op(
        *arguments, head_dim=head_dim, query_chunk_rows=128
    )
    selected = gate[gate_indices].view(base.shape)
    expected = (base.float() + selected.float() * projected.float()).to(base.dtype)

    result = output._projected_query_attention_residual_add_op(
        base,
        gate,
        gate_indices,
        True,
        *_attention_operands(arguments),
        head_dim=head_dim,
        query_chunk_rows=128,
    )

    assert result is None
    # Both paths share the projection kernels; besides FP32 multiply-add contraction,
    # attention rounding on some targets can flip an INT8 output-activation code
    # (see test_output.py). Bound local outliers and aggregate drift.
    torch.testing.assert_close(base, expected, atol=2e-2, rtol=2**-7)
    error_norm = torch.linalg.vector_norm(base.float() - expected.float())
    assert error_norm <= 5e-4 * torch.linalg.vector_norm(expected.float())


def test_residual_add_requires_one_base_row_per_query_token():
    arguments = _arguments(sequence=65, device="meta")
    base, gate, gate_indices = _residual_operands(arguments, "meta")
    with pytest.raises(ValueError, match="one row per query token"):
        output._projected_query_attention_residual_add_op_fake(
            base[:, 1:],
            gate,
            gate_indices,
            True,
            *_attention_operands(arguments),
            head_dim=64,
        )


def test_residual_add_validates_attention_operands_from_metadata():
    # Causal attention needs equal query and key lengths; these keys are 17 rows longer.
    arguments = list(_arguments(sequence=65, device="meta"))
    arguments[17] = True
    base, gate, gate_indices = _residual_operands(arguments, "meta")
    with pytest.raises(ValueError, match="square causal attention"):
        output._projected_query_attention_residual_add_op_fake(
            base,
            gate,
            gate_indices,
            True,
            *_attention_operands(arguments),
            head_dim=64,
        )
