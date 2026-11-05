"""Sparse attention adds its gated projection into the residual without a full output."""

from __future__ import annotations

import pytest
import torch

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile
from piper_kernels.fusions.convrot_int8_sparse_piper import output as output_fusion

from ._helpers import output_available
from .test_attention import _HEAD_DIM, _HEADS, _operands, _prepare
from .test_output import _OUTPUT_FEATURES, _projection


def _coarse_gate_arguments(sequence_length: int) -> dict[str, object]:
    """Coarse residual operands with a gate projected inside each Q window."""
    features = _HEADS * _HEAD_DIM
    query_blocks = (sequence_length + 63) // 64
    gate_weight, gate_scale, gate_bias = _projection(bias=True, bias_dtype=torch.float32)
    assert gate_bias is not None
    return {
        "block_mean": torch.randn(
            (1, _HEADS, query_blocks, _HEAD_DIM), device="cuda", dtype=torch.float32
        ),
        "coarse_scale": _HEAD_DIM**-0.5,
        "coarse_key_blocks": query_blocks,
        "gate_input": torch.randint(
            -127, 128, (1, sequence_length, features), device="cuda", dtype=torch.int8
        ),
        "gate_input_scale": torch.rand(
            (1, sequence_length), device="cuda", dtype=torch.float32
        ).mul_(0.01),
        "gate_weight_qdata": gate_weight[:features, :features].contiguous(),
        "gate_weight_scale": gate_scale[:features].contiguous(),
        "gate_bias": gate_bias[:features].contiguous(),
    }


@pytest.mark.gpu
@pytest.mark.skipif(not output_available(), reason="requires fused sparse output support")
@pytest.mark.parametrize("sequence_length", [193, 512])
@pytest.mark.parametrize("coarse", [False, True])
def test_residual_add_matches_materialized_projected_query_output(
    sequence_length: int,
    coarse: bool,
) -> None:
    operands = _operands(batch=1, sequence_length=sequence_length)
    _prepared_query, prepared_key, prepared_value = _prepare(operands)
    attention = SparsePiperAttention((0.5, 1.0))
    weight, scale, bias = _projection(bias=True)
    attention_operands = (
        operands.input_qdata,
        operands.input_scale,
        operands.query_weight,
        operands.query_weight_scale,
        operands.query_norm,
        operands.cos,
        operands.sin,
        1e-5,
        _HEAD_DIM**-0.5,
        *prepared_key,
        *prepared_value,
        attention._ratio_profile,
        sequence_length // 64,
        sequence_length,
        attention._routing_mode,
        weight,
        scale,
        bias,
        _HEADS * _HEAD_DIM,
        # Several Q windows overlap projection with attention on a consumer stream.
        64,
    )
    optional = _coarse_gate_arguments(sequence_length) if coarse else {}
    with torch.no_grad():
        projected = output_fusion._projected_query_attention_output_op(
            *attention_operands, **optional
        )
    torch.manual_seed(911)
    base = torch.randn_like(projected)
    gate_storage = torch.randn(5, 3 * _OUTPUT_FEATURES, device="cuda", dtype=base.dtype)
    gate = gate_storage[:, _OUTPUT_FEATURES : 2 * _OUTPUT_FEATURES]
    gate_indices = torch.randint(-5, 5, (base.shape[0] * base.shape[1],), device="cuda")
    selected = gate[gate_indices].view(base.shape)
    expected = (base.float() + selected.float() * projected.float()).to(base.dtype)

    with torch.no_grad():
        result = output_fusion._projected_query_attention_residual_add_op(
            base, gate, gate_indices, True, *attention_operands, **optional
        )

    assert result is None
    # Both paths share the attention and projection kernels; the residual path adds
    # each projected window in FP32 instead of rounding a materialized output first.
    torch.testing.assert_close(base, expected, atol=2e-2, rtol=2**-7)
    error_norm = torch.linalg.vector_norm(base.float() - expected.float())
    assert error_norm <= 5e-4 * torch.linalg.vector_norm(expected.float())


def _fake_arguments(base: torch.Tensor) -> tuple[object, ...]:
    """Metadata for a padded 192-row output of a 132-token sequence."""
    meta = {"device": "meta"}
    gate = torch.empty((5, _OUTPUT_FEATURES), dtype=base.dtype, **meta)
    gate_indices = torch.empty(base.shape[0] * base.shape[1], dtype=torch.int64, **meta)
    return (
        base,
        gate,
        gate_indices,
        False,
        torch.empty((2, 132, 384), dtype=torch.int8, **meta),
        torch.empty((2, 132), dtype=torch.float32, **meta),
        torch.empty((384, 384), dtype=torch.int8, **meta),
        torch.empty((384, 6), dtype=torch.float32, **meta),
        None,
        torch.empty((132, 96), dtype=torch.float32, **meta),
        torch.empty((132, 96), dtype=torch.float32, **meta),
        1e-5,
        128**-0.5,
        torch.empty((2, 3, 192, 128), dtype=torch.int8, **meta),
        torch.empty((2, 3, 3), dtype=torch.float32, **meta),
        torch.empty((2, 3, 3, 128), dtype=torch.float32, **meta),
        torch.empty((2, 3, 3, 128), dtype=torch.float32, **meta),
        torch.empty((2, 3, 128, 192), dtype=torch.int8, **meta),
        torch.empty((2, 3, 3, 1), dtype=torch.float32, **meta),
        torch.empty((2, 3, 128), dtype=torch.float32, **meta),
        _SparsePiperRatioProfile((500_000, 1_000_000, 750_000)),
        2,
        132,
        0,
        torch.empty((_OUTPUT_FEATURES, 384), dtype=torch.int8, **meta),
        torch.empty((_OUTPUT_FEATURES, 1), dtype=torch.float32, **meta),
        None,
        128,
        64,
        torch.empty(3, dtype=torch.int32, **meta),
    )


def test_residual_add_fake_accepts_the_padded_output_rows() -> None:
    base = torch.empty((2, 192, _OUTPUT_FEATURES), device="meta", dtype=torch.bfloat16)

    assert (
        output_fusion._projected_query_attention_residual_add_op_fake(*_fake_arguments(base))
        is None
    )


@pytest.mark.parametrize(
    "base",
    [
        torch.empty((2, 132, _OUTPUT_FEATURES), device="meta", dtype=torch.bfloat16),
        torch.empty((1, 192, _OUTPUT_FEATURES), device="meta", dtype=torch.bfloat16),
    ],
    ids=["logical-rows", "batch"],
)
def test_residual_add_requires_one_base_row_per_output_row(base: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="one row per output row"):
        output_fusion._projected_query_attention_residual_add_op_fake(*_fake_arguments(base))
