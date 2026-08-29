"""Tests for bounded sparse-attention-to-ConvRot output fusion."""

from __future__ import annotations

import pytest
import torch

from piper_kernels import SparsePiperAttention
from piper_kernels.fusions.convrot_sparse_piper import output as output_fusion
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

from .test_attention import (
    _HEAD_DIM,
    _HEADS,
    _operands,
    _prepare,
    _run_sparse_piper_attention_from_quantized,
)

_OUTPUT_FEATURES = 320


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _projection(*, bias: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    weight = torch.randint(
        -127,
        128,
        (_OUTPUT_FEATURES, _HEADS * _HEAD_DIM),
        device="cuda",
        dtype=torch.int8,
    )
    scale = (
        torch.rand((_OUTPUT_FEATURES, 1), device="cuda", dtype=torch.float32).mul_(0.01).add_(0.001)
    )
    projected_bias = (
        torch.randn(_OUTPUT_FEATURES, device="cuda", dtype=torch.bfloat16) if bias else None
    )
    return weight, scale, projected_bias


def _arguments(
    *,
    batch: int,
    sequence_length: int,
    bias: bool,
) -> tuple[tuple[object, ...], torch.Tensor]:
    operands = _operands(batch=batch, sequence_length=sequence_length)
    prepared_query, prepared_key, prepared_value = _prepare(operands)
    attention = SparsePiperAttention((0.5, 1.0))
    sparse_key_blocks = max(1, sequence_length // 64)
    weight, scale, projected_bias = _projection(bias=bias)
    arguments = (
        *prepared_query,
        *prepared_key,
        *prepared_value,
        list(attention._head_keep_ratio_units),
        sparse_key_blocks,
        sequence_length,
        weight,
        scale,
        projected_bias,
        _HEADS * _HEAD_DIM,
    )
    with torch.no_grad():
        materialized_attention = _run_sparse_piper_attention_from_quantized(
            prepared_query,
            prepared_key,
            prepared_value,
            attention,
            logical_sequence_length=sequence_length,
            sparse_key_blocks=sparse_key_blocks,
        )
        expected = convrot_backend.run_linear(
            materialized_attention.reshape(batch, sequence_length, _HEADS * _HEAD_DIM),
            weight,
            scale,
            projected_bias,
            _HEADS * _HEAD_DIM,
        )
    return arguments, expected


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("batch", "sequence_length", "query_chunk_rows", "bias"),
    [(1, 64, 64, False), (1, 65, 64, True), (2, 193, 128, True)],
)
def test_attention_output_matches_materialized_boundary(
    batch: int,
    sequence_length: int,
    query_chunk_rows: int,
    bias: bool,
) -> None:
    arguments, expected = _arguments(
        batch=batch,
        sequence_length=sequence_length,
        bias=bias,
    )

    with torch.no_grad():
        actual = output_fusion._attention_output_op(*arguments, query_chunk_rows)

    assert actual.shape == (batch, sequence_length, _OUTPUT_FEATURES)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_attention_output_obeys_a_nondefault_current_stream() -> None:
    arguments, expected = _arguments(batch=1, sequence_length=193, bias=False)
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        actual = output_fusion._attention_output_op(*arguments, 128)
    stream.synchronize()

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_attention_output_custom_op_passes_opcheck() -> None:
    arguments, _expected = _arguments(batch=1, sequence_length=128, bias=True)

    with torch.no_grad():
        result = torch.library.opcheck(
            output_fusion._attention_output_op,
            (*arguments, 64),
        )

    assert set(result.values()) == {"SUCCESS"}


def test_attention_output_fake_kernel_propagates_shape() -> None:
    output = output_fusion._attention_output_op(
        torch.empty((2, 3, 192, 128), device="meta", dtype=torch.int8),
        torch.empty((2, 3, 6), device="meta", dtype=torch.float32),
        torch.empty((2, 3, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((2, 3, 192, 128), device="meta", dtype=torch.int8),
        torch.empty((2, 3, 3), device="meta", dtype=torch.float32),
        torch.empty((2, 3, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((2, 3, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((2, 3, 128, 192), device="meta", dtype=torch.int8),
        torch.empty((2, 3, 3, 1), device="meta", dtype=torch.float32),
        torch.empty((2, 3, 128), device="meta", dtype=torch.float32),
        [500_000, 1_000_000, 750_000],
        2,
        191,
        torch.empty((_OUTPUT_FEATURES, 3 * 128), device="meta", dtype=torch.int8),
        torch.empty((_OUTPUT_FEATURES, 1), device="meta", dtype=torch.float32),
        None,
        128,
        64,
    )

    assert output.shape == (2, 191, _OUTPUT_FEATURES)
    assert output.dtype is torch.bfloat16


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("query_chunk_rows", [0, 63, 65])
def test_attention_output_rejects_invalid_query_chunk_rows(query_chunk_rows: int) -> None:
    arguments, _expected = _arguments(batch=1, sequence_length=128, bias=False)

    with pytest.raises(ValueError, match="positive multiple of 64"):
        output_fusion._attention_output_op(*arguments, query_chunk_rows)
