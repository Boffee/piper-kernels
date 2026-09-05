"""Tests for bounded sparse-attention-to-ConvRot output fusion."""

from __future__ import annotations

import pytest
import torch

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
    _sparse_piper_attention_with_coarse_residual_from_quantized_op,
)
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


def _projection(
    *,
    bias: bool,
    bias_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        torch.randn(_OUTPUT_FEATURES, device="cuda", dtype=bias_dtype) if bias else None
    )
    return weight, scale, projected_bias


def _arguments(
    *,
    batch: int,
    sequence_length: int,
    bias: bool,
    bias_dtype: torch.dtype = torch.bfloat16,
) -> tuple[tuple[object, ...], torch.Tensor]:
    operands = _operands(batch=batch, sequence_length=sequence_length)
    prepared_query, prepared_key, prepared_value = _prepare(operands)
    attention = SparsePiperAttention((0.5, 1.0))
    sparse_key_blocks = max(1, sequence_length // 64)
    weight, scale, projected_bias = _projection(bias=bias, bias_dtype=bias_dtype)
    arguments = (
        *prepared_query,
        *prepared_key,
        *prepared_value,
        list(attention._head_keep_ratio_units),
        sparse_key_blocks,
        sequence_length,
        attention._routing_mode,
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


def _padded_arguments(
    *,
    coarse: bool,
    sparse_query_blocks: int | None,
) -> tuple[tuple[object, ...], torch.Tensor]:
    sequence_length = 192
    arguments, _unbounded_expected = _arguments(
        batch=1,
        sequence_length=sequence_length,
        bias=True,
    )
    attention_arguments = arguments[:14]
    projection_arguments = arguments[14:]
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    block_mean = (
        torch.randn(
            (1, _HEADS, sequence_length // 64, _HEAD_DIM),
            device="cuda",
            dtype=torch.float32,
        )
        if coarse
        else None
    )
    coarse_gate = (
        torch.randn(
            (1, sequence_length, _HEADS, _HEAD_DIM),
            device="cuda",
            dtype=torch.bfloat16,
        )
        if coarse
        else None
    )
    coarse_scale = 0.125 if coarse else None
    coarse_key_blocks = sequence_length // 64 if coarse else None
    if coarse:
        assert block_mean is not None
        assert coarse_gate is not None
        materialized_attention = _sparse_piper_attention_with_coarse_residual_from_quantized_op(
            *attention_arguments[:10],
            block_mean,
            coarse_gate,
            *attention_arguments[10:],
            coarse_scale,
            block_lengths,
            coarse_key_blocks,
            sparse_query_blocks,
        )
    else:
        materialized_attention = _sparse_piper_attention_from_quantized_op(
            *attention_arguments,
            block_lengths,
            sparse_query_blocks,
        )
    weight, scale, bias, group_size = projection_arguments
    expected = convrot_backend.run_linear(
        materialized_attention.flatten(2),
        weight,
        scale,
        bias,
        group_size,
    )
    return (
        *arguments,
        64,
        block_lengths,
        block_mean,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
    ), expected


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
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32])
def test_attention_output_supports_mixed_precision_bias(
    bias_dtype: torch.dtype,
) -> None:
    arguments, expected = _arguments(
        batch=1,
        sequence_length=65,
        bias=True,
        bias_dtype=bias_dtype,
    )

    with torch.no_grad():
        actual = output_fusion._attention_output_op(*arguments, 64)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_projected_query_attention_output_matches_multiple_materialized_q_windows() -> None:
    sequence_length = 193
    operands = _operands(batch=1, sequence_length=sequence_length)
    prepared_query, prepared_key, prepared_value = _prepare(operands)
    attention = SparsePiperAttention((0.5, 1.0))
    sparse_key_blocks = sequence_length // 64
    weight, scale, bias = _projection(bias=True)
    attention_tail = (
        *prepared_key,
        *prepared_value,
        list(attention._head_keep_ratio_units),
        sparse_key_blocks,
        sequence_length,
        attention._routing_mode,
        weight,
        scale,
        bias,
        _HEADS * _HEAD_DIM,
        64,
    )

    with torch.no_grad():
        expected = output_fusion._attention_output_op(
            *prepared_query,
            *attention_tail,
        )
        actual = output_fusion._projected_query_attention_output_op(
            operands.input_qdata,
            operands.input_scale,
            operands.query_weight,
            operands.query_weight_scale,
            operands.query_norm,
            operands.cos,
            operands.sin,
            1e-5,
            _HEAD_DIM**-0.5,
            *attention_tail,
        )

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
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("sparse_query_blocks", [None, 2])
def test_attention_output_supports_bounded_attention_features(
    coarse: bool,
    sparse_query_blocks: int | None,
) -> None:
    arguments, expected = _padded_arguments(
        coarse=coarse,
        sparse_query_blocks=sparse_query_blocks,
    )

    with torch.no_grad():
        actual = output_fusion._attention_output_op(*arguments)

    assert actual.shape == expected.shape == (1, 192, _OUTPUT_FEATURES)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("sequence_length", "nondefault_stream"),
    [(128, False), (193, False), (512, True)],
)
def test_attention_output_projects_a_bounded_coarse_gate(
    sequence_length: int,
    nondefault_stream: bool,
) -> None:
    arguments, _fine_expected = _arguments(
        batch=1,
        sequence_length=sequence_length,
        bias=False,
    )
    attention_arguments = arguments[:14]
    output_arguments = arguments[14:]
    gate_features = _HEADS * _HEAD_DIM
    gate_input_qdata = torch.randint(
        -127,
        128,
        (1, sequence_length, gate_features),
        device="cuda",
        dtype=torch.int8,
    )
    gate_input_scale = torch.rand(
        (1, sequence_length),
        device="cuda",
        dtype=torch.float32,
    ).mul_(0.01)
    gate_weight, gate_scale, gate_bias = _projection(bias=True, bias_dtype=torch.float32)
    gate_weight = gate_weight[:gate_features, :gate_features].contiguous()
    gate_scale = gate_scale[:gate_features].contiguous()
    assert gate_bias is not None
    gate_bias = gate_bias[:gate_features].contiguous()
    coarse_gate = convrot_backend._execute_prepared_linear(
        gate_input_qdata,
        gate_input_scale,
        gate_weight,
        gate_scale,
        gate_bias,
        torch.bfloat16,
        convrot_backend.default_execution_plan(gate_weight),
    ).unflatten(-1, (_HEADS, _HEAD_DIM))
    query_blocks = (sequence_length + 63) // 64
    block_mean = torch.randn(
        (1, _HEADS, query_blocks, _HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
    )
    coarse_scale = _HEAD_DIM**-0.5
    coarse_key_blocks = query_blocks
    materialized_attention = _sparse_piper_attention_with_coarse_residual_from_quantized_op(
        *attention_arguments[:10],
        block_mean,
        coarse_gate,
        *attention_arguments[10:],
        coarse_scale,
        None,
        coarse_key_blocks,
        None,
    )
    output_weight, output_scale, output_bias, output_group_size = output_arguments
    expected = convrot_backend.run_linear(
        materialized_attention.flatten(2),
        output_weight,
        output_scale,
        output_bias,
        output_group_size,
    )

    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        actual = output_fusion._attention_output_op(
            *attention_arguments,
            *output_arguments,
            64,
            None,
            block_mean,
            None,
            coarse_scale,
            coarse_key_blocks,
            None,
            gate_input_qdata,
            gate_input_scale,
            gate_weight,
            gate_scale,
            gate_bias,
        )
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
        0,
        torch.empty((_OUTPUT_FEATURES, 3 * 128), device="meta", dtype=torch.int8),
        torch.empty((_OUTPUT_FEATURES, 1), device="meta", dtype=torch.float32),
        None,
        128,
        64,
    )

    assert output.shape == (2, 191, _OUTPUT_FEATURES)
    assert output.dtype is torch.bfloat16


def test_attention_output_fake_kernel_uses_padded_storage_length() -> None:
    arguments = (
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
        132,
        0,
        torch.empty((_OUTPUT_FEATURES, 3 * 128), device="meta", dtype=torch.int8),
        torch.empty((_OUTPUT_FEATURES, 1), device="meta", dtype=torch.float32),
        None,
        128,
        64,
        torch.empty(3, device="meta", dtype=torch.int32),
    )

    output = output_fusion._attention_output_op(*arguments)

    assert output.shape == (2, 192, _OUTPUT_FEATURES)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("query_chunk_rows", [0, 63, 65])
def test_attention_output_rejects_invalid_query_chunk_rows(query_chunk_rows: int) -> None:
    arguments, _expected = _arguments(batch=1, sequence_length=128, bias=False)

    with pytest.raises(ValueError, match="positive multiple of 64"):
        output_fusion._attention_output_op(*arguments, query_chunk_rows)
