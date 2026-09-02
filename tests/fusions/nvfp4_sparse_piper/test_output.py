"""Tests for bounded sparse-attention-to-NVFP4 output fusion."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    per_tensor_amax_to_scale,
)

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
)
from piper_kernels.fusions.nvfp4_sparse_piper import key, output, query, value
from piper_kernels.linear.nvfp4 import _ops
from piper_kernels.linear.nvfp4.triton import linear_mean

from ._helpers import exact_sm120_available, make_operands

_HEADS = 2
_HEAD_DIM = 128
_OUTPUT_FEATURES = 320


def _affine_nvfp4_linear(
    input: torch.Tensor,  # noqa: A002
    weight: TorchAONVFP4Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the best available NVFP4 affine reference for one weight."""
    if weight.per_tensor_scale is None:
        return _ops.linear(
            input,
            weight.qdata,
            weight.scale,
            None,
            activation_scale,
            bias,
            False,
        )
    input_qdata, input_scale, input_per_tensor_scale = _ops.prepare_input(
        input,
        activation_scale,
        False,
    )
    assert weight.per_tensor_scale is not None
    scaling_type = F.ScalingType
    swizzle_type = F.SwizzleType
    result = F.scaled_mm(
        input_qdata.view(torch.float4_e2m1fn_x2),
        weight.qdata.t().view(torch.float4_e2m1fn_x2),
        [input_scale.view(torch.float8_e4m3fn), input_per_tensor_scale],
        [scaling_type.BlockWise1x16, scaling_type.TensorWise],
        [weight.scale.view(torch.float8_e4m3fn), weight.per_tensor_scale],
        [scaling_type.BlockWise1x16, scaling_type.TensorWise],
        [swizzle_type.SWIZZLE_32_4_4, swizzle_type.NO_SWIZZLE],
        [swizzle_type.SWIZZLE_32_4_4, swizzle_type.NO_SWIZZLE],
        bias=bias,
        output_dtype=input.dtype,
    )
    return result.reshape(*input.shape[:-1], weight.shape[0])


def _arguments(
    *,
    sequence_length: int,
    bias: bool,
    weight_global_scale: bool = True,
) -> tuple[tuple[object, ...], torch.Tensor]:
    operands = make_operands(sequence_length=sequence_length)
    q_projection = operands.projection(0)
    k_projection = operands.projection(1)
    v_projection = operands.projection(2)
    prepared_query = query.project_query(
        *q_projection.as_tuple(),
        None,
        operands.query_norm,
        operands.cos,
        operands.sin,
        1e-5,
        _HEAD_DIM**-0.5,
        128,
    )
    prepared_key = key.project_key(
        *k_projection.as_tuple(),
        None,
        operands.key_norm,
        operands.cos,
        operands.sin,
        1e-5,
        128,
    )
    value_mean = linear_mean(
        *v_projection.as_tuple(),
        None,
        1,
        sequence_length,
    ).view(1, _HEADS, _HEAD_DIM)
    prepared_value = value.project_value(
        *v_projection.as_tuple(),
        None,
        value_mean,
        128,
    )
    attention = SparsePiperAttention((0.5, 1.0))
    sparse_key_blocks = max(1, sequence_length // 64)

    dense_weight = torch.randn(
        (_OUTPUT_FEATURES, _HEADS * _HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output_weight = TorchAONVFP4Tensor.to_nvfp4(
        dense_weight,
        per_tensor_scale=(
            per_tensor_amax_to_scale(dense_weight.abs().amax()) if weight_global_scale else None
        ),
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    activation_scale = per_tensor_amax_to_scale(
        torch.tensor(3.0, device="cuda", dtype=torch.float32)
    )
    projected_bias = (
        torch.randn(_OUTPUT_FEATURES, device="cuda", dtype=torch.bfloat16) if bias else None
    )
    arguments = (
        *prepared_query,
        *prepared_key,
        *prepared_value,
        value_mean,
        list(attention._head_keep_ratio_units),
        sparse_key_blocks,
        sequence_length,
        attention._routing_mode,
        output_weight.qdata,
        output_weight.scale,
        output_weight.per_tensor_scale,
        activation_scale,
        projected_bias,
    )
    with torch.no_grad():
        materialized_attention = _sparse_piper_attention_from_quantized_op(
            *prepared_query,
            *prepared_key,
            *prepared_value,
            value_mean,
            list(attention._head_keep_ratio_units),
            sparse_key_blocks,
            sequence_length,
            attention._routing_mode,
        )
        expected = _affine_nvfp4_linear(
            materialized_attention.flatten(2),
            output_weight,
            activation_scale,
            projected_bias,
        )
    return arguments, expected


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("sequence_length", "query_chunk_rows", "bias"),
    [(128, 128, False), (193, 128, True), (193, 4_096, True), (257, 256, True)],
)
def test_attention_output_matches_accumulator_affine_boundary(
    sequence_length: int,
    query_chunk_rows: int,
    bias: bool,
) -> None:
    arguments, expected = _arguments(sequence_length=sequence_length, bias=bias)

    with torch.no_grad():
        actual = output._attention_output_op(*arguments, query_chunk_rows)

    assert actual.shape == (1, sequence_length, _OUTPUT_FEATURES)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_attention_output_obeys_a_nondefault_current_stream() -> None:
    arguments, expected = _arguments(sequence_length=193, bias=False)
    stream = torch.cuda.Stream()

    with torch.no_grad(), torch.cuda.stream(stream):
        actual = output._attention_output_op(*arguments, 128)
    stream.synchronize()

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_attention_output_supports_blockwise_only_weight_scale() -> None:
    arguments, expected = _arguments(
        sequence_length=128,
        bias=True,
        weight_global_scale=False,
    )

    with torch.no_grad():
        actual = output._attention_output_op(*arguments, 128)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_attention_output_custom_op_passes_opcheck() -> None:
    arguments, _expected = _arguments(sequence_length=128, bias=True)

    with torch.no_grad():
        result = torch.library.opcheck(
            output._attention_output_op,
            (*arguments, 128),
            test_utils=(
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ),
        )

    assert set(result.values()) == {"SUCCESS"}


def test_attention_output_fake_kernel_propagates_shape() -> None:
    actual = output._attention_output_op(
        torch.empty((1, 2, 192, 128), device="meta", dtype=torch.int8),
        torch.empty((1, 2, 6), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 192, 128), device="meta", dtype=torch.int8),
        torch.empty((1, 2, 3), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 128, 192), device="meta", dtype=torch.int8),
        torch.empty((1, 2, 3, 1), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 128), device="meta", dtype=torch.float32),
        [500_000, 1_000_000],
        2,
        191,
        0,
        torch.empty((_OUTPUT_FEATURES, 128), device="meta", dtype=torch.uint8),
        torch.empty((96, 64), device="meta", dtype=torch.float8_e4m3fn),
        torch.empty((), device="meta", dtype=torch.float32),
        torch.empty((), device="meta", dtype=torch.float32),
        None,
        128,
    )

    assert actual.shape == (1, 191, _OUTPUT_FEATURES)
    assert actual.dtype is torch.bfloat16


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("query_chunk_rows", [0, 64, 192])
def test_attention_output_rejects_invalid_query_chunk_rows(query_chunk_rows: int) -> None:
    arguments, _expected = _arguments(sequence_length=128, bias=False)

    with pytest.raises(ValueError, match="positive multiple of 128"):
        output._attention_output_op(*arguments, query_chunk_rows)
