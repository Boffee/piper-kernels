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
    _sparse_piper_attention_with_coarse_residual_from_quantized_op,
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
            False,
        )
    input_qdata, input_scale, input_per_tensor_scale = _ops.prepare_input(
        input,
        activation_scale,
        False,
        None,
        False,
    )
    assert weight.per_tensor_scale is not None
    scaling_type = F.ScalingType
    swizzle_type = F.SwizzleType
    fused_bias = bias if bias is None or bias.dtype is input.dtype else None
    result = F.scaled_mm(
        input_qdata.view(torch.float4_e2m1fn_x2),
        weight.qdata.t().view(torch.float4_e2m1fn_x2),
        [input_scale.view(torch.float8_e4m3fn), input_per_tensor_scale],
        [scaling_type.BlockWise1x16, scaling_type.TensorWise],
        [weight.scale.view(torch.float8_e4m3fn), weight.per_tensor_scale],
        [scaling_type.BlockWise1x16, scaling_type.TensorWise],
        [swizzle_type.SWIZZLE_32_4_4, swizzle_type.NO_SWIZZLE],
        [swizzle_type.SWIZZLE_32_4_4, swizzle_type.NO_SWIZZLE],
        bias=fused_bias,
        output_dtype=input.dtype,
    )
    if bias is not None and fused_bias is None:
        result = (result.float() + bias.float()).to(result.dtype)
    return result.reshape(*input.shape[:-1], weight.shape[0])


def _arguments(
    *,
    sequence_length: int,
    bias: bool,
    bias_dtype: torch.dtype = torch.bfloat16,
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
        torch.randn(_OUTPUT_FEATURES, device="cuda", dtype=bias_dtype) if bias else None
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


def _padded_arguments(
    *,
    coarse: bool,
    sparse_query_blocks: int | None,
) -> tuple[tuple[object, ...], torch.Tensor]:
    sequence_length = 192
    arguments, _unbounded_expected = _arguments(
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
    weight_qdata, weight_scale, weight_per_tensor_scale, activation_scale, bias = (
        projection_arguments
    )
    output_weight = TorchAONVFP4Tensor(
        weight_qdata,
        weight_scale,
        16,
        torch.bfloat16,
        per_tensor_scale=weight_per_tensor_scale,
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    expected = _affine_nvfp4_linear(
        materialized_attention.flatten(2),
        output_weight,
        activation_scale,
        bias,
    )
    return (
        *arguments,
        128,
        block_lengths,
        block_mean,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
    ), expected


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_projected_query_attention_output_matches_multiple_materialized_q_windows() -> None:
    sequence_length = 193
    arguments, expected = _arguments(sequence_length=sequence_length, bias=True)
    operands = make_operands(sequence_length=sequence_length)
    query_projection = operands.projection(0)

    with torch.no_grad():
        actual = output._projected_query_attention_output_op(
            *query_projection.as_tuple(),
            None,
            operands.query_norm,
            operands.cos,
            operands.sin,
            1e-5,
            _HEAD_DIM**-0.5,
            *arguments[3:],
            128,
        )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _projected_gate_arguments(
    *,
    sequence_length: int,
    weight_global_scale: bool,
) -> tuple[tuple[object, ...], torch.Tensor]:
    arguments, _sparse_only = _arguments(sequence_length=sequence_length, bias=True)
    hidden = torch.randn(
        (1, sequence_length, _HEADS * _HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    activation_scale = per_tensor_amax_to_scale(hidden.abs().amax())
    gate_dense = torch.randn(
        (_HEADS * _HEAD_DIM, hidden.shape[-1]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate_weight = TorchAONVFP4Tensor.to_nvfp4(
        gate_dense,
        per_tensor_scale=(
            per_tensor_amax_to_scale(gate_dense.abs().amax()) if weight_global_scale else None
        ),
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    gate_bias = torch.randn(
        _HEADS * _HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
    )
    gate_input = _ops.prepare_input(hidden, activation_scale, False, None, False)
    with torch.no_grad():
        coarse_gate = _ops.linear_prepared(
            *gate_input,
            gate_weight.qdata,
            gate_weight.scale,
            gate_weight.per_tensor_scale,
            gate_bias,
            torch.bfloat16,
        ).view(1, sequence_length, _HEADS, _HEAD_DIM)
    coarse_key_blocks = (sequence_length + 63) // 64
    block_mean = torch.randn(
        (1, _HEADS, coarse_key_blocks, _HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
    )
    with torch.no_grad():
        expected = output._attention_output_op(
            *arguments,
            128,
            None,
            block_mean,
            coarse_gate,
            0.125,
            coarse_key_blocks,
            None,
        )
    return (
        *arguments,
        128,
        None,
        block_mean,
        None,
        0.125,
        coarse_key_blocks,
        None,
        *gate_input,
        gate_weight.qdata,
        gate_weight.scale,
        gate_weight.per_tensor_scale,
        gate_bias,
    ), expected


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
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32])
def test_attention_output_supports_mixed_precision_bias(
    bias_dtype: torch.dtype,
) -> None:
    arguments, expected = _arguments(
        sequence_length=193,
        bias=True,
        bias_dtype=bias_dtype,
    )

    with torch.no_grad():
        actual = output._attention_output_op(*arguments, 128)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_high_first_attention_output_matches_low_first() -> None:
    arguments, expected = _arguments(sequence_length=193, bias=True)
    high_arguments = list(arguments)
    qdata = high_arguments[14]
    assert isinstance(qdata, torch.Tensor)
    high_arguments[14] = ((qdata & 0x0F) << 4) | (qdata >> 4)

    with torch.no_grad():
        actual = output._attention_output_op(
            *high_arguments,
            128,
            high_first=True,
        )

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
        actual = output._attention_output_op(*arguments)

    assert actual.shape == expected.shape == (1, 192, _OUTPUT_FEATURES)
    block_lengths = arguments[-6]
    assert isinstance(block_lengths, torch.Tensor)
    valid_rows = torch.arange(64, device="cuda")[None] < block_lengths[:, None]
    torch.testing.assert_close(
        actual[:, valid_rows.flatten()],
        expected[:, valid_rows.flatten()],
        atol=0,
        rtol=0,
    )


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
@pytest.mark.parametrize("weight_global_scale", [False, True])
def test_attention_output_projects_a_bounded_coarse_gate(
    weight_global_scale: bool,
) -> None:
    arguments, expected = _projected_gate_arguments(
        sequence_length=1_024,
        weight_global_scale=weight_global_scale,
    )

    with torch.no_grad():
        actual = output._attention_output_op(*arguments)

    torch.testing.assert_close(actual, expected, atol=2**-7, rtol=2**-7)


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
