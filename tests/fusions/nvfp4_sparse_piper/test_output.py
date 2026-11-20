"""Tests for bounded sparse-attention-to-NVFP4 and ConvRot-NVFP4 output fusion."""

from __future__ import annotations

import pytest
import torch
from _accelerators import exact_sm120_available
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    per_tensor_amax_to_scale,
)

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
    _sparse_piper_attention_with_coarse_residual_from_quantized_op,
)
from piper_kernels.fusions.nvfp4_sparse_piper import key, output, query, value
from piper_kernels.linear.convrot.nvfp4 import _ops as convrot_nvfp4_ops
from piper_kernels.linear.nvfp4 import _ops, _projection, reference
from piper_kernels.weights.convrot._rotation import rotate_groups

from .._accuracy import assert_fusion_output_close
from ._helpers import make_operands

_HEADS = 2
_HEAD_DIM = 128
_OUTPUT_FEATURES = 320
_GROUP_SIZES = [pytest.param(None, id="nvfp4"), pytest.param(64, id="convrot64")]


def _affine_nvfp4_linear(
    input: torch.Tensor,  # noqa: A002
    weight: TorchAONVFP4Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    group_size: int | None = None,
) -> torch.Tensor:
    """Compare affine arithmetic independently on the integration test's prepared input."""
    if group_size is None:
        input_qdata, input_scale, input_per_tensor_scale = _ops.prepare_input(
            input,
            activation_scale,
            False,
            None,
            False,
        )
    else:
        input_qdata, input_scale, input_per_tensor_scale = convrot_nvfp4_ops.prepare_input(
            input,
            activation_scale,
            False,
            group_size,
            None,
            False,
        )
    result = reference.linear_prepared(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        bias,
        input.dtype,
    )
    return result.reshape(*input.shape[:-1], weight.shape[0])


def _arguments(
    *,
    sequence_length: int,
    bias: bool,
    bias_dtype: torch.dtype = torch.bfloat16,
    weight_global_scale: bool = True,
    group_size: int | None = None,
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
    prepared_value = value.project_value(
        *v_projection.as_tuple(),
        None,
        128,
    )
    attention = SparsePiperAttention((0.5, 1.0))
    sparse_key_blocks = max(1, sequence_length // 64)

    dense_weight = torch.randn(
        (_OUTPUT_FEATURES, _HEADS * _HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    quantized_source = (
        dense_weight if group_size is None else rotate_groups(dense_weight, group_size)
    )
    output_weight = TorchAONVFP4Tensor.to_nvfp4(
        quantized_source,
        per_tensor_scale=(
            per_tensor_amax_to_scale(quantized_source.abs().amax()) if weight_global_scale else None
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
        attention._ratio_profile,
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
            attention._ratio_profile,
            sparse_key_blocks,
            sequence_length,
            attention._routing_mode,
        )
        expected = _affine_nvfp4_linear(
            materialized_attention.flatten(2),
            output_weight,
            activation_scale,
            projected_bias,
            group_size=group_size,
        )
    return arguments, expected


def _padded_arguments(
    *,
    coarse: bool,
    sparse_query_blocks: int | None,
    group_size: int | None = None,
) -> tuple[tuple[object, ...], torch.Tensor]:
    sequence_length = 192
    arguments, _unbounded_expected = _arguments(
        sequence_length=sequence_length,
        bias=True,
        group_size=group_size,
    )
    attention_arguments = arguments[:16]
    projection_arguments = arguments[16:]
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
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
        assert coarse_gate is not None
        materialized_attention = _sparse_piper_attention_with_coarse_residual_from_quantized_op(
            *attention_arguments[:12],
            coarse_gate,
            *attention_arguments[12:],
            coarse_scale,
            coarse_key_blocks,
            block_lengths,
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
        group_size=group_size,
    )
    return (
        *arguments,
        128,
        block_lengths,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
    ), expected


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_projected_query_attention_output_matches_multiple_materialized_q_windows(
    group_size: int | None,
) -> None:
    sequence_length = 193
    arguments, expected = _arguments(
        sequence_length=sequence_length,
        bias=True,
        group_size=group_size,
    )
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
            group_size=group_size,
        )

    assert_fusion_output_close(actual, expected)


def _projected_gate_arguments(
    *,
    sequence_length: int,
    weight_global_scale: bool,
    group_size: int | None = None,
) -> tuple[tuple[object, ...], torch.Tensor]:
    arguments, _sparse_only = _arguments(
        sequence_length=sequence_length,
        bias=True,
        group_size=group_size,
    )
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
    gate_source = gate_dense if group_size is None else rotate_groups(gate_dense, group_size)
    gate_weight = TorchAONVFP4Tensor.to_nvfp4(
        gate_source,
        per_tensor_scale=(
            per_tensor_amax_to_scale(gate_source.abs().amax()) if weight_global_scale else None
        ),
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    gate_bias = torch.randn(
        _HEADS * _HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
    )
    if group_size is None:
        gate_input = _ops.prepare_input(hidden, activation_scale, False, None, False)
    else:
        gate_input = convrot_nvfp4_ops.prepare_input(
            hidden,
            activation_scale,
            False,
            group_size,
            None,
            False,
        )
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
    with torch.no_grad():
        expected = output._attention_output_op(
            *arguments,
            128,
            None,
            coarse_gate,
            0.125,
            coarse_key_blocks,
            None,
            group_size=group_size,
        )
    return (
        *arguments,
        128,
        None,
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
    ("sequence_length", "query_chunk_rows", "bias", "group_size"),
    [
        (128, 128, False, None),
        (193, 128, True, None),
        (193, 4_096, True, None),
        (257, 256, True, None),
        (128, 128, True, 16),
        (193, 128, True, 64),
        (193, 128, True, 256),
    ],
)
def test_attention_output_matches_accumulator_affine_boundary(
    sequence_length: int,
    query_chunk_rows: int,
    bias: bool,
    group_size: int | None,
) -> None:
    arguments, expected = _arguments(
        sequence_length=sequence_length,
        bias=bias,
        group_size=group_size,
    )

    with torch.no_grad():
        actual = output._attention_output_op(
            *arguments,
            query_chunk_rows,
            group_size=group_size,
        )

    assert actual.shape == (1, sequence_length, _OUTPUT_FEATURES)
    assert actual.is_contiguous()
    assert_fusion_output_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32])
def test_attention_output_supports_mixed_precision_bias(
    group_size: int | None,
    bias_dtype: torch.dtype,
) -> None:
    arguments, expected = _arguments(
        sequence_length=193,
        bias=True,
        bias_dtype=bias_dtype,
        group_size=group_size,
    )

    with torch.no_grad():
        actual = output._attention_output_op(*arguments, 128, group_size=group_size)

    assert actual.dtype is torch.bfloat16
    assert_fusion_output_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_high_first_attention_output_matches_low_first(group_size: int | None) -> None:
    arguments, _reference = _arguments(sequence_length=193, bias=True, group_size=group_size)
    high_arguments = list(arguments)
    qdata = high_arguments[16]
    assert isinstance(qdata, torch.Tensor)
    high_arguments[16] = ((qdata & 0x0F) << 4) | (qdata >> 4)

    with torch.no_grad():
        expected = output._attention_output_op(*arguments, 128, group_size=group_size)
        actual = output._attention_output_op(
            *high_arguments,
            128,
            high_first=True,
            group_size=group_size,
        )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_attention_output_obeys_a_nondefault_current_stream() -> None:
    arguments, _reference = _arguments(sequence_length=193, bias=False)
    with torch.no_grad():
        expected = output._attention_output_op(*arguments, 128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.no_grad(), torch.cuda.stream(stream):
        actual = output._attention_output_op(*arguments, 128)
    stream.synchronize()

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("sparse_query_blocks", [None, 2])
def test_attention_output_supports_bounded_attention_features(
    group_size: int | None,
    coarse: bool,
    sparse_query_blocks: int | None,
) -> None:
    arguments, expected = _padded_arguments(
        coarse=coarse,
        sparse_query_blocks=sparse_query_blocks,
        group_size=group_size,
    )

    with torch.no_grad():
        actual = output._attention_output_op(*arguments, group_size=group_size)

    assert actual.shape == expected.shape == (1, 192, _OUTPUT_FEATURES)
    block_lengths = arguments[-5]
    assert isinstance(block_lengths, torch.Tensor)
    valid_rows = torch.arange(64, device="cuda")[None] < block_lengths[:, None]
    assert_fusion_output_close(actual[:, valid_rows.flatten()], expected[:, valid_rows.flatten()])


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_attention_output_supports_blockwise_only_weight_scale(group_size: int | None) -> None:
    arguments, expected = _arguments(
        sequence_length=128,
        bias=True,
        weight_global_scale=False,
        group_size=group_size,
    )

    with torch.no_grad():
        actual = output._attention_output_op(*arguments, 128, group_size=group_size)

    assert_fusion_output_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("weight_global_scale", [False, True])
@pytest.mark.parametrize("sequence_length", [384, 1_024])
def test_attention_output_projects_a_bounded_coarse_gate(
    group_size: int | None,
    weight_global_scale: bool,
    sequence_length: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, expected = _projected_gate_arguments(
        sequence_length=sequence_length,
        weight_global_scale=weight_global_scale,
        group_size=group_size,
    )
    streams = set()
    matmul = _projection._matmul_affine_out

    def record_stream(*args, **kwargs):
        streams.add(torch.cuda.current_stream().cuda_stream)
        return matmul(*args, **kwargs)

    monkeypatch.setattr(_projection, "_matmul_affine_out", record_stream)
    # Gate and output GEMMs must retain their own scales on separate streams,
    # including under contention that exposed the native shared-alpha race.
    load_input = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    load_output = torch.empty_like(load_input)
    load_stream = torch.cuda.Stream()
    load_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(load_stream):
        for _ in range(8):
            torch.mm(load_input, load_input, out=load_output)

    with torch.no_grad():
        actual = output._attention_output_op(*arguments, group_size=group_size)
    torch.cuda.current_stream().wait_stream(load_stream)

    assert len(streams) == 2
    assert_fusion_output_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_attention_output_custom_op_passes_opcheck(group_size: int | None) -> None:
    arguments, _expected = _arguments(sequence_length=128, bias=True, group_size=group_size)

    with torch.no_grad():
        result = torch.library.opcheck(
            output._attention_output_op,
            (*arguments, 128),
            {"group_size": group_size},
            test_utils=(
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ),
        )

    assert set(result.values()) == {"SUCCESS"}


def _meta_attention_output_arguments(
    *,
    weight_packed_cols: int = 128,
) -> tuple[object, ...]:
    return (
        torch.empty((1, 2, 192, 128), device="meta", dtype=torch.int8),
        torch.empty((1, 2, 6), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 192, 128), device="meta", dtype=torch.int8),
        torch.empty((1, 2, 3), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 3, 2, 128), device="meta", dtype=torch.int8),
        torch.empty((1, 2, 3, 2), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 3, 128), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 128, 192), device="meta", dtype=torch.int8),
        torch.empty((1, 2, 3, 1), device="meta", dtype=torch.float32),
        torch.empty((1, 2, 3, 128), device="meta", dtype=torch.bfloat16),
        _SparsePiperRatioProfile((500_000, 1_000_000)),
        2,
        191,
        0,
        torch.empty((_OUTPUT_FEATURES, weight_packed_cols), device="meta", dtype=torch.uint8),
        torch.empty((96, 64), device="meta", dtype=torch.float8_e4m3fn),
        torch.empty((), device="meta", dtype=torch.float32),
        torch.empty((), device="meta", dtype=torch.float32),
        None,
        128,
    )


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_attention_output_fake_kernel_propagates_shape(
    group_size: int | None,
    dtype: torch.dtype,
) -> None:
    actual = output._attention_output_op(
        *_meta_attention_output_arguments(),
        output_dtype=dtype,
        group_size=group_size,
    )

    assert actual.shape == (1, 191, _OUTPUT_FEATURES)
    assert actual.dtype is dtype


def test_attention_output_fake_kernel_validates_group_size() -> None:
    with pytest.raises(ValueError, match="ConvRot group size"):
        output._attention_output_op(*_meta_attention_output_arguments(), group_size=32)

    with pytest.raises(ValueError, match="divisible by group size"):
        output._attention_output_op(
            *_meta_attention_output_arguments(weight_packed_cols=64),
            group_size=256,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("query_chunk_rows", [0, 64, 192])
def test_attention_output_rejects_invalid_query_chunk_rows(query_chunk_rows: int) -> None:
    arguments, _expected = _arguments(sequence_length=128, bias=False)

    with pytest.raises(ValueError, match="positive multiple of 128"):
        output._attention_output_op(*arguments, query_chunk_rows)
