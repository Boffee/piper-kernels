"""Tests for bounded sparse-attention-to-ConvRot-NVFP4 output fusion."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
    _sparse_piper_attention_with_coarse_residual_from_quantized_op,
)
from piper_kernels.fusions.convrot_nvfp4_sparse_piper import output
from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.nvfp4 import _ops as convrot_nvfp4_ops

from ..nvfp4_sparse_piper.test_output import _arguments as standard_arguments

_OUTPUT_FEATURES = 320
_GROUP_SIZE = 16


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _arguments(
    sequence_length: int,
    group_size: int = _GROUP_SIZE,
) -> tuple[tuple[object, ...], torch.Tensor]:
    standard, _unused = standard_arguments(sequence_length=sequence_length, bias=True)
    attention_arguments = standard[:14]
    activation_scale = standard[-2]
    bias = standard[-1]
    assert isinstance(activation_scale, torch.Tensor)
    assert isinstance(bias, torch.Tensor)
    dense = torch.randn(
        (_OUTPUT_FEATURES, 256),
        device="cuda",
        dtype=torch.bfloat16,
    )
    rotated = rotate_groups(dense, group_size)
    weight = TorchAONVFP4Tensor.to_nvfp4(
        rotated,
        per_tensor_scale=per_tensor_amax_to_scale(rotated.abs().amax()),
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    materialized_attention = _sparse_piper_attention_from_quantized_op(*attention_arguments)
    input_qdata, input_scale, input_per_tensor_scale = convrot_nvfp4_ops.prepare_input(
        materialized_attention.flatten(2),
        activation_scale,
        False,
        group_size,
    )
    assert weight.per_tensor_scale is not None
    expected = F.scaled_mm(
        input_qdata.view(torch.float4_e2m1fn_x2),
        weight.qdata.t().view(torch.float4_e2m1fn_x2),
        [input_scale.view(torch.float8_e4m3fn), input_per_tensor_scale],
        [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        [weight.scale.view(torch.float8_e4m3fn), weight.per_tensor_scale],
        [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        bias=bias,
        output_dtype=torch.bfloat16,
    ).reshape(1, sequence_length, _OUTPUT_FEATURES)
    return (
        *attention_arguments,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        activation_scale,
        bias,
        group_size,
    ), expected


def _padded_arguments(
    *,
    coarse: bool,
    sparse_query_blocks: int | None,
) -> tuple[tuple[object, ...], torch.Tensor]:
    sequence_length = 192
    arguments, _unbounded_expected = _arguments(sequence_length)
    attention_arguments = arguments[:14]
    projection_arguments = arguments[14:]
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    block_mean = (
        torch.randn(
            (1, 2, sequence_length // 64, 128),
            device="cuda",
            dtype=torch.float32,
        )
        if coarse
        else None
    )
    compression_gate = (
        torch.randn(
            (1, sequence_length, 2, 128),
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
        assert compression_gate is not None
        materialized_attention = _sparse_piper_attention_with_coarse_residual_from_quantized_op(
            *attention_arguments[:10],
            block_mean,
            compression_gate,
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
    (
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        activation_scale,
        bias,
        group_size,
    ) = projection_arguments
    input_qdata, input_scale, input_per_tensor_scale = convrot_nvfp4_ops.prepare_input(
        materialized_attention.flatten(2),
        activation_scale,
        False,
        group_size,
    )
    expected = F.scaled_mm(
        input_qdata.view(torch.float4_e2m1fn_x2),
        weight_qdata.t().view(torch.float4_e2m1fn_x2),
        [input_scale.view(torch.float8_e4m3fn), input_per_tensor_scale],
        [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        [weight_scale.view(torch.float8_e4m3fn), weight_per_tensor_scale],
        [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        bias=bias,
        output_dtype=torch.bfloat16,
    ).reshape(1, sequence_length, _OUTPUT_FEATURES)
    return (
        *arguments,
        128,
        block_lengths,
        block_mean,
        compression_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
    ), expected


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("sequence_length", "chunk_rows", "group_size"),
    [(128, 128, 16), (193, 128, 64), (193, 128, 256)],
)
def test_attention_output_matches_materialized_convrot_linear(
    sequence_length: int,
    chunk_rows: int,
    group_size: int,
) -> None:
    arguments, expected = _arguments(sequence_length, group_size)

    with torch.no_grad():
        actual = output._attention_output_op(*arguments, chunk_rows)

    assert torch.equal(actual, expected)


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
        actual = output._attention_output_op(*arguments)

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
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_attention_output_custom_op_passes_opcheck() -> None:
    arguments, _expected = _arguments(128)

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
        _GROUP_SIZE,
        128,
    )

    assert actual.shape == (1, 191, _OUTPUT_FEATURES)
    assert actual.dtype is torch.bfloat16
