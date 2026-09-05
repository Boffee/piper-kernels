"""Tests for direct NVFP4 preparation and projection epilogues."""

import pytest
import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.linear.nvfp4 import _layout
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import triton as nvfp4_triton


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(("rows", "features"), [(127, 80), (1_025, 5_376)])
def test_dynamic_scale_matches_portable_reduction(rows: int, features: int) -> None:
    torch.manual_seed(500 + rows)
    input = torch.randn(  # noqa: A001
        rows,
        features,
        device="cuda",
        dtype=torch.bfloat16,
    )

    expected = per_tensor_amax_to_scale(input.abs().amax())
    actual = nvfp4_triton.dynamic_scale(input)

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("rows", [127, 128, 129])
@pytest.mark.parametrize("activation_fn", [None, "swiglu"])
@pytest.mark.parametrize("high_first", [False, True])
def test_static_preparation_matches_compiled_decomposition(
    rows: int,
    activation_fn: str | None,
    high_first: bool,
) -> None:
    torch._dynamo.reset()
    torch.manual_seed(501)
    output_features = 80
    input_features = output_features * (2 if activation_fn == "swiglu" else 1)
    input = torch.randn(  # noqa: A001
        rows,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    per_tensor_scale = torch.tensor(1.0 / 448.0, device="cuda", dtype=torch.float32)

    expected = nvfp4_ops._compiled_prepare_static(
        input,
        per_tensor_scale,
        activation_fn,
        high_first,
    )
    actual = nvfp4_triton.prepare_static(
        input,
        per_tensor_scale,
        swiglu=activation_fn == "swiglu",
        high_first=high_first,
    )

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("activation_fn", [None, "swiglu"])
def test_static_preparation_preserves_noncontiguous_logical_order(
    activation_fn: str | None,
) -> None:
    torch.manual_seed(504)
    output_features = 80
    input_features = output_features * (2 if activation_fn == "swiglu" else 1)
    input = torch.randn(  # noqa: A001
        2,
        3,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(0, 1)
    assert not input.is_contiguous()
    per_tensor_scale = torch.tensor(1.0 / 448.0, device="cuda", dtype=torch.float32)

    expected = nvfp4_triton.prepare_static(
        input.contiguous(),
        per_tensor_scale,
        swiglu=activation_fn == "swiglu",
    )
    actual = nvfp4_triton.prepare_static(
        input,
        per_tensor_scale,
        swiglu=activation_fn == "swiglu",
    )

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_static_preparation_out_preserves_logical_shapes() -> None:
    torch.manual_seed(507)
    rows, input_features = 127, 80
    input = torch.randn(  # noqa: A001
        rows,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    per_tensor_scale = torch.tensor(1.0 / 448.0, device="cuda", dtype=torch.float32)
    qdata_storage = torch.empty(
        (256, input_features // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    scale_storage = torch.empty(
        (64, 32),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )

    expected_qdata, expected_scale, _ = nvfp4_triton.prepare_static(
        input,
        per_tensor_scale,
    )
    actual_qdata, actual_scale = nvfp4_triton.prepare_static_out(
        input,
        per_tensor_scale,
        (qdata_storage, scale_storage),
    )

    assert actual_qdata.shape == expected_qdata.shape
    assert actual_scale.shape == expected_scale.shape
    assert torch.equal(actual_qdata, expected_qdata)
    assert torch.equal(actual_scale, expected_scale)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_plain_preparation_preserves_noncontiguous_logical_order() -> None:
    torch.manual_seed(505)
    input = torch.randn(  # noqa: A001
        2,
        3,
        80,
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(0, 1)
    assert not input.is_contiguous()

    expected = nvfp4_ops._compiled_prepare_dynamic(input.contiguous(), None)
    actual = nvfp4_ops._compiled_prepare_dynamic(input, None)

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_add_bias_out_matches_pytorch(
    bias_dtype: torch.dtype,
    output_dtype: torch.dtype,
    strided: bool,
) -> None:
    torch.manual_seed(503)
    raw = (
        torch.randn(129, 160, device="cuda", dtype=torch.float32)[:, :80]
        if strided
        else torch.randn(129, 80, device="cuda", dtype=torch.float32)
    )
    bias = (
        torch.randn(160, device="cuda", dtype=bias_dtype)[::2]
        if strided
        else torch.randn(80, device="cuda", dtype=bias_dtype)
    )
    expected = raw + bias.float()
    if output_dtype is torch.float32:
        actual = raw
    else:
        actual = (
            torch.empty((129, 160), device="cuda", dtype=output_dtype)[:, :80]
            if strided
            else torch.empty_like(raw, dtype=output_dtype)
        )

    nvfp4_triton.add_bias_out(raw, bias, actual)

    torch.testing.assert_close(actual, expected.to(output_dtype))


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("two_level", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
def test_prepared_linear_bounds_workspace(
    bias_dtype: torch.dtype | None,
    two_level: bool,
    output_dtype: torch.dtype,
) -> None:
    rows, features, outputs = 8192, 256, 8192
    # Each packed 0x22 byte represents two ones, so the raw GEMM result is features.
    input_qdata = torch.full((rows, features // 2), 0x22, device="cuda", dtype=torch.uint8)
    weight_qdata = torch.full((outputs, features // 2), 0x22, device="cuda", dtype=torch.uint8)
    input_scale = torch.ones(_layout.scale_shape(rows, features), device="cuda").to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.ones(_layout.scale_shape(outputs, features), device="cuda").to(
        torch.float8_e4m3fn
    )
    input_global = torch.tensor(1 / 16, device="cuda")
    weight_global = torch.tensor(1 / 8, device="cuda") if two_level else None
    bias = torch.ones(outputs, device="cuda", dtype=bias_dtype) if bias_dtype is not None else None
    operands = (
        input_qdata,
        input_scale,
        input_global,
        weight_qdata,
        weight_scale,
        weight_global,
        bias,
        output_dtype,
    )
    warm = nvfp4_ops._execute_prepared(*operands)
    torch.cuda.synchronize()
    del warm
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    actual = nvfp4_ops._execute_prepared(*operands)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() - baseline
    # Above the 128 MiB output, mixed bias gets at most 32 MiB of FP32 row workspace.
    mixed_bias = bias_dtype is not None and bias_dtype is not output_dtype
    workspace_limit = (33 if mixed_bias else 1) * 1024 * 1024
    assert peak < actual.numel() * actual.element_size() + workspace_limit
    expected_value = features / 16 * (1 / 8 if two_level else 1) + (bias is not None)
    assert (actual == expected_value).all()


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_inplace_bias_addition_crosses_int32_element_boundary() -> None:
    if torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        pytest.skip("requires 5 GiB of free CUDA memory")
    features = 4096
    rows = 2**31 // features + 1
    raw = torch.full((rows, features), 4.0, device="cuda", dtype=torch.bfloat16)
    bias = torch.ones(features, device="cuda", dtype=torch.bfloat16)

    nvfp4_triton.add_bias_out(raw, bias, raw)

    assert (raw[[0, rows - 2, rows - 1]] == 5.0).all()


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_linear_mean_matches_batched_represented_projection() -> None:
    torch.manual_seed(506)
    batch, sequence_length = 2, 257
    input_features, output_features = 80, 128
    source = torch.randn(
        batch * sequence_length,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dense_weight = torch.randn(
        output_features,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    input_scale = per_tensor_amax_to_scale(source.abs().amax())
    weight_scale = per_tensor_amax_to_scale(dense_weight.abs().amax())
    prepared_input = TorchAONVFP4Tensor.to_nvfp4(
        source,
        per_tensor_scale=input_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    prepared_weight = TorchAONVFP4Tensor.to_nvfp4(
        dense_weight,
        per_tensor_scale=weight_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    bias = torch.randn(output_features, device="cuda", dtype=torch.bfloat16)

    expected = (
        prepared_input.dequantize(torch.float32)
        .view(batch, sequence_length, input_features)
        .mean(dim=1)
        @ prepared_weight.dequantize(torch.float32).t()
        + bias.float()
    )
    actual = nvfp4_triton.linear_mean(
        prepared_input.qdata,
        prepared_input.scale,
        input_scale,
        prepared_weight.qdata,
        prepared_weight.scale,
        weight_scale,
        bias,
        batch,
        sequence_length,
    )

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_linear_mean_ignores_valid_front_padding() -> None:
    torch.manual_seed(507)
    batch, sequence_length = 2, 128
    input_features, output_features = 80, 128
    source = torch.randn(
        batch * sequence_length,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dense_weight = torch.randn(
        output_features,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    input_scale = per_tensor_amax_to_scale(source.abs().amax())
    weight_scale = per_tensor_amax_to_scale(dense_weight.abs().amax())
    prepared_input = TorchAONVFP4Tensor.to_nvfp4(
        source,
        per_tensor_scale=input_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    prepared_weight = TorchAONVFP4Tensor.to_nvfp4(
        dense_weight,
        per_tensor_scale=weight_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    block_lengths = torch.tensor([37, 19], device="cuda", dtype=torch.int32)
    valid = torch.arange(64, device="cuda")[None, :] < block_lengths[:, None]
    valid = valid.flatten()
    represented = prepared_input.dequantize(torch.float32).view(
        batch,
        sequence_length,
        input_features,
    )
    expected = represented[:, valid].mean(dim=1) @ prepared_weight.dequantize(torch.float32).t()

    actual = nvfp4_triton.linear_mean(
        prepared_input.qdata,
        prepared_input.scale,
        input_scale,
        prepared_weight.qdata,
        prepared_weight.scale,
        weight_scale,
        None,
        batch,
        sequence_length,
        block_lengths,
    )

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
