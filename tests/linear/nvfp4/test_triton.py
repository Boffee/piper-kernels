"""Tests for direct NVFP4 Triton preparation."""

import pytest
import torch

from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import triton as nvfp4_triton


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("rows", [127, 128, 129])
@pytest.mark.parametrize("activation_fn", [None, "swiglu"])
def test_static_preparation_matches_portable_decomposition(
    rows: int,
    activation_fn: str | None,
) -> None:
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

    expected = nvfp4_ops._compiled_prepare_static(input, per_tensor_scale, activation_fn)
    actual = nvfp4_triton.prepare_static(
        input,
        per_tensor_scale,
        swiglu=activation_fn == "swiglu",
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
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_static_projected_swiglu_matches_separate_epilogue(with_bias: bool) -> None:
    torch.manual_seed(502)
    raw = torch.randn(129, 160, device="cuda", dtype=torch.bfloat16)
    source_scale = torch.tensor(0.01, device="cuda", dtype=torch.float32)
    source_bias = torch.randn(160, device="cuda", dtype=torch.bfloat16) if with_bias else None
    activation_scale = torch.tensor(1.0 / 448.0, device="cuda", dtype=torch.float32)
    projected = (
        nvfp4_ops._compiled_scale_result(raw, source_scale)
        if source_bias is None
        else nvfp4_ops._compiled_scale_result_and_add_bias(raw, source_scale, source_bias)
    )

    expected_qdata, expected_scale, _ = nvfp4_ops._compiled_prepare_static(
        projected,
        activation_scale,
        "swiglu",
    )
    actual = nvfp4_triton.prepare_static_projected_swiglu(
        raw,
        activation_scale,
        source_scale,
        source_bias,
    )

    for expected_tensor, actual_tensor in zip(
        (expected_qdata, expected_scale),
        actual,
        strict=True,
    ):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_projection_epilogue_matches_portable_decomposition(with_bias: bool) -> None:
    torch.manual_seed(503)
    raw = torch.randn(129, 80, device="cuda", dtype=torch.bfloat16)
    global_scale = torch.tensor(0.01, device="cuda", dtype=torch.float32)
    bias = torch.randn(80, device="cuda", dtype=torch.bfloat16) if with_bias else None
    expected = (
        nvfp4_ops._compiled_scale_result(raw, global_scale)
        if bias is None
        else nvfp4_ops._compiled_scale_result_and_add_bias(raw, global_scale, bias)
    )
    actual = torch.empty_like(raw)

    nvfp4_triton.apply_projection_epilogue(raw, global_scale, bias, actual)

    assert torch.equal(actual, expected)
