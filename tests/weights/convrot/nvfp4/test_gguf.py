"""Tests for direct GGUF-to-ConvRot-NVFP4 conversion."""

import pytest
import torch
from _accelerators import exact_sm120_available
from gguf_format._fixtures import finite_packed, reference_case

from piper_kernels.gguf import GGUFQuantizationType
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.weights.convrot import triton as convrot_backend
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor


def _materialized_reference(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotated = torch.empty_like(dense)
    convrot_backend.rotate_input(dense, rotated, 64, num_warps=4)
    return nvfp4_ops._compiled_prepare_dynamic(rotated, None)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
def test_from_gguf_matches_materialized_reference(quant_type: GGUFQuantizationType) -> None:
    packed, dense = reference_case(quant_type, seed=840 + int(quant_type))
    expected = _materialized_reference(dense.cuda())
    actual = ConvRotNVFP4Tensor.from_gguf(
        packed.cuda(),
        quant_type=quant_type,
        group_size=64,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
    )

    for actual_tensor, expected_tensor in zip(
        (actual.qdata, actual.scale, actual.per_tensor_scale),
        expected,
        strict=True,
    ):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_copy_from_gguf_refills_existing_storage_and_scale(dtype: torch.dtype) -> None:
    first = finite_packed(GGUFQuantizationType.IQ4_XS)
    second = finite_packed(GGUFQuantizationType.IQ4_XS)
    weight = ConvRotNVFP4Tensor.from_gguf(
        first.cuda(),
        quant_type=GGUFQuantizationType.IQ4_XS,
        group_size=64,
        logical_dtype=dtype,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
    )
    expected = ConvRotNVFP4Tensor.from_gguf(
        second.cuda(),
        quant_type=GGUFQuantizationType.IQ4_XS,
        group_size=64,
        logical_dtype=dtype,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
    )
    qdata = weight.qdata
    scale = weight.scale
    per_tensor_scale = weight.per_tensor_scale

    result = weight.copy_from_gguf_(
        second.cuda(),
        quant_type=GGUFQuantizationType.IQ4_XS,
        compute_per_tensor_scale=True,
    )

    assert result is weight
    assert weight.dtype is dtype
    assert weight.qdata is qdata
    assert weight.scale is scale
    assert weight.per_tensor_scale is per_tensor_scale
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale, expected.scale)
    assert torch.equal(weight.per_tensor_scale, expected.per_tensor_scale)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("is_swizzled_scales", [False, True])
def test_from_gguf_without_global_scale_matches_from_hp(
    is_swizzled_scales: bool,
) -> None:
    packed, dense = reference_case(GGUFQuantizationType.BF16, seed=870, rows=128)
    expected = ConvRotNVFP4Tensor.from_hp(
        dense.to(device="cuda", dtype=torch.bfloat16),
        group_size=64,
        is_swizzled_scales=is_swizzled_scales,
    )
    actual = ConvRotNVFP4Tensor.from_gguf(
        packed.cuda(),
        quant_type=GGUFQuantizationType.BF16,
        group_size=64,
        is_swizzled_scales=is_swizzled_scales,
    )

    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    assert actual.per_tensor_scale is None


def test_from_gguf_rejects_cpu_storage() -> None:
    packed = finite_packed(GGUFQuantizationType.Q4_0)

    with pytest.raises(ValueError, match="requires exact NVIDIA SM120"):
        ConvRotNVFP4Tensor.from_gguf(
            packed,
            quant_type=GGUFQuantizationType.Q4_0,
            group_size=64,
        )
