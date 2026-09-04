"""Tests for Piper's semantic TorchAO NVFP4 wrapper."""

import pytest
import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor, nvfp4_compile_options


def _quantization(dynamic: bool) -> QuantizeTensorToNVFP4Kwargs:
    return QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )


def test_from_torchao_reuses_storage_and_metadata() -> None:
    source = TorchAONVFP4Tensor(
        torch.empty(128, 128, dtype=torch.uint8, device="meta"),
        torch.empty(128, 16, dtype=torch.float8_e4m3fn, device="meta"),
        16,
        torch.bfloat16,
        torch.empty((), device="meta"),
        torch.empty((), device="meta"),
        True,
        False,
        _quantization(False),
    )

    wrapped = PiperNVFP4Tensor.from_torchao(source)

    assert type(wrapped) is PiperNVFP4Tensor
    assert wrapped.qdata is source.qdata
    assert wrapped.scale is source.scale
    assert wrapped.per_tensor_scale is source.per_tensor_scale
    assert wrapped.act_per_tensor_scale is source.act_per_tensor_scale
    assert wrapped.act_quant_kwargs == source.act_quant_kwargs
    assert PiperNVFP4Tensor.from_torchao(wrapped) is wrapped


def test_dequantize_accepts_an_output_dtype() -> None:
    source = torch.randn(32, 256, dtype=torch.bfloat16)
    weight = PiperNVFP4Tensor.from_hp(source)

    actual = weight.dequantize(torch.float32)
    expected = TorchAONVFP4Tensor.dequantize(weight, torch.float32)

    assert actual.dtype is torch.float32
    assert torch.equal(actual, expected)


def test_from_hp_matches_torchao_quantization_with_computed_global_scale() -> None:
    torch.manual_seed(418)
    source = torch.randn(128, 256, dtype=torch.bfloat16).requires_grad_()
    per_tensor_scale = per_tensor_amax_to_scale(source.float().abs().amax())

    weight = PiperNVFP4Tensor.from_hp(
        source,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(True),
    )
    expected = TorchAONVFP4Tensor.to_nvfp4(
        source.detach(),
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(True),
    )

    assert type(weight) is PiperNVFP4Tensor
    assert not weight.requires_grad
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))
    assert torch.equal(weight.per_tensor_scale, per_tensor_scale)


def test_from_hp_clamps_a_tiny_computed_global_scale() -> None:
    source = torch.tensor(
        [[1e-37, 5e-38, 0.0, -2.5e-38] * 4],
        dtype=torch.float32,
    )
    minimum_scale = torch.finfo(torch.float32).tiny / torch.finfo(torch.float8_e4m3fn).tiny
    expected = TorchAONVFP4Tensor.to_nvfp4(
        source,
        per_tensor_scale=per_tensor_amax_to_scale(source.abs().amax()).clamp_min(minimum_scale),
    )

    weight = PiperNVFP4Tensor.from_hp(
        source,
        compute_per_tensor_scale=True,
    )

    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))
    assert torch.equal(weight.per_tensor_scale, expected.per_tensor_scale)


def test_device_copy_preserves_piper_wrapper() -> None:
    source = PiperNVFP4Tensor(
        torch.empty(128, 128, dtype=torch.uint8),
        torch.empty(128, 16, dtype=torch.float8_e4m3fn),
        16,
        torch.bfloat16,
        torch.empty(()),
        torch.empty(()),
        True,
        False,
        _quantization(False),
    )

    moved = source.to(device="meta")

    assert type(moved) is PiperNVFP4Tensor
    assert moved.device.type == "meta"


def test_dtype_copy_preserves_concrete_piper_subclass() -> None:
    class DerivedPiperNVFP4Tensor(PiperNVFP4Tensor):
        pass

    source = DerivedPiperNVFP4Tensor(
        torch.empty(128, 128, dtype=torch.uint8),
        torch.empty(128, 16, dtype=torch.float8_e4m3fn),
        16,
        torch.bfloat16,
        torch.empty(()),
        torch.empty(()),
        True,
        False,
        _quantization(False),
    )

    moved = source.to(dtype=torch.float16)

    assert type(moved) is DerivedPiperNVFP4Tensor
    assert moved.orig_dtype is torch.float16
    assert moved.qdata is source.qdata
    assert moved.scale is source.scale
    assert moved.per_tensor_scale is source.per_tensor_scale
    assert moved.act_per_tensor_scale is source.act_per_tensor_scale


@pytest.mark.parametrize("positional", [False, True])
def test_explicit_dtype_copy_duplicates_storage(positional: bool) -> None:
    source = PiperNVFP4Tensor(
        torch.empty(128, 128, dtype=torch.uint8),
        torch.empty(128, 16, dtype=torch.float8_e4m3fn),
        16,
        torch.bfloat16,
        torch.empty(()),
        torch.empty(()),
        True,
        False,
        _quantization(False),
    )

    moved = (
        source.to(torch.float16, False, True)
        if positional
        else source.to(dtype=torch.float16, copy=True)
    )

    assert moved.orig_dtype is torch.float16
    assert moved.qdata is not source.qdata
    assert moved.scale is not source.scale
    assert moved.per_tensor_scale is not source.per_tensor_scale
    assert moved.act_per_tensor_scale is not source.act_per_tensor_scale


@pytest.mark.gpu
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_semantic_linear_matches_torchao(dynamic: bool, with_bias: bool) -> None:
    torch.manual_seed(417)
    input = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    activation_scale = None if dynamic else per_tensor_amax_to_scale(input.abs().amax())
    torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
        weight,
        per_tensor_scale=per_tensor_amax_to_scale(weight.abs().amax()),
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(dynamic),
    )
    piper_weight = PiperNVFP4Tensor.from_torchao(torchao_weight)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16) if with_bias else None

    expected = F.linear(input, torchao_weight, bias)
    actual = F.linear(input, piper_weight, bias)

    assert torch.equal(actual, expected)
    assert torch.ops.piper_kernels.nvfp4_linear.default is not None


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_fp16_autocast_normalizes_semantic_linear(compiled: bool) -> None:
    torch.manual_seed(419)
    input = torch.randn(17, 256, device="cuda", dtype=torch.float32) * 0.01  # noqa: A001
    source = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.01
    weight = PiperNVFP4Tensor.from_hp(
        source,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(True),
    )
    bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.01

    def projection(value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weight, offset)

    call = (
        torch.compile(projection, fullgraph=True, options=nvfp4_compile_options())
        if compiled
        else projection
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual = call(input, bias)
    with torch.no_grad():
        expected = F.linear(input.half(), weight.to(dtype=torch.float16), bias.half())

    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
