"""Convolution weights share the ConvRot conversion and storage lifecycle."""

import pytest
import torch

from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.int8._quantization import dynamic_quantize_rows


@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_quantization_and_dequantization_use_spatial_channel_groups(group_size, dtype):
    torch.manual_seed(2468)
    source = torch.randn(8, max(64, group_size), 3, 3, 3, dtype=dtype).mul_(0.02)
    channels_last = source.float().permute(0, 2, 3, 4, 1).contiguous()
    rotated = rotate_groups(channels_last, group_size)
    expected_qdata, expected_scale = dynamic_quantize_rows(rotated.flatten(1))
    weight = ConvRotInt8Tensor.from_hp(source, group_size=group_size)
    assert weight.shape == source.shape
    assert weight.dtype is dtype
    assert weight.act_per_tensor_scale is None  # Weight conversion requires no activation data.
    torch.testing.assert_close(weight.qdata.flatten(1), expected_qdata)
    torch.testing.assert_close(weight.scale, expected_scale)
    # FP64 gives an independent inverse-rotation oracle across CPU BLAS implementations.
    dequantized = (
        rotate_groups(
            weight.qdata.double() * weight.scale.double().view(-1, 1, 1, 1, 1),
            group_size,
        )
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )
    torch.testing.assert_close(
        weight.dequantize(torch.float32), dequantized.float(), atol=2e-8, rtol=2e-6
    )
    assert weight.dequantize().dtype is dtype
    # Quantization should be a useful approximation of the original logical weight.
    assert (weight.dequantize(torch.float32) - source.float()).square().mean().sqrt() < 0.0003


@pytest.mark.parametrize(
    "scale",
    [
        torch.ones(1),
        torch.tensor(0.02, dtype=torch.float16),
        torch.tensor(0.02, requires_grad=True),
    ],
)
def test_invalid_activation_storage_is_rejected(scale):
    with pytest.raises(ValueError, match="activation scale"):
        ConvRotInt8Tensor.from_hp(
            torch.ones(4, 64, 3, 3, 3), group_size=64, act_per_tensor_scale=scale
        )


def test_activation_scale_must_share_weight_device():
    with pytest.raises(ValueError, match="share the weight device"):
        ConvRotInt8Tensor.from_quantized(
            torch.empty(4, 3, 3, 3, 64, dtype=torch.int8, device="meta"),
            torch.empty(4, 1, device="meta"),
            group_size=64,
            act_per_tensor_scale=torch.tensor(0.02),
        )


@pytest.mark.parametrize("operation", ["linear", "transpose", "add", "addmm", "gguf"])
def test_matrix_operations_reject_convolution_weights(operation):
    weight = ConvRotInt8Tensor.from_hp(torch.ones(4, 64, 3, 3, 3), group_size=64)
    operations = {
        "linear": lambda: torch.nn.functional.linear(torch.ones(2, 64), weight),
        "transpose": weight.t,
        "add": lambda: weight.add_(torch.ones(4, 64, 3, 3, 3)),
        "addmm": lambda: weight.addmm_(torch.ones(4, 1), torch.ones(1, 64)),
        "gguf": lambda: weight.copy_from_gguf_(torch.empty(0)),
    }
    with pytest.raises(NotImplementedError, match="2-D"):
        operations[operation]()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size", [64, 256])
def test_gpu_dequantization_restores_logical_convolution_weight(group_size):
    torch.manual_seed(2468)
    source = torch.randn(8, group_size, 3, 3, 3, device="cuda", dtype=torch.float16) * 0.02
    weight = ConvRotInt8Tensor.from_hp(source, group_size=group_size)
    reconstructed = weight.dequantize(torch.float32)
    assert reconstructed.shape == source.shape
    assert reconstructed.is_contiguous()
    assert reconstructed.dtype is torch.float32
    assert (reconstructed - source.float()).square().mean().sqrt() < 0.0003


@pytest.mark.parametrize("operation", ["to", "copy", "aten"])
def test_convolution_weight_rejects_unsupported_memory_format(operation):
    weight = ConvRotInt8Tensor.from_hp(torch.ones(2, 64, 3, 3, 3), group_size=64)
    conversions = {
        "to": lambda: weight.to(memory_format=torch.channels_last_3d),
        "copy": lambda: weight.to(copy=True, memory_format=torch.channels_last_3d),
        "aten": lambda: torch.ops.aten._to_copy.default(
            weight, memory_format=torch.channels_last_3d
        ),
    }
    with pytest.raises(NotImplementedError, match="contiguous memory format"):
        conversions[operation]()
