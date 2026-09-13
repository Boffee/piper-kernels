"""Static INT8 input scaling follows weight storage through eager and compiled linears."""

import pytest
import torch

from piper_kernels.linear.convrot import convrot_int8_compile_options
from piper_kernels.linear.convrot.int8 import _ops
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm"),
        ],
    ),
]


def _static_reference(value, weight, bias=None):
    # FP64 rotation is independent of the fused preparation's accumulation order.
    rotated = rotate_groups(value.double(), weight.group_size)
    qdata = (rotated / weight.act_per_tensor_scale.double()).round().clamp(-128, 127)
    accumulated = qdata @ weight.qdata.double().T
    result = accumulated * weight.act_per_tensor_scale.double() * weight.scale.double().T
    if bias is not None:
        result = result + bias.double()
    return result.to(value.dtype)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [512, 5376, 65536])
def test_static_linear_uses_calibrated_scale_and_observes_mutation(device, dtype, width):
    torch.manual_seed(1027)
    input_scale = torch.tensor(0.00390625, device=device)
    weight = ConvRotInt8Tensor.from_quantized(
        torch.randint(-8, 9, (11, width), device=device, dtype=torch.int8),
        torch.full((11, 1), 0.0625, device=device),
        group_size=64,
        logical_dtype=dtype,
        act_per_tensor_scale=input_scale,
    )
    # These inputs rotate exactly even in the split path's BF16 workspace, isolating
    # clipping and static-scale propagation from intermediate rounding differences.
    value = torch.randint(-4, 5, (2, 3, width), device=device).to(dtype) / 8
    value[0, 0].zero_()
    bias = torch.arange(11, device=device, dtype=torch.float32) / 4
    original = None
    for scale in (0.00390625, 0.03125):
        input_scale.fill_(scale)
        expected = _static_reference(value, weight, bias)
        actual = torch.nn.functional.linear(value, weight, bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[0, 0], bias.to(dtype), rtol=0, atol=0)
        if original is None:
            original = actual
        else:
            assert not torch.equal(actual, original)


@pytest.mark.parametrize("device", _DEVICES)
def test_compiled_weight_observes_static_scale_mutation(device):
    torch.manual_seed(1028)
    input_scale = torch.tensor(0.03125, device=device)
    weight = ConvRotInt8Tensor.from_hp(
        torch.randn(96, 512, device=device, dtype=torch.bfloat16),
        group_size=64,
        act_per_tensor_scale=input_scale,
    )
    value = torch.randn(7, 512, device=device, dtype=weight.dtype) * 4
    compiled = torch.compile(
        torch.nn.functional.linear, fullgraph=True, options=convrot_int8_compile_options()
    )
    with torch.inference_mode():
        first = compiled(value, weight)
        torch.testing.assert_close(first, torch.nn.functional.linear(value, weight), rtol=0, atol=0)
        input_scale.fill_(0.125)
        second = compiled(value, weight)
        torch.testing.assert_close(
            second, torch.nn.functional.linear(value, weight), rtol=0, atol=0
        )
        assert not torch.equal(first, second)


@pytest.mark.parametrize("scale", [torch.ones(1), torch.ones((), dtype=torch.float16)])
def test_linear_validates_replaced_scale_storage(scale):
    weight = ConvRotInt8Tensor.from_hp(torch.ones(2, 64), group_size=64)
    weight.act_per_tensor_scale = scale
    with pytest.raises(ValueError, match="FP32 scalar"):
        torch.nn.functional.linear(torch.ones(1, 64), weight)


@pytest.mark.parametrize("device", _DEVICES)
def test_static_preparation_does_not_alias_checkpoint_scale(device):
    input_scale = torch.tensor(0.03125, device=device)
    _, row_scales = _ops.prepare_input(torch.zeros(2, 64, device=device), 64, None, input_scale)
    assert row_scales.untyped_storage().data_ptr() != input_scale.untyped_storage().data_ptr()
    torch.testing.assert_close(row_scales, input_scale.expand(2), rtol=0, atol=0)
    row_scales.zero_()
    assert input_scale.item() == 0.03125
