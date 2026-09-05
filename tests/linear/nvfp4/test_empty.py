"""Empty batches preserve linear shapes through semantic and prepared NVFP4 paths."""

import pytest
import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor, convrot_nvfp4_compile_options
from piper_kernels.linear.convrot.nvfp4 import _ops as convrot_ops
from piper_kernels.linear.nvfp4 import (
    PiperNVFP4Tensor,
    _layout,
    _ops,
    nvfp4_compile_options,
    reference,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("group_size", [0, 16])
@pytest.mark.parametrize("activation", [None, "gelu_tanh", "swiglu"])
def test_empty_reference_preparation_and_projection(dtype, dynamic, group_size, activation):
    width = 64 if activation == "swiglu" else 32
    value = torch.empty(2, 0, width, dtype=dtype)
    scale = None if dynamic else torch.tensor(0.25)
    prepared = reference.prepare_input(value, scale, dynamic, activation, group_size=group_size)
    assert prepared[0].shape == (0, 16)
    assert prepared[1].shape == _layout.scale_shape(0, 32)
    assert prepared[2].item() == (0 if dynamic else 0.25)
    weight = reference.prepare_input(torch.ones(16, 32), None, True, group_size=group_size)
    actual = reference.linear_prepared(*prepared, *weight, None, dtype)
    assert actual.shape == (0, 16)
    assert actual.dtype is dtype
    actual = reference.linear(value[..., :32], *weight, scale, None, dynamic, group_size=group_size)
    assert actual.shape == (2, 0, 16)
    assert actual.dtype is dtype


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("group_size", [0, 16])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_empty_cuda_linear_matches_dense(dtype, dynamic, group_size, compiled, prepared):
    torch._dynamo.reset()
    source = torch.ones(128, 256, device="cuda", dtype=dtype)
    scale = None if dynamic else torch.tensor(0.25, device="cuda")
    quantization = QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )
    factory = ConvRotNVFP4Tensor if group_size else PiperNVFP4Tensor
    weight = factory.from_hp(
        source.bfloat16(),
        compute_per_tensor_scale=True,
        act_per_tensor_scale=scale,
        is_swizzled_scales=True,
        act_quant_kwargs=quantization,
        **({"group_size": group_size} if group_size else {}),
    ).to(dtype=dtype)
    bias = torch.ones(128, device="cuda", dtype=torch.float32)

    def projections(value):
        if not prepared:
            return F.linear(value, weight), F.linear(value, weight, bias)
        encoded = (
            convrot_ops.prepare_input(value, scale, dynamic, group_size)
            if group_size
            else _ops.prepare_input(value, scale, dynamic)
        )
        operands = (*encoded, weight.qdata, weight.scale, weight.per_tensor_scale)
        shape = (*value.shape[:-1], 128)
        return (
            _ops.linear_prepared(*operands, None, dtype).reshape(shape),
            _ops.linear_prepared(*operands, bias, dtype).reshape(shape),
        )

    options = convrot_nvfp4_compile_options() if group_size else nvfp4_compile_options()
    call = torch.compile(projections, fullgraph=True, options=options) if compiled else projections
    with torch.no_grad():
        for shape in [(0, 256), (2, 0, 256)]:
            value = torch.empty(shape, device="cuda", dtype=dtype)
            for result in call(value):
                expected = F.linear(value, source)
                torch.testing.assert_close(result, expected)


def test_empty_static_preparation_still_requires_scale():
    with pytest.raises(ValueError, match="per-tensor scale"):
        reference.prepare_input(torch.empty(0, 32), None, False)
