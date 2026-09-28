"""Quantized weights convert and view the same way under inference mode as under no_grad.

Inference mode skips autograd dispatch, where PyTorch decomposes composite ops such
as ``Tensor.to``, so those ops reach the wrappers' ``__torch_dispatch__`` intact. It
still tracks views of a weight created outside it, which requires normal-tensor views.
"""

import pytest
import torch

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

_GRAD_MODES = [
    pytest.param(torch.no_grad, id="no_grad"),
    pytest.param(torch.inference_mode, id="inference_mode"),
]


@pytest.fixture(params=[ConvRotInt8Tensor, PiperNVFP4Tensor, ConvRotNVFP4Tensor])
def weight(request):
    torch.manual_seed(163)
    source = torch.randn(32, 64, dtype=torch.bfloat16)
    cls = request.param
    return cls.from_hp(source, **({} if cls is PiperNVFP4Tensor else {"group_size": 64}))


@pytest.mark.parametrize("grad_mode", _GRAD_MODES)
@pytest.mark.parametrize(
    "conversion", ["dtype", "keyword", "method", "device", "device_dtype", "other", "type_as"]
)
def test_unchanged_conversion_returns_weight(weight, grad_mode, conversion):
    like = torch.empty(0, dtype=torch.bfloat16)
    conversions = {
        "dtype": lambda: weight.to(torch.bfloat16),
        "keyword": lambda: weight.to(dtype=torch.bfloat16),
        "method": weight.bfloat16,
        "device": lambda: weight.to("cpu", non_blocking=True),
        "device_dtype": lambda: weight.to("cpu", torch.bfloat16),
        "other": lambda: weight.to(like),
        "type_as": lambda: weight.type_as(like),
    }
    with grad_mode():
        assert conversions[conversion]() is weight


@pytest.mark.parametrize("grad_mode", _GRAD_MODES)
@pytest.mark.parametrize(
    "conversion", ["dtype", "keyword", "method", "device_dtype", "other", "dtype_layout"]
)
def test_logical_dtype_conversion_reuses_quantized_storage(weight, grad_mode, conversion):
    conversions = {
        "dtype": lambda: weight.to(torch.float16),
        "keyword": lambda: weight.to(dtype=torch.float16),
        "method": weight.half,
        "device_dtype": lambda: weight.to("cpu", torch.float16),
        "other": lambda: weight.to(torch.empty(0, dtype=torch.float16)),
        "dtype_layout": lambda: torch.ops.aten.to.dtype_layout(weight, dtype=torch.float16),
    }
    with grad_mode():
        converted = conversions[conversion]()

    names, _ = weight.__tensor_flatten__()
    assert type(converted) is type(weight)
    assert converted.dtype is torch.float16
    assert converted.__tensor_flatten__()[0] == names
    assert all(getattr(converted, name) is getattr(weight, name) for name in names)
    assert torch.equal(converted.dequantize(torch.float32), weight.dequantize(torch.float32))


_VIEWS = {
    "t": lambda weight: weight.t(),
    "transpose": lambda weight: weight.transpose(0, 1),
    "view": lambda weight: weight.view(weight.shape),
    "view_as": lambda weight: weight.view_as(weight),
    "detach": lambda weight: weight.detach(),
    "as_strided": lambda weight: weight.as_strided(weight.shape, weight.stride()),
    "parameter": lambda weight: torch.nn.Parameter(weight, requires_grad=False),
}


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("view", list(_VIEWS))
def test_view_of_weight_created_outside_inference_mode_matches_no_grad(weight, view, compiled):
    torch.compiler.reset()
    function = _VIEWS[view]
    call = torch.compile(function, backend="aot_eager", fullgraph=True) if compiled else function
    with torch.no_grad():
        expected = function(weight)
    with torch.inference_mode():
        actual = call(weight)

    assert type(actual) is type(expected)
    assert not actual.is_inference()
    assert torch._C._is_alias_of(actual, weight)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert torch.equal(actual.dequantize(torch.float32), expected.dequantize(torch.float32))


def test_matmul_with_transposed_weight_matches_no_grad(weight):
    if isinstance(weight, ConvRotNVFP4Tensor):
        pytest.skip("ConvRot NVFP4 linear requires SM120 operands")
    torch.manual_seed(164)
    activation = torch.randn(3, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = activation @ weight.t()
    with torch.inference_mode():
        actual = activation @ weight.t()

    assert torch.equal(actual, expected)
