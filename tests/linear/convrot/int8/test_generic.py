"""Generic INT8 math works without tuned accelerator policies or exact cross-device rounding."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._input_activations import apply_input_activation
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import _backend, _generic, _ops, reference
from piper_kernels.linear.convrot.int8._generic import dispatch as generic_dispatch
from piper_kernels.linear.convrot.int8._generic import triton as generic_triton
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.int8 import _backend as int8_updates
from piper_kernels.weights.convrot.int8 import _quantization as int8_quantization
from piper_kernels.weights.convrot.int8 import triton as int8_weight_triton

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm GPU"),
        ],
    ),
]


@pytest.mark.parametrize("device", ["cuda", "xpu", "mps", "privateuseone"])
def test_generic_update_selection_does_not_query_architecture(monkeypatch, device):
    value = SimpleNamespace(device=torch.device(device))
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("architecture queried"))
    )
    assert int8_updates.select_add(value) is int8_updates.add_
    assert int8_updates.select_addmm(value) is int8_updates.addmm_
    assert _backend.select_preparation_backend(value) is _generic


def test_generic_package_exports_preparation():
    assert _generic.prepare_input is generic_dispatch.prepare_input


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
def test_rocm_uses_shared_triton_without_a_tuned_backend(monkeypatch):
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    value = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    assert generic_dispatch._use_triton(value)
    prepare = Mock(wraps=generic_triton.prepare_input)
    add = Mock(wraps=int8_weight_triton.add_)
    addmm = Mock(wraps=int8_weight_triton.addmm_)
    monkeypatch.setattr(generic_triton, "prepare_input", prepare)
    monkeypatch.setattr(int8_weight_triton, "add_", add)
    monkeypatch.setattr(int8_weight_triton, "addmm_", addmm)
    _ops.prepare_input(value, 256)
    weight = ConvRotInt8Tensor.from_hp(value, group_size=256)
    weight.add_(value)
    weight.addmm_(torch.eye(3, device="cuda", dtype=value.dtype), value)
    prepare.assert_called_once()
    add.assert_called_once()
    addmm.assert_called_once()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("activation", [None, "gelu_tanh", "swiglu"])
def test_generic_preparation_matches_math_and_preserves_output_storage(
    device, dtype, group_size, activation
):
    torch.manual_seed(975)
    width = 3 * group_size
    raw_width = width * (2 if activation == "swiglu" else 1)
    value = torch.randn(2, 3, raw_width * 2, device=device, dtype=dtype)[..., ::2]
    q_storage = torch.full((6 * width + 7,), 99, device=device, dtype=torch.int8)
    s_storage = torch.full((6 + 7,), -99.0, device=device)
    output = (q_storage[3:-4].reshape(2, 3, width), s_storage[3:-4].reshape(2, 3))
    actual = _generic.prepare_input(value, group_size, activation, out=output)
    expected = reference.prepare_input(apply_input_activation(value, activation), group_size)
    assert actual is output
    assert (actual[0].short() - expected[0].short()).abs().max().item() <= 1
    torch.testing.assert_close(
        actual[1], expected[1], rtol=max(2 * torch.finfo(dtype).eps, 1e-6), atol=1e-7
    )
    assert (q_storage[:3] == 99).all()
    assert (q_storage[-4:] == 99).all()
    assert (s_storage[:3] == -99).all()
    assert (s_storage[-4:] == -99).all()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_generic_quantize_dequantize_roundtrip(device, dtype, group_size):
    torch.manual_seed(678)
    value = torch.randn(13, 3 * group_size, device=device, dtype=dtype)
    quantized = ConvRotInt8Tensor.from_hp(value, group_size=group_size)
    actual = quantized.dequantize(torch.float32)
    relative_rms = (
        actual - value.float()
    ).square().mean().sqrt() / value.float().square().mean().sqrt()
    assert relative_rms.item() < 0.03
    assert actual.isfinite().all()


@pytest.mark.parametrize("device", _DEVICES)
def test_generic_preparation_custom_op_compiles_without_tuned_backend(monkeypatch, device):
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    value = torch.randn(3, 768, device=device)
    expected = _ops.prepare_input(value, 256)
    actual = torch.compile(_ops.prepare_input, fullgraph=True)(value, 256)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_wide_generic_preparation_uses_bounded_fallback(monkeypatch, device):
    unsupported = Mock(side_effect=AssertionError("wide row launched Triton"))
    monkeypatch.setattr(generic_triton, "prepare_input", unsupported)
    value = torch.randn(2, 32768, device=device, dtype=torch.bfloat16)
    actual = _generic.prepare_input(value, 256)
    expected = reference.prepare_input(value, 256)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 256), (2, 0)])
def test_empty_preparation_does_not_launch(shape):
    qdata, scale = _generic.prepare_input(torch.empty(shape), 256)
    assert qdata.shape == shape
    assert scale.shape == shape[:-1]
    assert (scale == 1e-30).all()


def test_generic_preparation_rejects_incompatible_outputs():
    value = torch.ones(2, 32)
    output = (torch.empty(2, 64, dtype=torch.int8)[:, ::2], torch.empty(2))
    with pytest.raises(ValueError, match="output storage is incompatible"):
        _generic.prepare_input(value, 16, out=output)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1e-31])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_generic_preparation_zero_and_tiny_scales(device, magnitude, dtype):
    value = torch.zeros(2, 256, device=device, dtype=dtype)
    value[:, 0] = magnitude
    qdata, scale = _generic.prepare_input(value, 256)
    rotated = rotate_groups(value.float(), 256)
    if device == "cuda":
        # The generic GPU path materializes a compact rotation workspace.
        rotated = rotated.to(dtype)
    expected_qdata, expected_scale = int8_quantization.dynamic_quantize_rows(rotated)
    expected_scale = expected_scale.squeeze(-1)
    assert torch.equal(qdata, expected_qdata)
    torch.testing.assert_close(scale, expected_scale, rtol=1e-6, atol=0)
    assert scale.isfinite().all()
