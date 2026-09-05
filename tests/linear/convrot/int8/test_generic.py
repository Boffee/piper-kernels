"""Generic INT8 math works without tuned accelerator policies or exact cross-device rounding."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._input_activations import apply_input_activation
from piper_kernels.linear.convrot import ConvRotInt8Tensor
from piper_kernels.linear.convrot.int8 import _backend, _generic, _ops, reference
from piper_kernels.linear.convrot.int8._amd import triton as amd
from piper_kernels.linear.convrot.int8._generic import triton as generic_triton
from piper_kernels.linear.convrot.int8._nvidia import triton as nvidia

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
    assert _backend.select_add(value) is _generic.add_
    assert _backend.select_addmm(value) is _generic.addmm_
    assert _backend.select_preparation_backend(value) is _generic


def test_backend_update_exports_are_shared():
    assert amd.add_ is nvidia.add_ is _generic.add_
    assert amd.addmm_ is nvidia.addmm_ is _generic.addmm_


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_host_and_fake_devices_do_not_initialize_triton(monkeypatch, device):
    monkeypatch.setattr(generic_triton.triton.runtime, "driver", None)
    assert not generic_triton.supports_device(torch.device(device))


@pytest.mark.parametrize(
    ("backend", "architecture", "device", "supported"),
    [
        ("cuda", 60, "cuda", False),
        ("cuda", 70, "cuda", True),
        ("hip", "gfx1036", "cuda", True),
        ("hip", "gfx9999", "cuda", True),
        ("xpu", "future_device", "xpu", True),
    ],
)
def test_generic_triton_probes_driver_not_model_allowlist(
    monkeypatch, backend, architecture, device, supported
):
    driver = SimpleNamespace(
        get_active_torch_device=lambda: torch.device(device),
        get_current_target=lambda: SimpleNamespace(backend=backend, arch=architecture),
    )
    monkeypatch.setattr(generic_triton.triton.runtime, "driver", SimpleNamespace(active=driver))
    assert generic_triton.supports_device(torch.device(device)) is supported
    assert not generic_triton.supports_device(torch.device("mps"))


def test_missing_driver_uses_pytorch(monkeypatch):
    driver = SimpleNamespace(get_active_torch_device=Mock(side_effect=RuntimeError("no driver")))
    monkeypatch.setattr(generic_triton.triton.runtime, "driver", SimpleNamespace(active=driver))
    assert not generic_triton.supports_device(torch.device("cuda"))


def test_driver_capabilities_are_not_reused_for_a_different_gpu(monkeypatch):
    driver = SimpleNamespace(
        get_active_torch_device=lambda: torch.device("cuda:0"),
        get_current_target=Mock(side_effect=AssertionError("wrong GPU capability query")),
    )
    monkeypatch.setattr(generic_triton.triton.runtime, "driver", SimpleNamespace(active=driver))
    assert not generic_triton.supports_device(torch.device("cuda:1"))


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
def test_rocm_uses_shared_triton_without_a_tuned_backend(monkeypatch):
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    value = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    assert _generic._use_triton(value)
    prepare = Mock(wraps=generic_triton.prepare_input)
    add = Mock(wraps=generic_triton.add_)
    addmm = Mock(wraps=generic_triton.addmm_)
    monkeypatch.setattr(generic_triton, "prepare_input", prepare)
    monkeypatch.setattr(generic_triton, "add_", add)
    monkeypatch.setattr(generic_triton, "addmm_", addmm)
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
@pytest.mark.parametrize("operation", ["add", "addmm"])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seed", [None, 123])
def test_updates_work_without_tuned_backends(monkeypatch, device, operation, fallback, dtype, seed):
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    if fallback:
        monkeypatch.setattr(_generic, "_triton_backend", None)
    torch.manual_seed(671)
    value = torch.randn(17, 256, device=device, dtype=dtype)
    weight = ConvRotInt8Tensor.from_hp(value, group_size=64)
    replay = weight.clone()
    qdata, scale = weight.qdata, weight.scale
    before = weight.dequantize(torch.float32)
    update = torch.randn(17, 512, device=device, dtype=dtype)[:, ::2]
    left = torch.randn(17, 3, device=device, dtype=dtype)
    right = torch.randn(3, 512, device=device, dtype=dtype)[:, ::2]
    if operation == "add":
        weight.add_(update, alpha=0.25, rounding_seed=seed)
        replay.add_(update, alpha=0.25, rounding_seed=seed)
        expected = before + 0.25 * update.float()
    else:
        weight.addmm_(left, right, beta=0.5, alpha=0.25, rounding_seed=seed)
        replay.addmm_(left, right, beta=0.5, alpha=0.25, rounding_seed=seed)
        expected = 0.5 * before + 0.25 * (left.float() @ right.float())
    actual = weight.dequantize(torch.float32)
    relative_rms = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rms.item() < 0.04
    assert actual.isfinite().all()
    assert weight.qdata is qdata
    assert weight.scale is scale
    assert torch.equal(weight.qdata, replay.qdata)
    assert torch.equal(weight.scale, replay.scale)


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
    expected_qdata, expected_scale = reference.prepare_input(value, 256)
    assert torch.equal(qdata, expected_qdata)
    torch.testing.assert_close(scale, expected_scale, rtol=1e-6, atol=0)
    assert scale.isfinite().all()


def test_execution_error_is_not_retried_after_an_inplace_update(monkeypatch):
    qdata, scale = torch.zeros(2, 32, dtype=torch.int8), torch.ones(2, 1)

    def partial_update(*args):
        qdata.fill_(3)
        raise RuntimeError("kernel execution failed")

    fallback = Mock(side_effect=AssertionError("unsafe retry"))
    monkeypatch.setattr(_generic, "_use_triton", lambda value: True)
    monkeypatch.setattr(generic_triton, "add_", partial_update)
    monkeypatch.setattr(reference, "add_", fallback)
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        _generic.add_(qdata, scale, torch.ones(2, 32), 16, 1.0)
    assert (qdata == 3).all()
    fallback.assert_not_called()
