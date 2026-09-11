"""Shared quantized weights must retain mmap storage through layer installation."""

import json

import pytest
import torch

from piper_kernels.conv3d.convrot.int8 import ConvRotInt8Conv3d
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor


def _weight(device="cpu"):
    return ConvRotInt8Tensor.from_quantized(
        torch.zeros(8, 3, 3, 3, 64, device=device, dtype=torch.int8),
        torch.ones(8, 1, device=device),
        group_size=64,
        logical_dtype=torch.float16,
        act_per_tensor_scale=torch.tensor(0.02, device=device),
    )


def _module(device="cpu"):
    return ConvRotInt8Conv3d(
        _weight(device),
        torch.zeros(8, device=device, dtype=torch.float16),
        stride=(2, 2, 2),
        padding="reflect_right",
    )


def test_layer_installs_weight_without_copying_quantized_storage():
    weight = _weight()
    layer = ConvRotInt8Conv3d(weight, padding="reflect")
    assert set(dict(layer.named_parameters())) == {"weight"}
    assert dict(layer.named_buffers()) == {}
    for name in weight.__tensor_flatten__()[0]:
        assert getattr(layer.weight, name).data_ptr() == getattr(weight, name).data_ptr()
    assert set(layer.state_dict()) == {"weight"}


def test_mmap_state_dict_assign_retains_all_weight_storage(tmp_path):
    path = tmp_path / "conv.pt"
    original = _module()
    torch.save(original.state_dict(), path)
    with torch.serialization.safe_globals([ConvRotInt8Tensor]):
        state = torch.load(path, mmap=True, weights_only=True)
    loaded = _module("meta")
    loaded.load_state_dict(state, assign=True)
    assert loaded.weight.act_per_tensor_scale.item() == pytest.approx(0.02)
    for name in loaded.weight.__tensor_flatten__()[0]:
        storage = getattr(loaded.weight, name)
        assert storage.data_ptr() == getattr(state["weight"], name).data_ptr()
        assert not storage.untyped_storage().resizable()
    assert loaded.bias.data_ptr() == state["bias"].data_ptr()
    del state
    result = loaded(torch.ones(1, 64, 3, 4, 4, dtype=torch.float16))
    torch.testing.assert_close(result, torch.zeros(1, 8, 2, 2, 2, dtype=torch.float16))


def test_safetensors_loading_keeps_mapped_storage_after_file_context_closes(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    path = tmp_path / "conv.safetensors"
    original = _weight()
    names, metadata = original.__tensor_flatten__()
    metadata["dtype"] = str(metadata["dtype"]).removeprefix("torch.")
    safetensors_torch.save_file(
        {name: getattr(original, name) for name in names},
        path,
        metadata={"weight": json.dumps(metadata)},
    )
    with safetensors.safe_open(path, framework="pt", device="cpu") as checkpoint:
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}  # noqa: SIM118
        metadata = json.loads(checkpoint.metadata()["weight"])
        weight = ConvRotInt8Tensor.from_quantized(
            **tensors,
            group_size=metadata["group_size"],
            logical_dtype=getattr(torch, metadata["dtype"]),
        )
        loaded = ConvRotInt8Conv3d(weight, padding="reflect")
        for name in names:
            assert getattr(loaded.weight, name).data_ptr() == tensors[name].data_ptr()
            assert not getattr(loaded.weight, name).untyped_storage().resizable()
    del tensors
    assert torch.count_nonzero(loaded(torch.ones(1, 64, 3, 4, 4, dtype=torch.float16))) == 0


def test_layer_requires_static_conv3d_weight():
    with pytest.raises(TypeError, match="5-D ConvRotInt8Tensor"):
        ConvRotInt8Conv3d(
            ConvRotInt8Tensor.from_hp(torch.ones(8, 64), group_size=64), padding="reflect"
        )
    weight = _weight()
    weight.act_per_tensor_scale = None
    with pytest.raises(ValueError, match="static activation scale"):
        ConvRotInt8Conv3d(weight, padding="reflect")


@pytest.mark.parametrize("config", [{"stride": (0, 1, 1)}, {"padding": "zeros"}])
def test_layer_rejects_invalid_geometry(config):
    with pytest.raises(ValueError, match="ConvRot INT8 Conv3D"):
        ConvRotInt8Conv3d(_weight(), **({"padding": "reflect"} | config))


def test_layer_dtype_conversion_preserves_fp32_scales():
    layer = _module().float()
    assert layer.weight.dtype is torch.float32
    assert layer.weight.qdata.dtype is torch.int8
    assert layer.weight.scale.dtype is torch.float32
    assert layer.weight.act_per_tensor_scale.dtype is torch.float32
