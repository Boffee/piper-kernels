"""Weight construction and checkpoint storage are independent of operator packages."""

import importlib
import subprocess
import sys

import pytest
import torch

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor


@pytest.mark.parametrize("format_name", ["int8", "nvfp4", "convrot_nvfp4"])
def test_quantization_and_updates_do_not_import_linear(format_name):
    script = """
import importlib.abc
import sys
import torch

class NoLinear(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "piper_kernels.linear" or fullname.startswith("piper_kernels.linear."):
            raise AssertionError(f"Weight storage imported an operator: {fullname}")

sys.meta_path.insert(0, NoLinear())
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
from piper_kernels.weights.sharding import shard_quantized_weight

source = torch.randn(8, 64)
format_name = sys.argv[1]
if format_name == "int8":
    weight = ConvRotInt8Tensor.from_hp(source, group_size=16)
elif format_name == "nvfp4":
    weight = PiperNVFP4Tensor.from_hp(source, compute_per_tensor_scale=True)
else:
    weight = ConvRotNVFP4Tensor.from_hp(source, group_size=16, compute_per_tensor_scale=True)
assert weight.dequantize().shape == source.shape
assert weight.clone().float().shape == source.shape
assert shard_quantized_weight(weight, dim=0, start=0, length=4).shape == (4, 64)
weight.add_(torch.ones_like(source), alpha=0.25)
weight.addmm_(torch.ones(8, 2), torch.ones(2, 64), alpha=0.125)
assert torch.isfinite(weight.dequantize()).all()
assert not any(name.startswith("piper_kernels.linear") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, format_name], check=True)


@pytest.mark.parametrize("weight_type", [ConvRotInt8Tensor, PiperNVFP4Tensor, ConvRotNVFP4Tensor])
def test_mmap_checkpoint_and_parameter_assignment_preserve_storage(tmp_path, weight_type):
    source = torch.randn(8, 64)
    kwargs = {} if weight_type is PiperNVFP4Tensor else {"group_size": 16}
    if weight_type is not ConvRotInt8Tensor:
        kwargs.update(compute_per_tensor_scale=True, act_per_tensor_scale=torch.tensor(0.125))
    weight = weight_type.from_hp(source, **kwargs)
    checkpoint = tmp_path / "weight.pt"
    torch.save({"weight": weight}, checkpoint)
    with torch.serialization.safe_globals([weight_type]):
        restored = torch.load(checkpoint, mmap=True, weights_only=True)["weight"]
    assert type(restored) is weight_type
    assert restored.__class__.__module__.startswith("piper_kernels.weights.")
    torch.testing.assert_close(restored.dequantize(), weight.dequantize(), rtol=0, atol=0)
    layer = torch.nn.Linear(64, 8, bias=False, device="meta")
    layer.load_state_dict({"weight": restored}, assign=True)
    names, metadata = restored.__tensor_flatten__()
    assert layer.weight.__tensor_flatten__()[1] == metadata
    for name in names:
        stored = getattr(restored, name)
        installed = getattr(layer.weight, name)
        assert installed.data_ptr() == stored.data_ptr()
        assert not installed.untyped_storage().resizable()


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("piper_kernels.linear.convrot", "ConvRotInt8Tensor"),
        ("piper_kernels.linear.convrot.int8", "ConvRotInt8Tensor"),
        ("piper_kernels.linear.convrot.nvfp4", "ConvRotNVFP4Tensor"),
        ("piper_kernels.linear.nvfp4", "PiperNVFP4Tensor"),
    ],
)
def test_tensor_exports_live_only_in_weight_packages(module, name):
    assert not hasattr(importlib.import_module(module), name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module + ".tensor")
