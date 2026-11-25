"""Weight construction and checkpoint storage are independent of operator packages."""

import importlib
import subprocess
import sys

import pytest
import torch

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor


def _sample_trellis_weight(*, variable_rows: bool = False) -> ConvRotTrellisTensor:
    generator = torch.Generator().manual_seed(41 if variable_rows else 40)
    rows, block_size, blocks_per_row, group_size = 8, 16, 4, 16
    block_scales = (torch.rand(rows, blocks_per_row, generator=generator) + 0.5).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(rows, 1, generator=generator) * 0.05 + 1e-3
    if variable_rows:
        row_widths = torch.tensor([6, 8, 9, 12, 7, 10, 11, 8], dtype=torch.uint8)
        bitstreams = torch.randint(
            0,
            256,
            (blocks_per_row * int(row_widths.sum().item()),),
            dtype=torch.uint8,
            generator=generator,
        )
        return ConvRotTrellisTensor.from_quantized(
            bitstreams,
            block_scales,
            scale,
            block_size=block_size,
            group_size=group_size,
            row_widths=row_widths,
        )
    bitstreams = torch.randint(
        0, 256, (rows, blocks_per_row, 9), dtype=torch.uint8, generator=generator
    )
    return ConvRotTrellisTensor.from_quantized(
        bitstreams,
        block_scales,
        scale,
        block_size=block_size,
        group_size=group_size,
    )


def test_quantization_and_updates_do_not_import_linear():
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
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
from piper_kernels.weights.sharding import shard_quantized_weight

# All formats share the same import barrier, with fresh weights for each case.
for weight_type, kwargs in (
    (ConvRotInt8Tensor, {"group_size": 16}),
    (PiperNVFP4Tensor, {"compute_per_tensor_scale": True}),
    (ConvRotNVFP4Tensor, {"group_size": 16, "compute_per_tensor_scale": True}),
):
    print(f"checking {weight_type.__name__}", flush=True)
    source = torch.randn(8, 64)
    weight = weight_type.from_hp(source, **kwargs)
    assert weight.dequantize().shape == source.shape
    assert weight.clone().float().shape == source.shape
    assert shard_quantized_weight(weight, dim=0, start=0, length=4).shape == (4, 64)
    weight.add_(torch.ones_like(source), alpha=0.25)
    weight.addmm_(torch.ones(8, 2), torch.ones(2, 64), alpha=0.125)
    assert torch.isfinite(weight.dequantize()).all()
    assert not any(name.startswith("piper_kernels.linear") for name in sys.modules)

trellis = ConvRotTrellisTensor.from_quantized(
    torch.randint(0, 256, (8, 4, 9), dtype=torch.uint8),
    torch.ones(8, 4, dtype=torch.float8_e4m3fn),
    torch.ones(8, 1, dtype=torch.float32) * 0.05,
    block_size=16,
    group_size=16,
)
assert trellis.dequantize().shape == (8, 64)
assert trellis.clone().float().shape == (8, 64)
assert torch.isfinite(trellis.dequantize()).all()
assert not any(name.startswith("piper_kernels.linear") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    "weight_type",
    [
        ConvRotInt8Tensor,
        PiperNVFP4Tensor,
        ConvRotNVFP4Tensor,
        "convrot_trellis_uniform",
        "convrot_trellis_variable",
    ],
)
def test_mmap_checkpoint_and_parameter_assignment_preserve_storage(tmp_path, weight_type):
    if weight_type == "convrot_trellis_uniform":
        weight = _sample_trellis_weight(variable_rows=False)
        expected_type = ConvRotTrellisTensor
    elif weight_type == "convrot_trellis_variable":
        weight = _sample_trellis_weight(variable_rows=True)
        expected_type = ConvRotTrellisTensor
    else:
        source = torch.randn(8, 64)
        kwargs = {} if weight_type is PiperNVFP4Tensor else {"group_size": 16}
        if weight_type is not ConvRotInt8Tensor:
            kwargs.update(compute_per_tensor_scale=True, act_per_tensor_scale=torch.tensor(0.125))
        weight = weight_type.from_hp(source, **kwargs)
        expected_type = weight_type
    checkpoint = tmp_path / "weight.pt"
    torch.save({"weight": weight}, checkpoint)
    with torch.serialization.safe_globals([expected_type]):
        restored = torch.load(checkpoint, mmap=True, weights_only=True)["weight"]
    assert type(restored) is expected_type
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
        ("piper_kernels.linear.convrot", "ConvRotTrellisTensor"),
        ("piper_kernels.linear.convrot.int8", "ConvRotInt8Tensor"),
        ("piper_kernels.linear.convrot.int8", "ConvRotTrellisTensor"),
        ("piper_kernels.linear.convrot.nvfp4", "ConvRotNVFP4Tensor"),
        ("piper_kernels.linear.nvfp4", "PiperNVFP4Tensor"),
    ],
)
def test_tensor_exports_live_only_in_weight_packages(module, name):
    assert not hasattr(importlib.import_module(module), name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module + ".tensor")
