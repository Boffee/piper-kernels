"""The portable NVFP4 reference must run independently of custom operators."""

import subprocess
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from piper_kernels.linear.nvfp4 import reference


def test_reference_and_tensor_imports_do_not_load_nvfp4_backends():
    script = """
import sys
import torch
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor, reference
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor

for group_size in (0, 16):
    prepared = reference.prepare_input(torch.ones(2, 32), None, True, group_size=group_size)
    output = reference.linear(torch.zeros(3, 32), *prepared, None, None, True,
                              group_size=group_size)
    assert torch.equal(output, torch.zeros(3, 2))
for package in ("piper_kernels.linear.nvfp4", "piper_kernels.linear.convrot.nvfp4"):
    for module in ("_ops", "_projection", "triton", "_compile"):
        assert f"{package}.{module}" not in sys.modules
from piper_kernels.linear.convrot.nvfp4 import prepare_dynamic, triton
assert prepare_dynamic is triton.prepare_dynamic
"""
    subprocess.run([sys.executable, "-c", script], check=True)


class _OnlyPyTorch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func.namespace in ("aten", "prims"), f"reference called custom operator {func}"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("group_size", [0, 16])
@pytest.mark.parametrize("high_first", [False, True])
def test_reference_linear_uses_only_pytorch(dtype, dynamic, group_size, high_first):
    torch.manual_seed(732)
    input = torch.randn(2, 3, 32, dtype=dtype).transpose(0, 1)  # noqa: A001
    dense_weight = torch.randn(16, 32, dtype=dtype)
    activation_scale = None if dynamic else torch.tensor(1 / 448)
    bias = torch.randn(16, dtype=torch.float32)

    with _OnlyPyTorch():
        weight = reference.prepare_input(
            dense_weight,
            None,
            True,
            high_first=high_first,
            group_size=group_size,
        )
        actual = reference.linear(
            input,
            *weight,
            activation_scale,
            bias,
            dynamic,
            high_first,
            group_size=group_size,
        )

    assert actual.shape == (3, 2, 16)
    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()


def test_reference_preserves_fp32_bias_until_output_cast():
    # The represented dot product is 255.5, halfway between two BF16 values.
    qdata = torch.full((1, 128), 0x22, dtype=torch.uint8)
    qdata[0, 0] = 0x12
    weight = torch.full((1, 128), 0x22, dtype=torch.uint8)
    scale = torch.ones((128, 16)).to(torch.float8_e4m3fn)
    bias = torch.tensor([-1.0009765625])
    with _OnlyPyTorch():
        actual = reference.linear_prepared(
            qdata,
            scale,
            torch.tensor(1 / 256),
            weight,
            scale,
            None,
            bias,
            torch.bfloat16,
        )
    assert actual.item() == -0.0029296875


def test_reference_zero_dynamic_input_returns_bias():
    input = torch.zeros((3, 32), dtype=torch.bfloat16)  # noqa: A001
    bias = torch.tensor([1.003, -0.1], dtype=torch.float32)
    with _OnlyPyTorch():
        weight = reference.prepare_input(torch.ones((2, 32)), None, True)
        actual = reference.linear(input, *weight, None, bias, True)
    assert torch.equal(actual, bias.to(input.dtype).expand(3, -1))
