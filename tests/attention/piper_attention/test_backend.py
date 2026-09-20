"""Dense Piper target selection and fake execution must use metadata only."""

import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _backend
from piper_kernels.attention.piper_attention.triton import triton_piper_attention


@pytest.mark.parametrize("architecture", ["sm80", "sm89", "sm90", "sm100", "sm120", "sm121"])
def test_nvidia_selection_is_unchanged(architecture):
    target = AcceleratorTarget("cuda", architecture)
    expected = _backend.nvidia_attention if target.supports_uint8_int8_mma else None
    assert _backend.select_backend(target) is expected


@pytest.mark.parametrize("backend", ["cpu", "meta", "mps"])
def test_non_gpu_targets_retain_fallback(backend):
    assert _backend.select_backend(AcceleratorTarget(backend)) is None


def test_missing_implementation_retains_fallback(monkeypatch):
    monkeypatch.setattr(_backend, "nvidia_attention", None)
    assert _backend.select_backend(AcceleratorTarget("cuda", "sm120")) is None


def test_fake_execution_does_not_select_or_launch_backend(monkeypatch):
    def forbidden(*args):
        pytest.fail("fake execution must not inspect hardware or launch kernels")

    monkeypatch.setattr(_backend, "select_backend", forbidden)
    with FakeTensorMode():
        query = torch.empty(2, 193, 6, 128, dtype=torch.bfloat16).transpose(1, 2)
        key = torch.empty(2, 2, 257, 128, dtype=query.dtype)
        output = triton_piper_attention(query, key, key, 128**-0.5, False)
        assert output.shape == query.shape
        assert output.dtype == query.dtype
        assert output.is_contiguous()


def test_dense_cpu_fallback_remains_usable_without_triton():
    script = """
import importlib.abc
import sys
import torch

class WithoutTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("Triton intentionally unavailable", name="triton")

sys.meta_path.insert(0, WithoutTriton())
from piper_kernels import piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _backend
query = torch.zeros(1, 3, 5, 64, dtype=torch.float16)
key = torch.zeros(1, 1, 7, 64, dtype=query.dtype)
assert torch.equal(piper_attention(query, key, key), query)
assert _backend.select_backend(AcceleratorTarget("cuda", "sm120")) is None
assert _backend.select_backend(AcceleratorTarget("hip", "gfx1201")) is None
assert not any(name == "triton" or name.startswith("triton.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
