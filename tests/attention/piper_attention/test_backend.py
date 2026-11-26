"""Dense Piper target selection and fake execution must use metadata only."""

import subprocess
import sys

import pytest
import torch
from _accelerators import forbid_device_probes, forbid_triton_driver
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels import piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _backend


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201", "gfx942", "gfx1100", "sm120", None])
@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_amd_target_gate(monkeypatch, architecture, platform):
    monkeypatch.setattr(sys, "platform", platform)
    target = AcceleratorTarget("hip", architecture)
    expected = platform in ("linux", "win32") and architecture in ("gfx1200", "gfx1201")
    assert _backend.supports_amd_target(target) is expected
    assert _backend.select_backend(target) is (_backend.amd_attention if expected else None)


@pytest.mark.parametrize("architecture", ["sm80", "sm89", "sm90", "sm100", "sm120", "sm121"])
def test_nvidia_selection_is_unchanged(architecture):
    target = AcceleratorTarget("cuda", architecture)
    expected = _backend.nvidia_attention if target.supports_uint8_int8_mma else None
    assert _backend.select_backend(target) is expected


@pytest.mark.parametrize("backend", ["cpu", "meta", "mps"])
def test_non_gpu_targets_retain_fallback(backend):
    assert _backend.select_backend(AcceleratorTarget(backend)) is None


def test_missing_implementation_retains_fallback(monkeypatch):
    monkeypatch.setattr(_backend, "amd_attention", None)
    monkeypatch.setattr(_backend, "nvidia_attention", None)
    assert _backend.select_backend(AcceleratorTarget("hip", "gfx1201")) is None
    assert _backend.select_backend(AcceleratorTarget("cuda", "sm120")) is None


@pytest.mark.parametrize(
    ("backend", "architecture", "expected"),
    [
        ("cuda", "sm75", None),
        ("cuda", "sm80", "nvidia"),
        ("cuda", "sm86", "nvidia"),
        ("cuda", "sm89", "nvidia"),
        ("cuda", "sm90", None),
        ("cuda", "sm100", None),
        ("cuda", "sm120", "nvidia"),
        ("hip", "gfx942", None),
        ("hip", "gfx1100", None),
        ("hip", "gfx1151", None),
        ("hip", "gfx1200", "amd"),
        ("hip", "gfx1201", "amd"),
        ("cpu", None, None),
    ],
)
def test_prefill_backend_target_gating(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    architecture: str | None,
    expected: str | None,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert _backend.prefill_backend(AcceleratorTarget(backend, architecture)) == expected


def test_fake_execution_does_not_select_or_launch_backend(monkeypatch):
    assert piper_attention is not None
    forbid_device_probes(monkeypatch, "fake execution must not inspect hardware")
    with forbid_triton_driver("fake execution must not launch kernels"), FakeTensorMode():
        query = torch.empty(2, 193, 6, 128, device="cuda:0", dtype=torch.bfloat16).transpose(1, 2)
        key = torch.empty(2, 2, 257, 128, device="cuda:0", dtype=query.dtype)
        output = torch.ops.piper_kernels.piper_attention.default(query, key, key, 128**-0.5, False)
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
