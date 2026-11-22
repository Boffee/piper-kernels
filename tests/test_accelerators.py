"""Tests for shared accelerator gates and hardware-probe test helpers."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _accelerators import (
    exact_sm120_available,
    forbid_device_probes,
    forbid_triton_driver,
    mock_cuda_target,
    mock_rocm_target,
    mock_triton_driver,
    rocm_fences_only_lds,
)


@pytest.mark.parametrize("hip", [None, "test"])
def test_sm120_gate_distinguishes_rocm_from_cuda(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1201"),
    )
    assert exact_sm120_available() is (hip is None)


def test_target_mocks_accept_zero_or_one_device_args_and_stub_device_guard(monkeypatch):
    mock_cuda_target(monkeypatch, (8, 9))
    assert torch.version.hip is None
    assert torch.cuda.get_device_capability() == (8, 9)
    assert torch.cuda.get_device_capability(torch.device("cuda:0")) == (8, 9)
    with torch.cuda.device(torch.device("cuda:0")):
        pass

    mock_rocm_target(monkeypatch, "gfx1201")
    assert torch.version.hip == "7.0"
    assert torch.cuda.get_device_properties().gcnArchName == "gfx1201"
    assert torch.cuda.get_device_properties(torch.device("cuda:0")).gcnArchName == "gfx1201"
    with torch.cuda.device(torch.device("cuda:0")):
        pass


def test_mock_rocm_target_does_not_enable_rocm_only_fence_assertion_on_cuda(monkeypatch):
    on_rocm_build = torch.version.hip is not None
    mock_rocm_target(monkeypatch, "gfx1201")
    if not on_rocm_build:
        assert not rocm_fences_only_lds()


def test_forbid_device_probes_and_triton_driver_mock(monkeypatch):
    probe = forbid_device_probes(monkeypatch, "no probe")
    with pytest.raises(AssertionError, match="no probe"):
        torch.cuda.get_device_capability(torch.device("cuda:0"))
    with pytest.raises(AssertionError, match="no probe"):
        torch.cuda.get_device_properties(torch.device("cuda:0"))
    assert probe.call_count == 2

    triton = pytest.importorskip("triton")
    import triton.runtime.jit as triton_jit  # noqa: PLC0415

    with forbid_triton_driver("no triton") as guard:
        with pytest.raises(AssertionError, match="no triton"):
            triton_jit.driver.active.get_current_device()
        assert guard.call_count == 1

    mock_triton_driver(monkeypatch, backend="cuda", arch=120, device_index=1)
    assert triton.runtime.driver.active.get_active_torch_device() == torch.device("cuda:1")
    target = triton.runtime.driver.active.get_current_target()
    assert (target.backend, target.arch) == ("cuda", 120)
    with torch.cuda.device(torch.device("cuda:1")):
        pass


def test_accelerators_helper_is_importable_without_triton() -> None:
    tests_dir = Path(__file__).resolve().parent
    script = f"""
import builtins, sys
orig_import = builtins.__import__
def without_triton(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        raise ModuleNotFoundError("Triton intentionally unavailable", name="triton")
    return orig_import(name, *args, **kwargs)
builtins.__import__ = without_triton
sys.path.insert(0, {str(tests_dir)!r})
import _accelerators
assert "triton" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
