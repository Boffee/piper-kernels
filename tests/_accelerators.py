"""Accelerator gates and public hardware-probe mocks shared by tests."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget

# The LLIR workaround hook in amd_compiler_workarounds registers only at import
# time on genuine ROCm builds, not when unit tests later stub torch.version.hip.
_ROCM_BUILD = torch.version.hip is not None


def exact_sm120_available() -> bool:
    """Return whether the default CUDA device is exact NVIDIA SM120.

    This matches the library's SM120 device checks, which exclude ROCm devices
    that report capability ``(12, 0)``.
    """
    return torch.cuda.is_available() and AcceleratorTarget.from_device(
        torch.device("cuda")
    ).is_cuda_capability(12, 0)


def rdna4_available() -> bool:
    """Return whether the default CUDA device is a supported Linux/Windows RDNA4 GPU."""
    if sys.platform not in ("linux", "win32") or not torch.cuda.is_available():
        return False
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    return target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201")


def rocm_fences_only_lds() -> bool:
    """Return whether opted-in RDNA kernels compile here with LDS-only barrier fences.

    The LLIR rewrite runs only on ROCm builds whose Triton fences global memory at
    LDS barriers; elsewhere opted-in kernels compile with Triton's own fences.
    """
    if not _ROCM_BUILD:
        return False
    from piper_kernels._triton import (  # noqa: PLC0415 - keep tests._accelerators importable without Triton
        amd_compiler_workarounds,
    )

    return bool(amd_compiler_workarounds.LDS_ONLY_FENCE_ATTRS)


def mock_cuda_target(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int] = (12, 0)
) -> None:
    """Stub PyTorch's CUDA capability probe and device guard for FakeTensor / CPU tests."""
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "device", lambda _device=None: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: capability)


def mock_rocm_target(monkeypatch: pytest.MonkeyPatch, architecture: str = "gfx1201") -> None:
    """Stub PyTorch's HIP properties probe and device guard for FakeTensor / CPU tests."""
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(torch.cuda, "device", lambda _device=None: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device=None: SimpleNamespace(gcnArchName=architecture),
    )


def mock_triton_driver(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
    arch: object,
    device_type: str = "cuda",
) -> None:
    """Stub Triton's active compiler driver and CUDA device guard without a real GPU."""
    import triton  # noqa: PLC0415 - keep tests._accelerators importable without Triton

    monkeypatch.setattr(torch.cuda, "device", lambda _device=None: nullcontext())
    driver = SimpleNamespace(
        get_active_torch_device=lambda: torch.device(device_type, 0),
        get_current_target=lambda: SimpleNamespace(backend=backend, arch=arch),
    )
    monkeypatch.setattr(triton.runtime, "driver", SimpleNamespace(active=driver))
