"""Accelerator gates and public hardware-probe mocks shared by tests."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

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


def sm8x_available() -> bool:
    """Return whether the default CUDA device is an NVIDIA SM8x GPU."""
    return torch.cuda.is_available() and AcceleratorTarget.from_device(
        torch.device("cuda")
    ).is_cuda_capability(8)


def _resolve_cuda_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        return torch.device("cuda")
    return torch.device("cuda", device) if isinstance(device, int) else torch.device(device)


def rdna4_available(device: torch.device | str | int | None = None) -> bool:
    """Return whether ``device`` (default CUDA device) is a supported Linux/Windows RDNA4 GPU."""
    if sys.platform not in ("linux", "win32") or not torch.cuda.is_available():
        return False
    return AcceleratorTarget.from_device(_resolve_cuda_device(device)).is_rdna4


def convrot_int8_w8a8_available(device: torch.device | str | int | None = None) -> bool:
    """Return whether ``device`` (default CUDA) supports an optimized ConvRot INT8 W8A8 backend."""
    if not torch.cuda.is_available():
        return False
    target = AcceleratorTarget.from_device(_resolve_cuda_device(device))
    return (target.is_nvidia_cuda and target.cuda_capability_at_least(7, 5)) or (
        sys.platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx942", "gfx1100", "gfx1151", "gfx1200", "gfx1201")
    )


def exact_gfx1201_available() -> bool:
    """Return whether the default GPU is a supported Linux/Windows ROCm gfx1201 device."""
    if sys.platform not in ("linux", "win32") or not torch.cuda.is_available():
        return False
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    return target.is_amd_hip and target.is_architecture("gfx1201")


def blackwell_cuda_available() -> bool:
    """Return whether the default CUDA device is an NVIDIA Blackwell (SM10x+) GPU."""
    return torch.cuda.is_available() and AcceleratorTarget.from_device(
        torch.device("cuda")
    ).cuda_capability_at_least(10)


def w4a16_nvfp4_available() -> bool:
    """Return whether the default GPU supports weight-only W4A16 NVFP4 execution."""
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).supports_nvfp4_w4a16
    )


def trellis_encoder_available() -> bool:
    """Return whether the default GPU supports GPU Trellis encoding (NVIDIA SM80+, >= 96 KiB)."""
    if not torch.cuda.is_available():
        return False
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not target.is_nvidia_cuda or not target.cuda_capability_at_least(8, 0):
        return False
    return (
        torch.cuda.get_device_properties(torch.device("cuda")).shared_memory_per_block_optin
        >= 96 * 1024 + 128
    )


def uint8_int8_mma_available() -> bool:
    """Return whether the default CUDA device supports NVIDIA UINT8-by-INT8 MMAv2."""
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).supports_uint8_int8_mma
    )


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


def forbid_device_probes(
    monkeypatch: pytest.MonkeyPatch, message: str = "unexpected hardware probe"
) -> Mock:
    """Fail if code probes PyTorch's CUDA/ROCm device capability or properties."""
    probe = Mock(side_effect=AssertionError(message))
    monkeypatch.setattr(torch.cuda, "get_device_capability", probe)
    monkeypatch.setattr(torch.cuda, "get_device_properties", probe)
    return probe


@contextmanager
def forbid_triton_driver(message: str = "unexpected Triton driver access") -> Iterator[Mock]:
    """Fail if code accesses Triton's active driver, including cached JIT references."""
    guard = Mock(side_effect=AssertionError(message))
    try:
        import triton  # noqa: PLC0415 - keep tests._accelerators importable without Triton
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
        yield guard
        return

    driver = triton.runtime.driver
    previous = getattr(driver, "_active", None)
    driver.set_active(
        SimpleNamespace(
            get_current_device=guard,
            get_active_torch_device=guard,
            get_current_target=guard,
        )
    )
    try:
        yield guard
    finally:
        driver.set_active(previous)


def mock_triton_driver(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
    arch: object,
    device_type: str = "cuda",
    device_index: int = 0,
) -> None:
    """Stub Triton's active compiler driver and CUDA device guard without a real GPU."""
    import triton  # noqa: PLC0415 - keep tests._accelerators importable without Triton

    monkeypatch.setattr(torch.cuda, "device", lambda _device=None: nullcontext())
    driver = SimpleNamespace(
        get_active_torch_device=lambda: torch.device(device_type, device_index),
        get_current_target=lambda: SimpleNamespace(backend=backend, arch=arch),
    )
    monkeypatch.setattr(triton.runtime, "driver", SimpleNamespace(active=driver))
