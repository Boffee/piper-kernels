"""Accelerator gates and public hardware-probe mocks shared by tests."""

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

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
    return AcceleratorTarget.from_device(_resolve_cuda_device(device)).is_hip_architecture(
        "gfx1200", "gfx1201"
    )


def convrot_int8_w8a8_available(device: torch.device | str | int | None = None) -> bool:
    """Return whether ``device`` (default CUDA) supports an optimized ConvRot INT8 W8A8 backend."""
    if not torch.cuda.is_available():
        return False
    target = AcceleratorTarget.from_device(_resolve_cuda_device(device))
    return (target.is_nvidia_cuda and target.cuda_capability_at_least(7, 5)) or (
        sys.platform in ("linux", "win32")
        and target.is_hip_architecture("gfx942", "gfx1100", "gfx1151", "gfx1200", "gfx1201")
    )


def exact_gfx1201_available() -> bool:
    """Return whether the default GPU is a supported Linux/Windows ROCm gfx1201 device."""
    if sys.platform not in ("linux", "win32") or not torch.cuda.is_available():
        return False
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    return target.is_hip_architecture("gfx1201")


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


def trellis_moe_available() -> bool:
    """Return whether the default GPU supports ConvRot Trellis MoE execution."""
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).supports_trellis_moe
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


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def sm89_available() -> bool:
    """Return whether the default CUDA device is an NVIDIA SM89 GPU."""
    return torch.cuda.is_available() and AcceleratorTarget.from_device(
        torch.device("cuda")
    ).is_cuda_capability(8, 9)


def nvidia_cuda_available() -> bool:
    """Return whether the default CUDA device is an NVIDIA CUDA GPU."""
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).is_nvidia_cuda
    )


def sparse_piper_native_available(head_dim: int = 128) -> bool:
    """Return whether the default device supports native sparse Piper attention kernels."""
    return _triton_available() and (
        exact_sm120_available() or sm8x_available() or (rdna4_available() and head_dim in (64, 128))
    )


def convrot_int8_sparse_projection_available(head_dim: int = 128) -> bool:
    """Return whether the default device supports fused ConvRot INT8 sparse Q/K/V projections."""
    return sparse_piper_native_available(head_dim)


def convrot_int8_dense_projection_available(head_dim: int = 128) -> bool:
    """Return whether the default device supports fused ConvRot INT8 dense Q/K/V projections."""
    return (
        _triton_available()
        and head_dim in (64, 128)
        and (exact_sm120_available() or sm89_available() or rdna4_available())
    )


def convrot_int8_sparse_output_available(head_dim: int = 128) -> bool:
    """Return whether the default device supports fused ConvRot INT8 sparse attention output."""
    return convrot_int8_sparse_projection_available(head_dim) and convrot_int8_w8a8_available()


def ensure_convrot_int8_linear_ops() -> None:
    """Register ConvRot INT8 linear custom ops via public ``F.linear`` if not yet loaded."""
    if hasattr(torch.ops.piper_kernels, "convrot_int8_linear"):
        return
    from piper_kernels.weights.convrot.int8 import (  # noqa: PLC0415 - keep import lazy
        ConvRotInt8Tensor,
    )

    weight = ConvRotInt8Tensor.from_quantized(
        torch.empty(32, 64, dtype=torch.int8, device="meta"),
        torch.empty(32, 1, dtype=torch.float32, device="meta"),
        group_size=64,
        logical_dtype=torch.bfloat16,
    )
    with torch.no_grad():
        F.linear(torch.empty(0, 64, dtype=torch.bfloat16, device="meta"), weight)


def ensure_nvfp4_linear_ops() -> None:
    """Register NVFP4 linear custom ops via public ``F.linear`` without ConvRot registration."""
    if hasattr(torch.ops.piper_kernels, "nvfp4_linear"):
        return
    from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor  # noqa: PLC0415 - keep import lazy

    weight = PiperNVFP4Tensor.from_quantized(
        torch.empty(128, 32, dtype=torch.uint8, device="meta"),
        torch.empty(32, 16, dtype=torch.float8_e4m3fn, device="meta"),
        logical_dtype=torch.bfloat16,
        is_swizzled_scales=True,
        act_quant="dynamic",
    )
    with torch.no_grad():
        F.linear(torch.empty(0, 64, dtype=torch.bfloat16, device="meta"), weight)


def ensure_convrot_nvfp4_linear_ops() -> None:
    """Register ConvRot NVFP4 linear custom ops via public ``F.linear`` if not yet loaded."""
    if hasattr(torch.ops.piper_kernels, "convrot_nvfp4_linear"):
        return
    from piper_kernels.weights.convrot.nvfp4 import (  # noqa: PLC0415 - keep import lazy
        ConvRotNVFP4Tensor,
    )

    weight = ConvRotNVFP4Tensor.from_quantized(
        torch.empty(128, 32, dtype=torch.uint8, device="meta"),
        torch.empty(32, 16, dtype=torch.float8_e4m3fn, device="meta"),
        group_size=16,
        logical_dtype=torch.bfloat16,
        is_swizzled_scales=True,
        act_quant="dynamic",
    )
    with torch.no_grad():
        F.linear(torch.empty(0, 64, dtype=torch.bfloat16, device="meta"), weight)


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
