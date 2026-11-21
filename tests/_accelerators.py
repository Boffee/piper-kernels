"""Accelerator gates shared by GPU tests."""

import sys

import torch

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.targets import AcceleratorTarget


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
    return torch.version.hip is not None and bool(amd_compiler_workarounds.LDS_ONLY_FENCE_ATTRS)
