"""Accelerator gates shared by GPU tests."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget


def exact_sm120_available() -> bool:
    """Return whether the default CUDA device is exact NVIDIA SM120.

    This matches the library's SM120 device checks, which exclude ROCm devices
    that report capability ``(12, 0)``.
    """
    return torch.cuda.is_available() and AcceleratorTarget.from_device(
        torch.device("cuda")
    ).is_cuda_capability(12, 0)
