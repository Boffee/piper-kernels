"""Validated NVIDIA projection and chunked output integration targets."""

from piper_kernels._triton.targets import AcceleratorTarget


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native sparse Piper attention targets: exact SM120 and SM89."""
    return target.is_cuda_capability(12, 0) or is_sm89(target)


def is_sm89(target: AcceleratorTarget) -> bool:
    """SM89 has its own projection kernels; SM120 keeps the original configurations."""
    return target.is_cuda_capability(8, 9)
