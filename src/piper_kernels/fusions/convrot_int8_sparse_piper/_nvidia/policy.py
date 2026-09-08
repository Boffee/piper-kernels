"""Validated NVIDIA projection and chunked output integration targets."""

from piper_kernels._triton.targets import AcceleratorTarget


def supports_target(target: AcceleratorTarget) -> bool:
    return target.is_cuda_capability(12, 0)
