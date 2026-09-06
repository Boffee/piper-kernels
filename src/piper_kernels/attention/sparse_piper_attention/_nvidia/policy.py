"""Preserve the validated SM120 support boundary for sparse Piper."""

from piper_kernels._triton.targets import AcceleratorTarget


def supports_target(target: AcceleratorTarget) -> bool:
    """Enable only exact NVIDIA SM120; HIP capability tuples are not CUDA."""
    return target.is_cuda_capability(12, 0)
