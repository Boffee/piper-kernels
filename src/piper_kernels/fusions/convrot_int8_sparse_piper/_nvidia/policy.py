"""Validated NVIDIA projection and chunked output integration targets."""

from piper_kernels._triton.targets import AcceleratorTarget


def uses_async_copies(target: AcceleratorTarget) -> bool:
    """Select SM89's projections, whose D128 kernels stage operands with ``cp.async``."""
    return target.is_cuda_capability(8, 9)


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native sparse Piper attention targets: exact SM120 and SM89."""
    return target.is_cuda_capability(12, 0) or uses_async_copies(target)
