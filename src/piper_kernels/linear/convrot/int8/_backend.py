"""Optional optimized backend selection for ConvRot INT8 operations."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

try:
    from . import triton as triton_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton_backend = None


def supports_triton(input: torch.Tensor) -> bool:  # noqa: A002
    """Return whether the optimized backend can execute on this device."""
    return triton_backend is not None and AcceleratorTarget.from_device(
        input.device
    ).cuda_capability_at_least(7, 5)


__all__ = ["supports_triton", "triton_backend"]
