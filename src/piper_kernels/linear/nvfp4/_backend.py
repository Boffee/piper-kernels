"""Optional optimized backend selection for NVFP4 weight updates."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

try:
    from . import triton as triton_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton_backend = None


def supports_triton(input: torch.Tensor) -> bool:  # noqa: A002
    """Return whether the fused NVFP4 update backend can execute on this device."""
    return triton_backend is not None and AcceleratorTarget.from_device(
        input.device
    ).cuda_capability_at_least(10)


__all__ = ["supports_triton", "triton_backend"]
