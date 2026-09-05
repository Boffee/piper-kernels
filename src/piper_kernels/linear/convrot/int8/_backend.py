"""Operation-specific selection shared by eager dispatch and custom-op execution."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._interfaces import Add, Addmm, DequantizedMean, GGUFConvert, LinearBackend
from ._nvidia import policy as nvidia_policy

try:
    from ._nvidia import triton as _nvidia_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _nvidia_backend = None


def select_linear_backend(input: torch.Tensor) -> LinearBackend | None:  # noqa: A002
    """Select compatible linear/preparation/projection operations, or a portable fallback."""
    if _nvidia_backend is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    return _nvidia_backend if nvidia_policy.supports_target(target) else None


def require_linear_backend(input: torch.Tensor) -> LinearBackend:  # noqa: A002
    """Resolve an optimized op emitted by the compiler or called explicitly."""
    backend = select_linear_backend(input)
    if backend is None:
        raise ValueError(f"ConvRot INT8 optimized linear is unavailable on {input.device}")
    return backend


def select_add(input: torch.Tensor) -> Add | None:  # noqa: A002
    """Select dense weight updates independently of linear execution."""
    if _nvidia_backend is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    return _nvidia_backend.add_ if nvidia_policy.supports_target(target) else None


def select_addmm(input: torch.Tensor) -> Addmm | None:  # noqa: A002
    """Select low-rank weight updates independently of linear execution."""
    if _nvidia_backend is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    return _nvidia_backend.addmm_ if nvidia_policy.supports_target(target) else None


def select_gguf_converter(input: torch.Tensor) -> GGUFConvert | None:  # noqa: A002
    """Select direct GGUF conversion without requiring INT8 matrix instructions."""
    if _nvidia_backend is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    return (
        _nvidia_backend._convert_gguf_out
        if nvidia_policy.supports_preparation_target(target)
        else None
    )


def select_dequantized_mean(input: torch.Tensor) -> DequantizedMean | None:  # noqa: A002
    """Select the prepared-input mean independently of projection support."""
    if _nvidia_backend is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    return (
        _nvidia_backend.dequantized_input_mean
        if nvidia_policy.supports_preparation_target(target)
        else None
    )
