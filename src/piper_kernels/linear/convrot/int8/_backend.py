"""Operation-specific selection shared by eager dispatch and custom-op execution."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from . import _generic
from ._amd import policy as amd_policy
from ._interfaces import Add, Addmm, DequantizedMean, GGUFConvert, LinearBackend, PreparationBackend
from ._nvidia import policy as nvidia_policy

try:
    from ._generic import triton as _generic_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _generic_backend = None

try:
    from ._nvidia import triton as _nvidia_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _nvidia_backend = None

try:
    from ._amd import triton as _amd_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _amd_backend = None


def select_linear_backend(input: torch.Tensor) -> LinearBackend | None:  # noqa: A002
    """Select compatible linear/preparation/projection operations, or a portable fallback."""
    if _nvidia_backend is None and _amd_backend is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    if nvidia_policy.supports_target(target):
        return _nvidia_backend
    return _amd_backend if amd_policy.supports_target(target) else None


def require_linear_backend(input: torch.Tensor) -> LinearBackend:  # noqa: A002
    """Resolve an optimized op emitted by the compiler or called explicitly."""
    backend = select_linear_backend(input)
    if backend is None:
        raise ValueError(f"ConvRot INT8 optimized linear is unavailable on {input.device}")
    return backend


def select_add(input: torch.Tensor) -> Add | None:  # noqa: A002
    """Use shared accelerator updates; CPU keeps its directly traced reference."""
    return _generic.add_ if input.device.type not in ("cpu", "meta") else None


def select_addmm(input: torch.Tensor) -> Addmm | None:  # noqa: A002
    """Do not require an INT8 matrix policy for a floating-point update product."""
    return _generic.addmm_ if input.device.type not in ("cpu", "meta") else None


def select_preparation_backend(input: torch.Tensor) -> PreparationBackend:  # noqa: A002
    """Prefer tuned preparation, with generic execution on other devices."""
    return select_linear_backend(input) or _generic


def select_gguf_converter(input: torch.Tensor) -> GGUFConvert | None:  # noqa: A002
    """Select direct GGUF conversion without requiring INT8 matrix instructions."""
    if _generic_backend is not None and _generic_backend.supports_device(input.device):
        return _generic_backend.convert_gguf_out
    return None


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
