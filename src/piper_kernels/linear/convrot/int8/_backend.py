"""Operation-specific selection shared by eager dispatch and custom-op execution."""

import torch

from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget

from . import _generic
from ._amd import policy as amd_policy
from ._interfaces import DequantizedMean, LinearBackend, PreparationBackend
from ._nvidia import policy as nvidia_policy

try:
    from ._generic import mean as _mean_backend
    from ._generic import triton as _generic_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _generic_backend = None
    _mean_backend = None

try:
    from ._nvidia import dispatch as _nvidia_backend
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


def select_preparation_backend(input: torch.Tensor) -> PreparationBackend:  # noqa: A002
    """Prefer tuned preparation, with generic execution on other devices."""
    return select_linear_backend(input) or _generic


def select_dequantized_mean(input: torch.Tensor) -> DequantizedMean | None:  # noqa: A002
    """Use the shared reduction independently of optimized INT8 matrix support."""
    if _mean_backend is not None and runtime.supports_device(input.device):
        return _mean_backend.dequantized_input_mean
    return None


def source_files() -> tuple[str, ...]:
    """Track prepared linear contracts, dispatch, and accelerator implementations."""
    from piper_kernels import _residual  # noqa: PLC0415

    from . import _interfaces, _plan  # noqa: PLC0415
    from ._amd import _plan as amd_plan  # noqa: PLC0415
    from ._nvidia import _plan as nvidia_plan  # noqa: PLC0415

    modules = [
        _residual,
        _interfaces,
        _plan,
        amd_plan,
        nvidia_plan,
        amd_policy,
        nvidia_policy,
        _nvidia_backend,
        _amd_backend,
    ]
    if _nvidia_backend is not None or _amd_backend is not None:
        from piper_kernels._residual import triton as residual_kernels  # noqa: PLC0415

        from ._kernels import triton as matmul  # noqa: PLC0415

        modules.extend((residual_kernels, matmul))
    if _nvidia_backend is not None:
        modules.extend((_nvidia_backend.triton_kernels, _nvidia_backend.gluon_async_copy))
    return (
        __file__,
        *(
            module.__file__
            for module in modules
            if module is not None and module.__file__ is not None
        ),
    )
