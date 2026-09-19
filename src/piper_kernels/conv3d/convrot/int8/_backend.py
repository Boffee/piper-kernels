"""Target selection for validated ConvRot INT8 convolution operations."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from . import _interfaces, _plan
from ._interfaces import ConvolutionBackend
from ._nvidia import policy as nvidia_policy

try:
    from . import triton as _shared
    from ._nvidia import triton as _nvidia_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _shared = None
    _nvidia_backend = None


def select_backend(input: torch.Tensor) -> ConvolutionBackend | None:  # noqa: A002
    """Select a validated native implementation, or request the portable reference."""
    if _nvidia_backend is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    return _nvidia_backend if nvidia_policy.supports_target(target) else None


def source_files() -> tuple[str, ...]:
    """Include backend selection, shared execution, and launch policy in compiler keys."""
    return tuple(
        path
        for path in (
            __file__,
            _interfaces.__file__,
            _plan.__file__,
            nvidia_policy.__file__,
            None if _shared is None else _shared.__file__,
            None if _nvidia_backend is None else _nvidia_backend.__file__,
        )
        if path is not None
    )
