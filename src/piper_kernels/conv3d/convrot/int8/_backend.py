"""Target selection for validated ConvRot INT8 convolution operations."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._amd import policy as amd_policy
from ._interfaces import ConvolutionBackend
from ._nvidia import policy as nvidia_policy

try:
    from ._nvidia import dispatch as _nvidia_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _nvidia_backend = None


try:
    from ._amd import dispatch as _amd_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _amd_backend = None


def select_backend(input: torch.Tensor) -> ConvolutionBackend | None:  # noqa: A002
    """Select a validated native implementation, or request the portable reference."""
    if _nvidia_backend is None and _amd_backend is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    if nvidia_policy.supports_target(target):
        return _nvidia_backend
    return _amd_backend if amd_policy.supports_target(target) else None
