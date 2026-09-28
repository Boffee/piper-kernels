"""Select target configurations for dense Piper Q/K/V projections."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._amd import policy as amd_policy

from ._interfaces import ProjectionBackend
from ._nvidia import policy as nvidia_projection_policy

try:
    from . import triton as _projection
    from ._amd import triton as _amd_projection
    from ._nvidia import dispatch as _nvidia_projection
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _projection = None
    _amd_projection = None
    _nvidia_projection = None


def select_projection_backend(
    input: torch.Tensor,  # noqa: A002
    *,
    head_dim: int = 128,
) -> ProjectionBackend | None:
    """Return typed Q/K/V operations for exact SM120, SM89, or RDNA4 D64/D128."""
    if _projection is None or head_dim not in (64, 128):
        return None
    target = AcceleratorTarget.from_device(input.device)
    if nvidia_projection_policy.supports_target(target):
        return _nvidia_projection
    return _amd_projection if amd_policy.supports_target(target) else None


def require_projection_backend(
    input: torch.Tensor,  # noqa: A002
    *,
    head_dim: int = 128,
) -> ProjectionBackend:
    """Require a native projection backend after validating execution inputs."""
    backend = select_projection_backend(input, head_dim=head_dim)
    if backend is None:
        raise ValueError(f"ConvRot INT8 dense projections are unavailable on {input.device}")
    return backend
