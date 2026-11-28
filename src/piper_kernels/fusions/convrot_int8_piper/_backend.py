"""Select target configurations for dense Piper Q/K/V projections."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._amd import policy as amd_policy

from ._interfaces import ProjectionBackend
from ._nvidia import policy as nvidia_projection_policy

try:
    from ._amd import gluon as _amd_projection
    from ._nvidia import gluon as _nvidia_projection
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _amd_projection = None
    _nvidia_projection = None


def select_projection_backend(
    input: torch.Tensor,  # noqa: A002
    *,
    head_dim: int = 128,
    target: AcceleratorTarget | None = None,
) -> ProjectionBackend | None:
    """Return typed Q/K/V operations for exact SM120, SM89, or RDNA4 D64/D128."""
    if (_nvidia_projection is None and _amd_projection is None) or head_dim not in (64, 128):
        return None
    resolved_target = AcceleratorTarget.from_device(input.device) if target is None else target
    if nvidia_projection_policy.supports_target(resolved_target):
        return _nvidia_projection
    return _amd_projection if amd_policy.supports_target(resolved_target) else None


def require_projection_backend(
    input: torch.Tensor,  # noqa: A002
    *,
    head_dim: int = 128,
    target: AcceleratorTarget | None = None,
) -> ProjectionBackend:
    """Require a native projection backend after validating execution inputs."""
    backend = select_projection_backend(input, head_dim=head_dim, target=target)
    if backend is None:
        raise ValueError(f"ConvRot INT8 dense projections are unavailable on {input.device}")
    return backend
