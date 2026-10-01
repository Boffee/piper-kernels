"""Select target configurations for dense Piper Q/K/V projections."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._amd import policy as amd_policy
from piper_kernels.fusions.convrot_int8_projection import _validation as projection_validation
from piper_kernels.fusions.convrot_int8_projection._nvidia import _plan as nvidia_plan
from piper_kernels.fusions.convrot_int8_sage_qk import _validation as qk_validation
from piper_kernels.fusions.projected_qk import _validation as head_validation

from . import _interfaces
from ._amd import policy as amd_projection_policy
from ._interfaces import ProjectionBackend
from ._nvidia import policy as nvidia_projection_policy

try:
    from piper_kernels.fusions.convrot_int8_projection._nvidia import (
        fragments as _nvidia_fragments,
    )
    from piper_kernels.linear.convrot.int8._nvidia import gluon_async_copy as _linear_async_copy

    from . import triton as _projection
    from ._amd import triton as _amd_projection
    from ._nvidia import dispatch as _nvidia_projection
    from ._nvidia import gluon_async_copy as _nvidia_async_copy_gluon
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _projection = None
    _amd_projection = None
    _nvidia_projection = None
    _nvidia_async_copy_gluon = None
    _nvidia_fragments = None
    _linear_async_copy = None


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


def source_files() -> tuple[str, ...]:
    """Track target policy and all shared projection/quantization arithmetic."""
    paths: list[str | None] = [
        __file__,
        _interfaces.__file__,
        amd_policy.__file__,
        amd_projection_policy.__file__,
        nvidia_projection_policy.__file__,
        nvidia_plan.__file__,
        qk_validation.__file__,
        projection_validation.__file__,
        head_validation.__file__,
    ]
    if _projection is not None:
        from piper_kernels.attention.kernels.qk_quantization.int8.sage import (  # noqa: PLC0415
            _rotation as rotation,
        )
        from piper_kernels.attention.kernels.qk_quantization.int8.sage import (  # noqa: PLC0415
            triton as quantization,
        )
        from piper_kernels.fusions.convrot_int8_projection import (  # noqa: PLC0415
            triton as shared_projection,
        )
        from piper_kernels.fusions.convrot_int8_sage_qk import (  # noqa: PLC0415
            key as key_projection,
        )
        from piper_kernels.fusions.convrot_int8_sage_qk import (  # noqa: PLC0415
            triton as projection_qk,
        )
        from piper_kernels.fusions.projected_qk import triton as transforms  # noqa: PLC0415
        from piper_kernels.linear.convrot.int8 import _backend as linear_backend  # noqa: PLC0415
        from piper_kernels.linear.convrot.int8._generic import mean  # noqa: PLC0415
        from piper_kernels.linear.convrot.int8._kernels import triton as matmul  # noqa: PLC0415

        from . import _kernels  # noqa: PLC0415

        paths.extend(
            (
                _projection.__file__,
                _kernels.__file__,
                *key_projection.source_files(),
                _amd_projection.__file__ if _amd_projection is not None else None,
                _nvidia_projection.__file__ if _nvidia_projection is not None else None,
                _nvidia_async_copy_gluon.__file__ if _nvidia_async_copy_gluon is not None else None,
                _nvidia_fragments.__file__ if _nvidia_fragments is not None else None,
                _linear_async_copy.__file__ if _linear_async_copy is not None else None,
                shared_projection.__file__,
                linear_backend.__file__,
                mean.__file__,
                quantization.__file__,
                rotation.__file__,
                projection_qk.__file__,
                transforms.__file__,
                matmul.__file__,
            )
        )
    return tuple(path for path in paths if path is not None)
