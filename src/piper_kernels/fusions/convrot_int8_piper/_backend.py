"""Select validated target configurations for dense Piper query projection."""

from collections.abc import Callable
from functools import partial

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._amd import policy as amd_policy
from piper_kernels.fusions.convrot_int8_sage_qk import _validation as qk_validation
from piper_kernels.fusions.projected_qk import _validation as head_validation

try:
    from . import triton as _projection
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _projection = None

type ProjectionBackend = Callable[..., None]


def select_projection_backend(
    input: torch.Tensor,  # noqa: A002
    *,
    head_dim: int = 128,
) -> ProjectionBackend | None:
    """Return a projection launcher for exact SM120 or RDNA4 D64/D128."""
    if _projection is None or head_dim not in (64, 128):
        return None
    target = AcceleratorTarget.from_device(input.device)
    if target.is_cuda_capability(12, 0):
        config = _projection.ProjectionConfig(
            block_k=128,
            heads_per_program=2,
            num_warps=8,
            num_stages=3,
            round_rsqrt_to_nearest=True,
        )
    elif amd_policy.supports_target(target):
        config = _projection.ProjectionConfig(
            block_k=64,
            heads_per_program=1,
            num_warps=4,
            num_stages=2,
            group_m=8,
        )
    else:
        return None
    return partial(_projection.project_query, config=config)


def source_files() -> tuple[str, ...]:
    """Track target policy and all shared projection/quantization arithmetic."""
    paths = [__file__, amd_policy.__file__, qk_validation.__file__, head_validation.__file__]
    if _projection is not None:
        from piper_kernels.attention.kernels.qk_quantization.int8.sage import (  # noqa: PLC0415
            _rotation as rotation,
        )
        from piper_kernels.attention.kernels.qk_quantization.int8.sage import (  # noqa: PLC0415
            triton as quantization,
        )
        from piper_kernels.fusions.convrot_int8_sage_qk import (  # noqa: PLC0415
            triton as projection_qk,
        )
        from piper_kernels.fusions.projected_qk import triton as transforms  # noqa: PLC0415
        from piper_kernels.linear.convrot.int8._kernels import triton as matmul  # noqa: PLC0415

        paths.extend(
            (
                _projection.__file__,
                quantization.__file__,
                rotation.__file__,
                projection_qk.__file__,
                transforms.__file__,
                matmul.__file__,
            )
        )
    return tuple(path for path in paths if path is not None)
