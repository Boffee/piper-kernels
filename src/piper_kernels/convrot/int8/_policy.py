"""Backend-independent policy for optimized INT8 ConvRot preparation."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

_FUSED_GROUP_SIZE = 256
_FUSED_MIN_ROWS = 512
_FUSED_MAX_BLOCK_SIZE = 16_384
_FUSED_DTYPES = (torch.float16, torch.bfloat16)


def is_sm120(device: torch.device) -> bool:
    """Return whether ``device`` is the exact architecture measured here."""
    return AcceleratorTarget.from_device(device).is_cuda_capability(12, 0)


def can_fuse_rotation_quantization(
    rows: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    sm120: bool | None = None,
) -> bool:
    """Return whether the measured one-row preparation kernel is eligible."""
    if in_features <= 0:
        return False
    block_size = max(128, 1 << (in_features - 1).bit_length())
    return (
        group_size == _FUSED_GROUP_SIZE
        and (is_sm120(device) if sm120 is None else sm120)
        and dtype in _FUSED_DTYPES
        and rows >= _FUSED_MIN_ROWS
        and block_size <= _FUSED_MAX_BLOCK_SIZE
    )
