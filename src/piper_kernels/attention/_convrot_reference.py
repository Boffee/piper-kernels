"""Portable ConvRot primitive used by experimental attention references."""

import math
from functools import cache

import torch

_SUPPORTED_GROUP_SIZES = (16, 64, 256)


@cache
def _hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size not in _SUPPORTED_GROUP_SIZES:
        supported = ", ".join(map(str, _SUPPORTED_GROUP_SIZES))
        raise ValueError(f"rotation group must be one of {supported}, got {size}")
    h4 = torch.tensor(
        ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1)),
        device=device,
        dtype=dtype,
    )
    result = h4
    current_size = 4
    while current_size < size:
        result = torch.kron(result, h4)
        current_size *= 4
    return result / math.sqrt(size)


def rotate_attention_groups(value: torch.Tensor, group_size: int) -> torch.Tensor:
    """Multiply groups along the final dimension by ConvRot's regular Hadamard."""
    features = value.shape[-1]
    if features % group_size:
        raise ValueError(
            f"head dimension {features} must be divisible by rotation group {group_size}"
        )
    matrix = _hadamard(group_size, value.device, value.dtype)
    grouped = value.reshape(-1, features // group_size, group_size)
    return torch.matmul(grouped, matrix).reshape(value.shape)
