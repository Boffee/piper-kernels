"""Portable rotation primitives shared by ConvRot storage formats."""

import math
from functools import cache

import torch

from ._torch_compat import is_fake_mode_active

SUPPORTED_GROUP_SIZES = (16, 64, 256)


def validate_group_size(group_size: int) -> None:
    """Validate a regular block-Hadamard group size supported by ConvRot."""
    if group_size not in SUPPORTED_GROUP_SIZES:
        supported = ", ".join(map(str, SUPPORTED_GROUP_SIZES))
        raise ValueError(f"ConvRot group size must be one of {supported}, got {group_size}")


def build_hadamard(
    size: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build ConvRot's normalized regular Hadamard matrix in its fixed H4 order."""
    validate_group_size(size)
    device = torch.device("cpu") if device is None else device
    if device.type == "meta" or torch.compiler.is_compiling() or is_fake_mode_active():
        return _build_hadamard(size, device, dtype)
    return _build_hadamard_cached(size, device, dtype)


@cache
def _build_hadamard_cached(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Cache concrete matrices without leaking them into fake/meta tracing."""
    return _build_hadamard(size, device, dtype)


def _build_hadamard(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
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


def rotate_groups(value: torch.Tensor, group_size: int) -> torch.Tensor:
    """Multiply groups along the final dimension by ConvRot's regular Hadamard."""
    validate_group_size(group_size)
    features = value.shape[-1]
    if features % group_size:
        raise ValueError(
            f"ConvRot feature dimension {features} is not divisible by group size {group_size}"
        )
    if features == 0:
        return value.clone()
    matrix = build_hadamard(group_size, value.device, value.dtype)
    grouped = value.reshape(-1, features // group_size, group_size)
    return torch.matmul(grouped, matrix).reshape(value.shape)
