"""Metadata-only compiler guards shared by projected Q/K transformations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels.linear import _preparation_sharing as preparation_sharing


def static_int(value: object) -> int | None:
    """Return a non-boolean static integer."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def integer_scalar_metadata(value: object) -> int | torch.SymInt | None:
    """Resolve static or symbolic integer metadata from an FX argument."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, torch.SymInt)):
        return value
    if isinstance(value, torch.fx.Node):
        metadata = cast(torch.fx.Node, value).meta.get("val")
        if isinstance(metadata, (int, torch.SymInt)) and not isinstance(metadata, bool):
            return metadata
    return None


def integer_scalar_argument(value: object) -> Argument | None:
    """Return an FX-compatible integer argument when its metadata is valid."""
    if integer_scalar_metadata(value) is None:
        return None
    return value if isinstance(value, (int, torch.SymInt, torch.fx.Node)) else None


def positive_float(value: object) -> float | None:
    """Return a finite positive Python configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted > 0 else None


def valid_rmsnorm(
    norm_weight: object,
    epsilon: object,
    *,
    head_dim: int,
    device: torch.device,
    supported_dtypes: Sequence[torch.dtype],
) -> bool:
    """Check an optional affine weight and the Python RMSNorm epsilon."""
    if norm_weight is not None:
        if not isinstance(norm_weight, torch.fx.Node):
            return False
        norm = preparation_sharing.tensor_metadata(norm_weight)
        if (
            norm is None
            or norm.layout is not torch.strided
            or not norm.is_contiguous()
            or norm.dtype not in supported_dtypes
            or tuple(norm.shape) != (head_dim,)
            or norm.device != device
        ):
            return False
    return positive_float(epsilon) is not None


def valid_rope_tables(
    cos_node: object,
    sin_node: object,
    rotary_dim_value: object,
    half_rotary_dim_value: object,
    *,
    sequence_length: int | torch.SymInt,
    head_dim: int,
    device: torch.device,
) -> bool:
    """Check contiguous FP32 split-half tables without reading tensor contents.

    Keep symbolic dimension comparisons structural: symbolic widths are checked
    against their captured expressions, while static widths enforce the range
    and parity requirements.
    """
    if not isinstance(cos_node, torch.fx.Node) or not isinstance(sin_node, torch.fx.Node):
        return False
    rotary_dim = integer_scalar_metadata(rotary_dim_value)
    half_rotary_dim = integer_scalar_metadata(half_rotary_dim_value)
    if rotary_dim is None or half_rotary_dim is None:
        return False
    for node in (cos_node, sin_node):
        table = preparation_sharing.tensor_metadata(node)
        if (
            table is None
            or table.layout is not torch.strided
            or not table.is_contiguous()
            or table.dtype is not torch.float32
            or table.ndim != 2
            or table.device != device
            or preparation_sharing.dimension_key(table.shape[0])
            != preparation_sharing.dimension_key(sequence_length)
            or preparation_sharing.dimension_key(table.shape[1])
            != preparation_sharing.dimension_key(rotary_dim)
        ):
            return False
    if preparation_sharing.dimension_key(half_rotary_dim) != preparation_sharing.dimension_key(
        (rotary_dim + 1) // 2
    ):
        return False
    if isinstance(rotary_dim, int):
        return (
            2 <= rotary_dim <= head_dim
            and rotary_dim % 2 == 0
            and isinstance(half_rotary_dim, int)
            and half_rotary_dim == rotary_dim // 2
        )
    return True


__all__ = [
    "integer_scalar_argument",
    "integer_scalar_metadata",
    "positive_float",
    "static_int",
    "valid_rmsnorm",
    "valid_rope_tables",
]
