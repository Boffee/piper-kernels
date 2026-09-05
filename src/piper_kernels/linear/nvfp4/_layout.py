"""Canonical layout and allocation helpers for hardware-ready NVFP4 storage."""

from __future__ import annotations

from typing import Any, cast

import torch

BLOCK_SIZE = 16
QDATA_BLOCK_SIZE = BLOCK_SIZE // 2
SCALE_ROW_TILE = 128
SCALE_COLUMN_TILE = 64


def swap_packed_pairs(qdata: torch.Tensor) -> torch.Tensor:
    """Exchange the two E2M1 values stored in every packed byte."""
    return ((qdata & 0x0F) << 4) | (qdata >> 4)


def qdata_shape(
    rows: int | torch.SymInt,
    features: int | torch.SymInt,
) -> tuple[int | torch.SymInt, int | torch.SymInt]:
    """Return the UINT8 shape holding two logical FP4 values per byte."""
    features_value = cast(Any, features)
    return cast(
        tuple[int | torch.SymInt, int | torch.SymInt],
        (rows, features_value // 2),
    )


def scale_shape(
    rows: int | torch.SymInt,
    features: int | torch.SymInt,
) -> tuple[int | torch.SymInt, int | torch.SymInt]:
    """Return the physical shape of the swizzled FP8 block scales."""
    rows_value = cast(Any, rows)
    features_value = cast(Any, features)
    return cast(
        tuple[int | torch.SymInt, int | torch.SymInt],
        (
            (rows_value + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * 32,
            (features_value + SCALE_COLUMN_TILE - 1) // SCALE_COLUMN_TILE * 16,
        ),
    )


def has_scale_padding(rows: int, features: int) -> bool:
    """Return whether unused physical scale lanes must be initialized."""
    return rows % SCALE_ROW_TILE != 0 or features % SCALE_COLUMN_TILE != 0


def prepare_activation_storage(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    rows: int,
    features: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate or view caller-owned canonical activation storage."""
    qdata_dimensions = cast(tuple[int, int], qdata_shape(rows, features))
    scale_dimensions = cast(tuple[int, int], scale_shape(rows, features))
    scale_elements = scale_dimensions[0] * scale_dimensions[1]
    if out is None:
        qdata = torch.empty(qdata_dimensions, device=input.device, dtype=torch.uint8)
        scale = torch.empty(scale_dimensions, device=input.device, dtype=torch.float8_e4m3fn)
    else:
        qdata_storage, scale_storage = out
        if (
            qdata_storage.ndim != 2
            or qdata_storage.shape[0] < rows
            or qdata_storage.shape[1] != qdata_dimensions[1]
            or qdata_storage.dtype is not torch.uint8
            or scale_storage.numel() < scale_elements
            or scale_storage.dtype is not torch.float8_e4m3fn
            or qdata_storage.device != input.device
            or scale_storage.device != input.device
            or not qdata_storage.is_contiguous()
            or not scale_storage.is_contiguous()
        ):
            raise ValueError("NVFP4 activation storage is incompatible")
        qdata = qdata_storage[:rows]
        scale = scale_storage.flatten()[:scale_elements].view(scale_dimensions)
    if has_scale_padding(rows, features):
        scale.zero_()
    return qdata, scale


__all__ = [
    "BLOCK_SIZE",
    "QDATA_BLOCK_SIZE",
    "SCALE_COLUMN_TILE",
    "SCALE_ROW_TILE",
    "has_scale_padding",
    "prepare_activation_storage",
    "qdata_shape",
    "scale_shape",
    "swap_packed_pairs",
]
