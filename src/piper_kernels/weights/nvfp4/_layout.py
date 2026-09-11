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


__all__ = [
    "BLOCK_SIZE",
    "QDATA_BLOCK_SIZE",
    "SCALE_COLUMN_TILE",
    "SCALE_ROW_TILE",
    "has_scale_padding",
    "qdata_shape",
    "scale_shape",
    "swap_packed_pairs",
]
