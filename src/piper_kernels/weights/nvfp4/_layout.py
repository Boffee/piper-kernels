"""Canonical layout, swizzle, and allocation helpers for hardware-ready NVFP4 storage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch

BLOCK_SIZE = 16
QDATA_BLOCK_SIZE = BLOCK_SIZE // 2
SCALE_ROW_TILE = 128
SCALE_COLUMN_TILE = 64
E4M3_MAX = 448.0
E2M1_MAX = 6.0
# Keep reciprocal(global) / smallest_nonzero_E4M3 finite when computing weight scales.
MIN_WEIGHT_GLOBAL_SCALE = torch.finfo(torch.float32).tiny / (2.0**-9)


def swap_packed_pairs(qdata: torch.Tensor) -> torch.Tensor:
    """Exchange the two E2M1 values stored in every packed byte."""
    return ((qdata & 0x0F) << 4) | (qdata >> 4)


def per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert a per-tensor amax into the matching FP32 NVFP4 global decoding scale."""
    return amax.to(torch.float32) / (E4M3_MAX * E2M1_MAX)


def tensor_size_fp4x2_to_hp(
    qdata_shape: Sequence[int | torch.SymInt],
    *,
    transposed: bool,
) -> tuple[int | torch.SymInt, ...]:
    """Compute the logical high-precision shape corresponding to packed FP4x2 storage."""
    if not transposed:
        return (*qdata_shape[:-1], cast(Any, qdata_shape[-1]) * 2)
    return (*qdata_shape[:-2], cast(Any, qdata_shape[-2]) * 2, qdata_shape[-1])


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


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Swizzle 2-D row-major block scales into hardware ``SWIZZLE_32_4_4`` layout."""
    is_fp8 = input_matrix.dtype is torch.float8_e4m3fn
    # Route FP8 zero-padding through uint8 so CPU and meta devices avoid unsupported FP8 fills.
    work = input_matrix.view(torch.uint8) if is_fp8 else input_matrix
    rows, cols = work.shape
    n_row_blocks = (rows + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * SCALE_ROW_TILE
    padded_cols = n_col_blocks * 4
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=work.device, dtype=work.dtype)
        padded[:rows, :cols] = work
    else:
        padded = work
    blocks = padded.view(n_row_blocks, SCALE_ROW_TILE, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = (
        blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(n_row_blocks * 32, n_col_blocks * 16)
    )
    return rearranged.view(torch.float8_e4m3fn) if is_fp8 else rearranged


def from_blocked(
    blocked_tensor: torch.Tensor,
    original_rows: int,
    original_cols: int,
) -> torch.Tensor:
    """Inverse of :func:`to_blocked`: recover 2-D row-major scales ``[rows, cols]``."""
    is_fp8 = blocked_tensor.dtype is torch.float8_e4m3fn
    work = blocked_tensor.view(torch.uint8) if is_fp8 else blocked_tensor
    n_row_blocks = (original_rows + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE
    n_col_blocks = (original_cols + 3) // 4
    rearranged = work.reshape(n_row_blocks * n_col_blocks, 32, 16)
    temp = rearranged.reshape(n_row_blocks * n_col_blocks, 32, 4, 4).transpose(1, 2)
    blocks = temp.reshape(n_row_blocks, n_col_blocks, SCALE_ROW_TILE, 4)
    padded = blocks.permute(0, 2, 1, 3).reshape(n_row_blocks * SCALE_ROW_TILE, n_col_blocks * 4)
    result = padded[:original_rows, :original_cols]
    return result.view(torch.float8_e4m3fn) if is_fp8 else result


def has_scale_padding(rows: int, features: int) -> bool:
    """Return whether unused physical scale lanes must be initialized."""
    return rows % SCALE_ROW_TILE != 0 or features % SCALE_COLUMN_TILE != 0


__all__ = [
    "BLOCK_SIZE",
    "E2M1_MAX",
    "E4M3_MAX",
    "QDATA_BLOCK_SIZE",
    "SCALE_COLUMN_TILE",
    "SCALE_ROW_TILE",
    "from_blocked",
    "has_scale_padding",
    "per_tensor_amax_to_scale",
    "qdata_shape",
    "scale_shape",
    "swap_packed_pairs",
    "tensor_size_fp4x2_to_hp",
    "to_blocked",
]
