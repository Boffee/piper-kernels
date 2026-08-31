"""Raw prepared-NVFP4 projection helpers for fused consumers."""

# pyright: reportCallIssue=false

from __future__ import annotations

import torch

_SWIZZLE_ROWS = 128
_SWIZZLE_COLUMNS = 64
_SCALE_ELEMENTS_PER_TILE = 32 * 16


def _scale_slice(
    scale: torch.Tensor,
    row_start: int,
    row_end: int,
    rows: int,
    input_features: int,
) -> torch.Tensor:
    if row_start % _SWIZZLE_ROWS or (row_end != rows and row_end % _SWIZZLE_ROWS):
        raise ValueError("NVFP4 projection chunks must align to 128-row scale blocks")
    column_blocks = (input_features + _SWIZZLE_COLUMNS - 1) // _SWIZZLE_COLUMNS
    elements_per_row_block = column_blocks * _SCALE_ELEMENTS_PER_TILE
    start = row_start // _SWIZZLE_ROWS * elements_per_row_block
    end_block = (row_end + _SWIZZLE_ROWS - 1) // _SWIZZLE_ROWS
    end = end_block * elements_per_row_block
    return scale.flatten()[start:end]


def matmul_prepared_chunk_out(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    row_start: int,
    row_end: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run one raw prepared-NVFP4 GEMM chunk into caller-owned storage."""
    rows = input_qdata.shape[0]
    input_features = 2 * input_qdata.shape[1]
    chunk_rows = row_end - row_start
    if not 0 <= row_start < row_end <= rows:
        raise ValueError("NVFP4 projection chunk is outside the prepared input")
    if output.ndim != 2 or output.shape[0] < chunk_rows:
        raise ValueError("NVFP4 projection output does not have enough rows")
    if output.shape[1] != weight_qdata.shape[0]:
        raise ValueError("NVFP4 projection output width does not match its weight")
    output_chunk = output[:chunk_rows]
    chunk_scale = _scale_slice(input_scale, row_start, row_end, rows, input_features)
    torch.ops.aten._scaled_mm.out(
        input_qdata[row_start:row_end].view(torch.float4_e2m1fn_x2),
        weight_qdata.t().view(torch.float4_e2m1fn_x2),
        chunk_scale.view(torch.float8_e4m3fn),
        weight_scale.view(torch.float8_e4m3fn),
        out_dtype=output.dtype,
        out=output_chunk,
    )
    return output_chunk


__all__ = ["matmul_prepared_chunk_out"]
