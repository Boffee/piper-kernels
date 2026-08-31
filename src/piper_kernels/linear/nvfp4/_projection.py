"""Raw prepared-NVFP4 projection helpers for fused consumers."""

# pyright: reportCallIssue=false

from __future__ import annotations

import torch

_SWIZZLE_ROWS = 128
_SWIZZLE_COLUMNS = 64
_SCALE_ELEMENTS_PER_TILE = 32 * 16
_BLOCKWISE_RECIPE = torch.nn.functional.ScalingType.BlockWise1x16.value
_TENSORWISE_RECIPE = torch.nn.functional.ScalingType.TensorWise.value
_SCALE_SWIZZLE = torch.nn.functional.SwizzleType.SWIZZLE_32_4_4.value
_NO_SWIZZLE = torch.nn.functional.SwizzleType.NO_SWIZZLE.value


def prepared_input_chunk(
    input_qdata: torch.Tensor,
    scale: torch.Tensor,
    row_start: int,
    row_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned packed-data and swizzled-scale views for one row chunk."""
    rows = input_qdata.shape[0]
    input_features = 2 * input_qdata.shape[1]
    if not 0 <= row_start < row_end <= rows:
        raise ValueError("NVFP4 projection chunk is outside the prepared input")
    if row_start % _SWIZZLE_ROWS or (row_end != rows and row_end % _SWIZZLE_ROWS):
        raise ValueError("NVFP4 projection chunks must align to 128-row scale blocks")
    column_blocks = (input_features + _SWIZZLE_COLUMNS - 1) // _SWIZZLE_COLUMNS
    elements_per_row_block = column_blocks * _SCALE_ELEMENTS_PER_TILE
    start = row_start // _SWIZZLE_ROWS * elements_per_row_block
    end_block = (row_end + _SWIZZLE_ROWS - 1) // _SWIZZLE_ROWS
    end = end_block * elements_per_row_block
    return input_qdata[row_start:row_end], scale.flatten()[start:end]


def _projection_chunk(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    row_start: int,
    row_end: int,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_rows = row_end - row_start
    if output.ndim != 2 or output.shape[0] < chunk_rows:
        raise ValueError("NVFP4 projection output does not have enough rows")
    if output.shape[1] != weight_qdata.shape[0]:
        raise ValueError("NVFP4 projection output width does not match its weight")
    input_chunk, chunk_scale = prepared_input_chunk(
        input_qdata,
        input_scale,
        row_start,
        row_end,
    )
    return input_chunk, chunk_scale, output[:chunk_rows]


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
    input_chunk, chunk_scale, output_chunk = _projection_chunk(
        input_qdata,
        input_scale,
        weight_qdata,
        row_start,
        row_end,
        output,
    )
    torch.ops.aten._scaled_mm.out(
        input_chunk.view(torch.float4_e2m1fn_x2),
        weight_qdata.t().view(torch.float4_e2m1fn_x2),
        chunk_scale.view(torch.float8_e4m3fn),
        weight_scale.view(torch.float8_e4m3fn),
        out_dtype=output.dtype,
        out=output_chunk,
    )
    return output_chunk


def matmul_prepared_chunk_affine_out(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    row_start: int,
    row_end: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run one prepared-NVFP4 GEMM chunk with its affine epilogue."""
    input_chunk, chunk_scale, output_chunk = _projection_chunk(
        input_qdata,
        input_scale,
        weight_qdata,
        row_start,
        row_end,
        output,
    )
    torch.ops.aten._scaled_mm_v2.out(
        input_chunk.view(torch.float4_e2m1fn_x2),
        weight_qdata.t().view(torch.float4_e2m1fn_x2),
        [chunk_scale.view(torch.float8_e4m3fn), input_per_tensor_scale],
        [_BLOCKWISE_RECIPE, _TENSORWISE_RECIPE],
        [_SCALE_SWIZZLE, _NO_SWIZZLE],
        [weight_scale.view(torch.float8_e4m3fn), weight_per_tensor_scale],
        [_BLOCKWISE_RECIPE, _TENSORWISE_RECIPE],
        [_SCALE_SWIZZLE, _NO_SWIZZLE],
        bias,
        output.dtype,
        out=output_chunk,
    )
    return output_chunk


__all__ = [
    "matmul_prepared_chunk_affine_out",
    "matmul_prepared_chunk_out",
    "prepared_input_chunk",
]
