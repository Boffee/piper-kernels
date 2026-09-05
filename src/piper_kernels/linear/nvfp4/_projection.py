"""Shared prepared-NVFP4 GEMMs and bounded FP32 affine workspaces."""

# pyright: reportCallIssue=false

from __future__ import annotations

import torch

from . import _layout
from . import triton as nvfp4_backend

_SCALE_ELEMENTS_PER_TILE = _layout.SCALE_ROW_TILE * _layout.SCALE_COLUMN_TILE // _layout.BLOCK_SIZE
_BLOCKWISE_RECIPE = torch.nn.functional.ScalingType.BlockWise1x16.value
_TENSORWISE_RECIPE = torch.nn.functional.ScalingType.TensorWise.value
_SCALE_SWIZZLE = torch.nn.functional.SwizzleType.SWIZZLE_32_4_4.value
_NO_SWIZZLE = torch.nn.functional.SwizzleType.NO_SWIZZLE.value
_FP32_WORKSPACE_BYTES = 32 * 1024 * 1024


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
    if row_start % _layout.SCALE_ROW_TILE or (row_end != rows and row_end % _layout.SCALE_ROW_TILE):
        raise ValueError("NVFP4 projection chunks must align to 128-row scale blocks")
    column_blocks = (input_features + _layout.SCALE_COLUMN_TILE - 1) // _layout.SCALE_COLUMN_TILE
    elements_per_row_block = column_blocks * _SCALE_ELEMENTS_PER_TILE
    start = row_start // _layout.SCALE_ROW_TILE * elements_per_row_block
    end_block = (row_end + _layout.SCALE_ROW_TILE - 1) // _layout.SCALE_ROW_TILE
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


def _matmul_affine_out(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    torch.ops.aten._scaled_mm_v2.out(
        input_qdata.view(torch.float4_e2m1fn_x2),
        weight_qdata.t().view(torch.float4_e2m1fn_x2),
        [input_scale.view(torch.float8_e4m3fn), input_per_tensor_scale],
        [_BLOCKWISE_RECIPE, _TENSORWISE_RECIPE],
        [_SCALE_SWIZZLE, _NO_SWIZZLE],
        [weight_scale.view(torch.float8_e4m3fn), weight_per_tensor_scale],
        [_BLOCKWISE_RECIPE, _TENSORWISE_RECIPE],
        [_SCALE_SWIZZLE, _NO_SWIZZLE],
        # The native GEMM bias operand has no stride argument.
        bias.contiguous() if bias is not None else None,
        output.dtype,
        out=output,
    )


def matmul_prepared_chunk_affine_out(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    row_start: int,
    row_end: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Scale and add bias before the final low-precision store.

    GEMM fuses compatible bias directly. Mixed bias uses FP32 row chunks so it
    retains its precision without a full-size FP32 intermediate. The workspace
    holds at most 32 MiB or one 128-row scale block, whichever is larger.
    """
    input_chunk, chunk_scale, output_chunk = _projection_chunk(
        input_qdata,
        input_scale,
        weight_qdata,
        row_start,
        row_end,
        output,
    )
    if weight_per_tensor_scale is None:
        weight_per_tensor_scale = input_per_tensor_scale.new_ones(())
    if bias is None or (bias.dtype is output.dtype and output.dtype is not torch.float32):
        _matmul_affine_out(
            input_chunk,
            chunk_scale,
            input_per_tensor_scale,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            bias,
            output_chunk,
        )
        return output_chunk
    if output.dtype is torch.float32:
        _matmul_affine_out(
            input_chunk,
            chunk_scale,
            input_per_tensor_scale,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            None,
            output_chunk,
        )
        nvfp4_backend.add_bias_out(output_chunk, bias, output_chunk)
        return output_chunk

    rows, features = output_chunk.shape
    rows_per_block = max(1, _FP32_WORKSPACE_BYTES // (4 * features * _layout.SCALE_ROW_TILE))
    chunk_rows = min(rows, rows_per_block * _layout.SCALE_ROW_TILE)
    workspace = torch.empty((chunk_rows, features), device=output.device, dtype=torch.float32)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        qdata, scale = prepared_input_chunk(input_chunk, chunk_scale, start, stop)
        accumulated = workspace[: stop - start]
        _matmul_affine_out(
            qdata,
            scale,
            input_per_tensor_scale,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            None,
            accumulated,
        )
        nvfp4_backend.add_bias_out(
            accumulated,
            bias,
            output_chunk[start:stop],
        )
    return output_chunk


__all__ = [
    "matmul_prepared_chunk_affine_out",
    "matmul_prepared_chunk_out",
    "prepared_input_chunk",
]
