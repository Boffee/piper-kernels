"""Triton block summaries for exact DSA routing."""

# Triton's JIT launcher options and tensor return types are not represented in
# its Python signatures.
# pyright: reportCallIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_ROWS = 64
_HEAD_DIM = 128


@triton.jit(
    do_not_specialize=[
        "logical_block_count",
        "logical_row_count",
        "stride_ib",
        "stride_ih",
        "stride_il",
        "stride_ir",
    ]
)
def _block_summary_kernel(  # noqa: PLR0913, PLR0917
    input_ptr: torch.Tensor,
    output_max_ptr: torch.Tensor,
    output_min_ptr: torch.Tensor,
    block_lengths_ptr: torch.Tensor,
    logical_block_count: int,
    logical_row_count: int,
    stride_ib: int,
    stride_ih: int,
    stride_il: int,
    stride_ir: int,
    block_rows: tl.constexpr,
    head_dim: tl.constexpr,
    heads: tl.constexpr,
    sum_extrema: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
) -> None:
    block = tl.program_id(0)
    batch_head = block // logical_block_count
    logical_block = block % logical_block_count
    batch = batch_head // heads
    head = batch_head % heads
    row_offsets = tl.arange(0, block_rows)[:, None]
    feature_offsets = tl.arange(0, head_dim)[None, :]
    input_offsets = (
        batch * stride_ib
        + head * stride_ih
        + logical_block * stride_il
        + row_offsets * stride_ir
        + feature_offsets
    )
    if mask_block_lengths or mask_ragged_tail:
        if mask_block_lengths:
            valid_rows = row_offsets < tl.load(block_lengths_ptr + logical_block)
        else:
            valid_rows = logical_block * block_rows + row_offsets < logical_row_count
        values = tl.load(
            input_ptr + input_offsets,
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)
        maximum = tl.max(tl.where(valid_rows, values, -float("inf")), axis=0)
        minimum = tl.min(tl.where(valid_rows, values, float("inf")), axis=0)
    else:
        values = tl.load(input_ptr + input_offsets).to(tl.float32)
        maximum = tl.max(values, axis=0)
        minimum = tl.min(values, axis=0)
    output_offsets = block * head_dim + tl.arange(0, head_dim)
    tl.store(output_max_ptr + output_offsets, maximum + minimum if sum_extrema else maximum)
    if not sum_extrema:
        tl.store(output_min_ptr + output_offsets, minimum)


def sequence_block_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize compact or valid-front padded Q/K blocks."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("optimized DSA sequence summaries require rank-four Q/K tensors")
    if (
        query.shape[:2] != key.shape[:2]
        or query.shape[-1] != _HEAD_DIM
        or key.shape[-1] != _HEAD_DIM
        or query.shape[2] < 1
        or key.shape[2] < _BLOCK_ROWS
    ):
        raise ValueError("optimized DSA sequences require nonempty ragged Q/K with D128 K")
    if query.device.type != "cuda" or query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("optimized DSA sequence summaries require CUDA BF16/FP16 inputs")
    if key.device != query.device or key.dtype != query.dtype:
        raise ValueError("optimized DSA Q/K sequences must share a device and dtype")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("optimized DSA sequences require contiguous feature dimensions")

    batch, heads, query_rows, _head_dim = query.shape
    key_rows = key.shape[2]
    if block_lengths is not None and (
        query_rows % _BLOCK_ROWS
        or block_lengths.shape != (query_rows // _BLOCK_ROWS,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != query.device
        or not block_lengths.is_contiguous()
        or key_rows % _BLOCK_ROWS
        or key_rows // _BLOCK_ROWS > block_lengths.numel()
    ):
        raise ValueError(
            "optimized padded DSA requires one contiguous device INT32 length per query K64"
        )
    query_blocks = (query_rows + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    key_blocks = (key_rows + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    query_summary = torch.empty(
        (batch, heads, query_blocks, _HEAD_DIM),
        device=query.device,
        dtype=torch.float32,
    )
    key_max = torch.empty(
        (batch, heads, key_blocks, _HEAD_DIM),
        device=key.device,
        dtype=torch.float32,
    )
    key_min = torch.empty_like(key_max)
    _block_summary_kernel[(batch * heads * query_blocks,)](
        query,
        query_summary,
        query_summary,
        query if block_lengths is None else block_lengths,
        logical_block_count=query_blocks,
        logical_row_count=query_rows,
        stride_ib=query.stride(0),
        stride_ih=query.stride(1),
        stride_il=_BLOCK_ROWS * query.stride(2),
        stride_ir=query.stride(2),
        block_rows=_BLOCK_ROWS,
        head_dim=_HEAD_DIM,
        heads=heads,
        sum_extrema=True,
        mask_block_lengths=block_lengths is not None,
        mask_ragged_tail=query_rows % _BLOCK_ROWS != 0,
        num_warps=4,
        num_stages=1,
    )
    _block_summary_kernel[(batch * heads * key_blocks,)](
        key,
        key_max,
        key_min,
        key if block_lengths is None else block_lengths,
        logical_block_count=key_blocks,
        logical_row_count=key_rows,
        stride_ib=key.stride(0),
        stride_ih=key.stride(1),
        stride_il=_BLOCK_ROWS * key.stride(2),
        stride_ir=key.stride(2),
        block_rows=_BLOCK_ROWS,
        head_dim=_HEAD_DIM,
        heads=heads,
        sum_extrema=False,
        mask_block_lengths=block_lengths is not None,
        mask_ragged_tail=block_lengths is None and key_rows % _BLOCK_ROWS != 0,
        num_warps=4,
        num_stages=1,
    )
    return query_summary, key_max, key_min
