"""Triton primitives for exact blockwise DSA routing."""

# Triton's JIT launcher options and tensor return types are not represented in
# its Python signatures.
# pyright: reportCallIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_ROWS = 64
_HEAD_DIM = 128
_RADIX_BINS = tl.constexpr(256)
_SELECTOR_TILE = 512


@triton.jit
def _ordered_float_bits(scores):  # noqa: ANN001, ANN202
    """Map finite FP32 values to unsigned integers with the same ordering."""
    # Canonicalize signed zero, which compares equal under the reference sort.
    scores = tl.where(scores == 0.0, 0.0, scores)
    bits = scores.to(tl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return tl.where(sign != 0, ~bits, bits ^ 0x80000000)


@triton.jit
def _tiled_radix_select_pass(  # noqa: ANN202
    scores_ptr,  # noqa: ANN001
    score_base,  # noqa: ANN001
    sparse_key_blocks,  # noqa: ANN001
    prefix,  # noqa: ANN001
    rank,  # noqa: ANN001
    shift: tl.constexpr,
    selector_tile: tl.constexpr,
):
    """Select one radix byte while streaming a score row in fixed tiles."""
    histogram = tl.zeros([_RADIX_BINS], tl.int32)
    for tile_start in range(0, sparse_key_blocks, selector_tile):
        key_offsets = tile_start + tl.arange(0, selector_tile)
        valid = key_offsets < sparse_key_blocks
        scores = tl.load(scores_ptr + score_base + key_offsets, mask=valid, other=0.0)
        ordered_bits = _ordered_float_bits(scores)
        active = valid
        if shift < 24:
            active &= (ordered_bits >> (shift + 8)) == prefix
        digits = (ordered_bits >> shift) & 0xFF
        histogram += tl.histogram(digits.to(tl.int32), _RADIX_BINS, mask=active)

    bins = tl.arange(0, _RADIX_BINS)
    descending_counts = tl.flip(histogram, dim=0)
    cumulative = tl.cumsum(descending_counts, axis=0)
    descending_bin = tl.min(tl.where(cumulative >= rank, bins, _RADIX_BINS), axis=0)
    preceding = tl.sum(
        tl.where(bins < descending_bin, descending_counts, 0),
        axis=0,
    )
    chosen_digit = (_RADIX_BINS - 1 - descending_bin).to(tl.uint32)
    return (prefix << 8) | chosen_digit, rank - preceding


@triton.jit(
    do_not_specialize=[
        "sparse_key_blocks",
        "query_block_offset",
        "score_batch_stride",
        "score_head_stride",
        "score_query_stride",
        "route_batch_stride",
        "route_query_stride",
        "route_route_stride",
    ]
)
def _tiled_radix_select_packed_routes_kernel(  # noqa: PLR0913, PLR0917
    scores_ptr: torch.Tensor,
    routes_ptr: torch.Tensor,
    head_keep_blocks_ptr: torch.Tensor,
    route_head_offsets_ptr: torch.Tensor,
    sparse_key_blocks: int,
    query_block_offset: int,
    score_batch_stride: int,
    score_head_stride: int,
    score_query_stride: int,
    route_batch_stride: int,
    route_query_stride: int,
    route_route_stride: int,
    selector_tile: tl.constexpr,
) -> None:
    """Select exact stable FP32 top-k routes with runtime row dimensions."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    score_base = (
        batch * score_batch_stride + head * score_head_stride + query_block * score_query_stride
    )
    route_head_offset = tl.load(route_head_offsets_ptr + head)
    output_base = (
        batch * route_batch_stride
        + (query_block_offset + query_block) * route_query_stride
        + route_head_offset * route_route_stride
    )
    head_keep_block_count = tl.load(head_keep_blocks_ptr + head)

    prefix = 0
    rank = head_keep_block_count
    prefix, rank = _tiled_radix_select_pass(
        scores_ptr,
        score_base,
        sparse_key_blocks,
        prefix,
        rank,
        24,
        selector_tile,
    )
    prefix, rank = _tiled_radix_select_pass(
        scores_ptr,
        score_base,
        sparse_key_blocks,
        prefix,
        rank,
        16,
        selector_tile,
    )
    prefix, rank = _tiled_radix_select_pass(
        scores_ptr,
        score_base,
        sparse_key_blocks,
        prefix,
        rank,
        8,
        selector_tile,
    )
    threshold_bits, equal_keep = _tiled_radix_select_pass(
        scores_ptr,
        score_base,
        sparse_key_blocks,
        prefix,
        rank,
        0,
        selector_tile,
    )

    output_offset = 0
    equal_offset = 0
    for tile_start in range(0, sparse_key_blocks, selector_tile):
        key_offsets = tile_start + tl.arange(0, selector_tile)
        valid = key_offsets < sparse_key_blocks
        scores = tl.load(scores_ptr + score_base + key_offsets, mask=valid, other=0.0)
        ordered_bits = _ordered_float_bits(scores)
        equal = valid & (ordered_bits == threshold_bits)
        equal_rank = equal_offset + tl.cumsum(equal.to(tl.int32), axis=0)
        selected = valid & ((ordered_bits > threshold_bits) | (equal & (equal_rank <= equal_keep)))
        local_output_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
        tl.store(
            routes_ptr + output_base + (output_offset + local_output_rank) * route_route_stride,
            key_offsets,
            mask=selected,
        )
        output_offset += tl.sum(selected.to(tl.int32), axis=0)
        equal_offset += tl.sum(equal.to(tl.int32), axis=0)


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


def tiled_radix_select_packed_routes(
    scores: torch.Tensor,
    routes: torch.Tensor,
    head_keep_blocks: torch.Tensor,
    route_head_offsets: torch.Tensor,
    *,
    query_block_offset: int,
) -> None:
    """Write exact FP32 top-k routes directly into packed UINT16 storage."""
    if scores.ndim != 4 or scores.stride(-1) != 1 or scores.dtype != torch.float32:
        raise ValueError("tiled DSA radix selection requires rank-four FP32 scores with dense keys")
    if routes.ndim != 3 or routes.dtype != torch.uint16 or not routes.is_contiguous():
        raise ValueError(
            "packed tiled DSA radix selection requires contiguous rank-three uint16 routes"
        )
    if (
        head_keep_blocks.ndim != 1
        or head_keep_blocks.shape[0] != scores.shape[1]
        or head_keep_blocks.dtype != torch.int32
        or head_keep_blocks.device != routes.device
        or not head_keep_blocks.is_contiguous()
    ):
        raise ValueError("tiled DSA head keep blocks must be a contiguous device int32 head vector")
    if (
        route_head_offsets.shape != (scores.shape[1] + 1,)
        or route_head_offsets.dtype != torch.int32
        or route_head_offsets.device != routes.device
        or not route_head_offsets.is_contiguous()
    ):
        raise ValueError(
            "packed tiled DSA route head offsets must be a contiguous device int32 vector"
        )
    if scores.device != routes.device or scores.device.type != "cuda":
        raise ValueError("tiled DSA scores and routes must share a CUDA device")
    if not 0 <= query_block_offset <= routes.shape[1] - scores.shape[2]:
        raise ValueError("tiled DSA query range must fit the packed route output")
    if routes.shape[0] != scores.shape[0]:
        raise ValueError("tiled DSA scores and packed routes must share the batch dimension")
    batch, heads, query_blocks, sparse_key_blocks = scores.shape
    _tiled_radix_select_packed_routes_kernel[(query_blocks, heads, batch)](
        scores,
        routes,
        head_keep_blocks,
        route_head_offsets,
        sparse_key_blocks,
        query_block_offset,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        routes.stride(0),
        routes.stride(1),
        routes.stride(2),
        selector_tile=_SELECTOR_TILE,
        num_warps=4,
        num_stages=1,
    )
