"""Triton route selection shared by sparse Piper routing policies."""

# Triton's JIT launcher options and tensor return types are not represented in
# its Python signatures.
# pyright: reportCallIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context

_RADIX_BINS = tl.constexpr(256)
_SELECTOR_TILE = 512


@triton.jit(do_not_specialize=["capacity", "mask_words", "query_blocks"])
def _scatter_force_tile_bits(
    ids_ptr: torch.Tensor,
    counts_ptr: torch.Tensor,
    mask_ptr: torch.Tensor,
    capacity: int,
    mask_words: int,
    query_blocks: int,
    block: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    query = offsets // capacity
    slot = offsets % capacity
    count = tl.load(counts_ptr + query, query < query_blocks, other=0)
    active = (query < query_blocks) & (slot < count)
    key = tl.load(ids_ptr + offsets, active, other=0)
    tl.atomic_or(
        mask_ptr + query * mask_words + key // 32,
        (1 << (key % 32)).to(tl.int32),
        active,
        sem="relaxed",
    )


def build_force_tile_bitmask(
    ids: torch.Tensor, counts: torch.Tensor, sparse_key_blocks: int
) -> torch.Tensor:
    """Pack active IDs; membership is shared by every batch and query head."""
    with device_context(ids.device):
        mask = torch.zeros(
            (ids.shape[0], triton.cdiv(sparse_key_blocks, 32)),
            dtype=torch.int32,
            device=ids.device,
        )
        if ids.numel():
            _scatter_force_tile_bits[(triton.cdiv(ids.numel(), 256),)](
                ids,
                counts,
                mask,
                ids.shape[1],
                mask.shape[1],
                ids.shape[0],
                256,  # pyright: ignore[reportArgumentType]
            )
    return mask


@triton.jit
def _route_priority_bits(  # noqa: ANN202
    scores,  # noqa: ANN001
    key_offsets,  # noqa: ANN001
    valid,  # noqa: ANN001
    force_mask_ptr,  # noqa: ANN001
    has_force: tl.constexpr,
):
    ordered_bits = _ordered_float_bits(scores)
    if has_force:
        words = tl.load(force_mask_ptr + key_offsets // 32, valid, other=0).to(tl.uint32)
        forced = ((words >> (key_offsets % 32)) & 1) != 0
        # Reserve the maximum priority for forced IDs. Original scores are never
        # modified, including when coarse attention consumes the same chunk.
        ordered_bits = tl.where(forced, 0xFFFFFFFF, ordered_bits)
    return ordered_bits


@triton.jit
def _fill_full_keep_routes_kernel(  # noqa: ANN202
    routes_ptr,  # noqa: ANN001
    count,  # noqa: ANN001
    sparse_key_blocks: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    tl.store(routes_ptr + offsets, offsets % sparse_key_blocks, offsets < count)


def fill_full_keep_routes(routes: torch.Tensor, sparse_key_blocks: int) -> None:
    """Write canonical routes directly, without summaries or top-k selection."""
    with device_context(routes.device):
        _fill_full_keep_routes_kernel[(triton.cdiv(routes.numel(), 1024),)](
            routes,
            routes.numel(),
            sparse_key_blocks,  # pyright: ignore[reportArgumentType]
            1024,  # pyright: ignore[reportArgumentType]
        )


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
    force_mask_ptr,  # noqa: ANN001
    has_force: tl.constexpr,
    start_key_block,  # noqa: ANN001
):
    """Select one radix byte while streaming a score row in fixed tiles."""
    histogram = tl.zeros([_RADIX_BINS], tl.int32)
    for tile_start in range(0, sparse_key_blocks, selector_tile):
        key_offsets = tile_start + tl.arange(0, selector_tile)
        valid = key_offsets < sparse_key_blocks
        scores = tl.load(scores_ptr + score_base + key_offsets, mask=valid, other=0.0)
        ordered_bits = _route_priority_bits(
            scores, start_key_block + key_offsets, valid, force_mask_ptr, has_force
        )
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
    chosen_digit = (_RADIX_BINS - 1 - descending_bin).to(tl.uint32)  # pyright: ignore[reportAttributeAccessIssue]
    return (prefix << 8) | chosen_digit, rank - preceding


@triton.jit(
    do_not_specialize=[
        "sparse_key_blocks",
        "query_blocks",
        "query_block_offset",
        "score_batch_stride",
        "score_head_stride",
        "score_query_stride",
        "route_batch_stride",
        "route_query_stride",
        "route_route_stride",
        "force_mask_stride",
    ]
)
def _tiled_radix_select_packed_routes_kernel(  # noqa: PLR0913, PLR0917
    scores_ptr: torch.Tensor,
    routes_ptr: torch.Tensor,
    head_keep_blocks_ptr: torch.Tensor,
    route_head_offsets_ptr: torch.Tensor,
    force_mask_ptr: torch.Tensor,
    region_metadata_ptr: torch.Tensor,
    sparse_key_blocks: int,
    query_blocks: int,
    query_block_offset: int,
    score_batch_stride: int,
    score_head_stride: int,
    score_query_stride: int,
    route_batch_stride: int,
    route_query_stride: int,
    route_route_stride: int,
    force_mask_stride: int,
    num_heads: tl.constexpr,
    has_regions: tl.constexpr,
    compact_scores: tl.constexpr,
    selector_tile: tl.constexpr,
    has_force: tl.constexpr,
) -> None:
    """Select exact stable FP32 top-k routes with runtime row dimensions."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    start_key_block = 0
    start_score_block = 0
    if has_regions:
        region = query_block // query_blocks
        query_block = query_block % query_blocks
        metadata = region_metadata_ptr + region * (2 * num_heads + 3)
        start_key_block = tl.load(metadata)
        sparse_key_blocks = tl.load(metadata + 1)
        start_score_block = tl.load(metadata + 2) if compact_scores else start_key_block
        head_keep_block_count = tl.load(metadata + 3 + head)
        route_head_offset = tl.load(metadata + 3 + num_heads + head)
    else:
        route_head_offset = tl.load(route_head_offsets_ptr + head)
        head_keep_block_count = tl.load(head_keep_blocks_ptr + head)
    score_base = (
        batch * score_batch_stride
        + head * score_head_stride
        + query_block * score_query_stride
        + start_score_block
    )
    output_base = (
        batch * route_batch_stride
        + (query_block_offset + query_block) * route_query_stride
        + route_head_offset * route_route_stride
    )
    force_row = force_mask_ptr + (query_block_offset + query_block) * force_mask_stride

    if has_regions and head_keep_block_count == sparse_key_blocks:
        for tile_start in range(0, sparse_key_blocks, selector_tile):
            keys = tile_start + tl.arange(0, selector_tile)
            tl.store(
                routes_ptr + output_base + keys * route_route_stride,
                start_key_block + keys,
                keys < sparse_key_blocks,
            )
    else:
        prefix = 0
        rank = head_keep_block_count
        prefix, rank = _tiled_radix_select_pass(
            scores_ptr,
            score_base,
            sparse_key_blocks,
            prefix,
            rank,
            tl.constexpr(24),
            selector_tile,
            force_row,
            has_force,
            start_key_block,
        )
        prefix, rank = _tiled_radix_select_pass(
            scores_ptr,
            score_base,
            sparse_key_blocks,
            prefix,
            rank,
            tl.constexpr(16),
            selector_tile,
            force_row,
            has_force,
            start_key_block,
        )
        prefix, rank = _tiled_radix_select_pass(
            scores_ptr,
            score_base,
            sparse_key_blocks,
            prefix,
            rank,
            tl.constexpr(8),
            selector_tile,
            force_row,
            has_force,
            start_key_block,
        )
        threshold_bits, equal_keep = _tiled_radix_select_pass(
            scores_ptr,
            score_base,
            sparse_key_blocks,
            prefix,
            rank,
            tl.constexpr(0),
            selector_tile,
            force_row,
            has_force,
            start_key_block,
        )

        output_offset = 0
        equal_offset = 0
        for tile_start in range(0, sparse_key_blocks, selector_tile):
            key_offsets = tile_start + tl.arange(0, selector_tile)
            valid = key_offsets < sparse_key_blocks
            scores = tl.load(scores_ptr + score_base + key_offsets, mask=valid, other=0.0)
            ordered_bits = _route_priority_bits(
                scores, start_key_block + key_offsets, valid, force_row, has_force
            )
            equal = valid & (ordered_bits == threshold_bits)
            equal_rank = equal_offset + tl.cumsum(equal.to(tl.int32), axis=0)
            selected = valid & (
                (ordered_bits > threshold_bits) | (equal & (equal_rank <= equal_keep))
            )
            local_output_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
            tl.store(
                routes_ptr + output_base + (output_offset + local_output_rank) * route_route_stride,
                start_key_block + key_offsets,
                mask=selected,
            )
            output_offset += tl.sum(selected.to(tl.int32), axis=0)  # pyright: ignore[reportOperatorIssue]
            equal_offset += tl.sum(equal.to(tl.int32), axis=0)  # pyright: ignore[reportOperatorIssue]


def tiled_radix_select_packed_routes(
    scores: torch.Tensor,
    routes: torch.Tensor,
    head_keep_blocks: torch.Tensor,
    route_head_offsets: torch.Tensor,
    *,
    query_block_offset: int,
    force_tile_mask: torch.Tensor | None = None,
    region_metadata: torch.Tensor | None = None,
    compact_scores: bool = False,
) -> None:
    """Write exact stable FP32 top-k routes directly into packed UINT16 storage."""
    if scores.ndim != 4 or scores.stride(-1) != 1 or scores.dtype != torch.float32:
        raise ValueError("tiled radix selection requires rank-four FP32 scores with dense keys")
    if routes.ndim != 3 or routes.dtype != torch.uint16 or not routes.is_contiguous():
        raise ValueError(
            "packed tiled radix selection requires contiguous rank-three uint16 routes"
        )
    if (
        head_keep_blocks.ndim != 1
        or head_keep_blocks.shape[0] != scores.shape[1]
        or head_keep_blocks.dtype != torch.int32
        or head_keep_blocks.device != routes.device
        or not head_keep_blocks.is_contiguous()
    ):
        raise ValueError("tiled head keep blocks must be a contiguous device int32 head vector")
    if (
        route_head_offsets.shape != (scores.shape[1] + 1,)
        or route_head_offsets.dtype != torch.int32
        or route_head_offsets.device != routes.device
        or not route_head_offsets.is_contiguous()
    ):
        raise ValueError("packed tiled route head offsets must be a contiguous device int32 vector")
    if scores.device != routes.device or scores.device.type != "cuda":
        raise ValueError("tiled scores and routes must share a CUDA device")
    if not 0 <= query_block_offset <= routes.shape[1] - scores.shape[2]:
        raise ValueError("tiled query range must fit the packed route output")
    if routes.shape[0] != scores.shape[0]:
        raise ValueError("tiled scores and packed routes must share the batch dimension")
    batch, heads, query_blocks, sparse_key_blocks = scores.shape
    if region_metadata is not None and (
        region_metadata.ndim != 2
        or region_metadata.shape[1] != 2 * heads + 3
        or region_metadata.dtype is not torch.int32
        or region_metadata.device != scores.device
        or not region_metadata.is_contiguous()
    ):
        raise ValueError("region metadata must use contiguous INT32 [regions, 2 * heads + 3]")
    num_regions = 1 if region_metadata is None else region_metadata.shape[0]
    if force_tile_mask is not None and (
        force_tile_mask.ndim != 2
        or force_tile_mask.shape[0] != routes.shape[1]
        or (
            region_metadata is None
            and force_tile_mask.shape[1] != triton.cdiv(sparse_key_blocks, 32)
        )
        or force_tile_mask.dtype is not torch.int32
        or force_tile_mask.device != scores.device
        or not force_tile_mask.is_contiguous()
    ):
        raise ValueError("forced membership must use contiguous INT32 [query_blocks,key_words]")
    with device_context(scores.device):
        # Regions can exceed CUDA's 65,535 limit on the second grid dimension
        # when combined with heads. Keep them on the larger first dimension.
        _tiled_radix_select_packed_routes_kernel[(query_blocks * num_regions, heads, batch)](
            scores,
            routes,
            head_keep_blocks,
            route_head_offsets,
            head_keep_blocks if force_tile_mask is None else force_tile_mask,
            head_keep_blocks if region_metadata is None else region_metadata,
            sparse_key_blocks,
            query_blocks,
            query_block_offset,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            routes.stride(0),
            routes.stride(1),
            routes.stride(2),
            0 if force_tile_mask is None else force_tile_mask.stride(0),
            num_heads=heads,
            has_regions=region_metadata is not None,
            compact_scores=compact_scores,
            selector_tile=_SELECTOR_TILE,
            has_force=force_tile_mask is not None,
            num_warps=4,
            num_stages=1,
        )


__all__ = ["tiled_radix_select_packed_routes"]
