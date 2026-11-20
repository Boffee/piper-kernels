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
_SINGLE_LOAD_BLOCK_SIZES = (512, 1024, 2048)
_RADIX_DO_NOT_SPECIALIZE = (
    "sparse_key_blocks",
    "query_block_offset",
    "score_batch_stride",
    "score_head_stride",
    "score_query_stride",
    "route_batch_stride",
    "route_query_stride",
    "route_route_stride",
    "force_mask_stride",
)


def _select_single_load_block_k(sparse_key_blocks: int) -> int | None:
    """Choose a power-of-two single-load radix width when rows fit without spilling."""
    for block_k in _SINGLE_LOAD_BLOCK_SIZES:
        if sparse_key_blocks <= block_k:
            return block_k
    return None


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
def _select_radix_digit(histogram, prefix, rank):  # noqa: ANN001, ANN202
    """Resolve the winning descending radix digit and remaining rank."""
    bins = tl.arange(0, _RADIX_BINS)
    descending_counts = tl.flip(histogram, dim=0)
    cumulative = tl.cumsum(descending_counts, axis=0)
    descending_bin = tl.min(tl.where(cumulative >= rank, bins, _RADIX_BINS), axis=0)
    preceding = tl.sum(
        tl.where(bins < descending_bin, descending_counts, 0),
        axis=0,
    )
    bin_count = tl.sum(
        tl.where(bins == descending_bin, descending_counts, 0),
        axis=0,
    )
    chosen_digit = (_RADIX_BINS - 1 - descending_bin).to(tl.uint32)  # pyright: ignore[reportAttributeAccessIssue]
    return (prefix << 8) | chosen_digit, rank - preceding, bin_count


@triton.jit
def _resolve_route_row(  # noqa: PLR0913, PLR0917, ANN202
    head_keep_blocks_ptr,  # noqa: ANN001
    route_head_offsets_ptr,  # noqa: ANN001
    force_mask_ptr,  # noqa: ANN001
    query_block_offset,  # noqa: ANN001
    score_batch_stride,  # noqa: ANN001
    score_head_stride,  # noqa: ANN001
    score_query_stride,  # noqa: ANN001
    route_batch_stride,  # noqa: ANN001
    route_query_stride,  # noqa: ANN001
    route_route_stride,  # noqa: ANN001
    force_mask_stride,  # noqa: ANN001
):
    """Compute score/output row bases, head keep count, and force row pointer."""
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
    rank = tl.load(head_keep_blocks_ptr + head)
    force_row = force_mask_ptr + (query_block_offset + query_block) * force_mask_stride
    return score_base, output_base, rank, force_row


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
):
    """Select one radix byte while streaming a score row in fixed tiles."""
    histogram = tl.zeros([_RADIX_BINS], tl.int32)
    for tile_start in range(0, sparse_key_blocks, selector_tile):
        key_offsets = tile_start + tl.arange(0, selector_tile)
        valid = key_offsets < sparse_key_blocks
        scores = tl.load(scores_ptr + score_base + key_offsets, mask=valid, other=0.0)
        ordered_bits = _route_priority_bits(scores, key_offsets, valid, force_mask_ptr, has_force)
        active = valid
        if shift < 24:
            active &= (ordered_bits >> (shift + 8)) == prefix
        digits = (ordered_bits >> shift) & 0xFF
        histogram += tl.histogram(digits.to(tl.int32), _RADIX_BINS, mask=active)

    return _select_radix_digit(histogram, prefix, rank)


@triton.jit
def _in_register_radix_select_pass(  # noqa: ANN202
    ordered_bits,  # noqa: ANN001
    valid,  # noqa: ANN001
    prefix,  # noqa: ANN001
    rank,  # noqa: ANN001
    shift: tl.constexpr,
):
    """Select one radix byte from score priorities already loaded in registers."""
    active = valid
    if shift < 24:
        active &= (ordered_bits >> (shift + 8)) == prefix
    digits = (ordered_bits >> shift) & 0xFF
    histogram = tl.histogram(digits.to(tl.int32), _RADIX_BINS, mask=active)
    return _select_radix_digit(histogram, prefix, rank)


@triton.jit(do_not_specialize=_RADIX_DO_NOT_SPECIALIZE)
def _tiled_radix_select_packed_routes_kernel(  # noqa: PLR0913, PLR0917
    scores_ptr: torch.Tensor,
    routes_ptr: torch.Tensor,
    head_keep_blocks_ptr: torch.Tensor,
    route_head_offsets_ptr: torch.Tensor,
    force_mask_ptr: torch.Tensor,
    sparse_key_blocks: int,
    query_block_offset: int,
    score_batch_stride: int,
    score_head_stride: int,
    score_query_stride: int,
    route_batch_stride: int,
    route_query_stride: int,
    route_route_stride: int,
    force_mask_stride: int,
    selector_tile: tl.constexpr,
    has_force: tl.constexpr,
) -> None:
    """Select exact stable FP32 top-k routes with runtime row dimensions."""
    score_base, output_base, rank, force_row = _resolve_route_row(
        head_keep_blocks_ptr,
        route_head_offsets_ptr,
        force_mask_ptr,
        query_block_offset,
        score_batch_stride,
        score_head_stride,
        score_query_stride,
        route_batch_stride,
        route_query_stride,
        route_route_stride,
        force_mask_stride,
    )

    prefix = tl.full((), 0, tl.uint32)
    resolved_shift = tl.full((), 24, tl.uint32)
    prefix, rank, bin_count = _tiled_radix_select_pass(
        scores_ptr,
        score_base,
        sparse_key_blocks,
        prefix,
        rank,
        tl.constexpr(24),
        selector_tile,
        force_row,
        has_force,
    )
    # When the threshold digit's bucket count equals the remaining rank, all of
    # its items belong in top-k so lower bytes cannot affect the selected set.
    if bin_count != rank:
        prefix, rank, bin_count = _tiled_radix_select_pass(
            scores_ptr,
            score_base,
            sparse_key_blocks,
            prefix,
            rank,
            tl.constexpr(16),
            selector_tile,
            force_row,
            has_force,
        )
        resolved_shift = tl.full((), 16, tl.uint32)
        if bin_count != rank:
            prefix, rank, bin_count = _tiled_radix_select_pass(
                scores_ptr,
                score_base,
                sparse_key_blocks,
                prefix,
                rank,
                tl.constexpr(8),
                selector_tile,
                force_row,
                has_force,
            )
            resolved_shift = tl.full((), 8, tl.uint32)
            if bin_count != rank:
                prefix, rank, bin_count = _tiled_radix_select_pass(
                    scores_ptr,
                    score_base,
                    sparse_key_blocks,
                    prefix,
                    rank,
                    tl.constexpr(0),
                    selector_tile,
                    force_row,
                    has_force,
                )
                resolved_shift = tl.full((), 0, tl.uint32)

    output_offset = 0
    equal_offset = 0
    for tile_start in range(0, sparse_key_blocks, selector_tile):
        key_offsets = tile_start + tl.arange(0, selector_tile)
        valid = key_offsets < sparse_key_blocks
        scores = tl.load(scores_ptr + score_base + key_offsets, mask=valid, other=0.0)
        ordered_bits = _route_priority_bits(scores, key_offsets, valid, force_row, has_force)
        shifted = ordered_bits >> resolved_shift
        if resolved_shift == 0:
            equal = valid & (shifted == prefix)
            equal_rank = equal_offset + tl.cumsum(equal.to(tl.int32), axis=0)
            selected = valid & ((shifted > prefix) | (equal & (equal_rank <= rank)))
            equal_offset += tl.sum(equal.to(tl.int32), axis=0)  # pyright: ignore[reportOperatorIssue]
        else:
            selected = valid & (shifted >= prefix)
        local_output_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
        tl.store(
            routes_ptr + output_base + (output_offset + local_output_rank) * route_route_stride,
            key_offsets,
            mask=selected,
        )
        output_offset += tl.sum(selected.to(tl.int32), axis=0)  # pyright: ignore[reportOperatorIssue]


@triton.jit(do_not_specialize=_RADIX_DO_NOT_SPECIALIZE)
def _single_load_radix_select_packed_routes_kernel(  # noqa: PLR0913, PLR0917
    scores_ptr: torch.Tensor,
    routes_ptr: torch.Tensor,
    head_keep_blocks_ptr: torch.Tensor,
    route_head_offsets_ptr: torch.Tensor,
    force_mask_ptr: torch.Tensor,
    sparse_key_blocks: int,
    query_block_offset: int,
    score_batch_stride: int,
    score_head_stride: int,
    score_query_stride: int,
    route_batch_stride: int,
    route_query_stride: int,
    route_route_stride: int,
    force_mask_stride: int,
    block_k: tl.constexpr,
    has_force: tl.constexpr,
) -> None:
    """Select exact stable FP32 top-k routes from one register-resident score load."""
    score_base, output_base, rank, force_row = _resolve_route_row(
        head_keep_blocks_ptr,
        route_head_offsets_ptr,
        force_mask_ptr,
        query_block_offset,
        score_batch_stride,
        score_head_stride,
        score_query_stride,
        route_batch_stride,
        route_query_stride,
        route_route_stride,
        force_mask_stride,
    )

    key_offsets = tl.arange(0, block_k)
    valid = key_offsets < sparse_key_blocks
    scores = tl.load(scores_ptr + score_base + key_offsets, mask=valid, other=0.0)
    ordered_bits = _route_priority_bits(scores, key_offsets, valid, force_row, has_force)

    prefix = tl.full((), 0, tl.uint32)
    resolved_shift = tl.full((), 24, tl.uint32)
    prefix, rank, bin_count = _in_register_radix_select_pass(
        ordered_bits,
        valid,
        prefix,
        rank,
        tl.constexpr(24),
    )
    if bin_count != rank:
        prefix, rank, bin_count = _in_register_radix_select_pass(
            ordered_bits,
            valid,
            prefix,
            rank,
            tl.constexpr(16),
        )
        resolved_shift = tl.full((), 16, tl.uint32)
        if bin_count != rank:
            prefix, rank, bin_count = _in_register_radix_select_pass(
                ordered_bits,
                valid,
                prefix,
                rank,
                tl.constexpr(8),
            )
            resolved_shift = tl.full((), 8, tl.uint32)
            if bin_count != rank:
                prefix, rank, bin_count = _in_register_radix_select_pass(
                    ordered_bits,
                    valid,
                    prefix,
                    rank,
                    tl.constexpr(0),
                )
                resolved_shift = tl.full((), 0, tl.uint32)

    shifted = ordered_bits >> resolved_shift
    if resolved_shift == 0:
        equal = valid & (shifted == prefix)
        equal_rank = tl.cumsum(equal.to(tl.int32), axis=0)
        selected = valid & ((shifted > prefix) | (equal & (equal_rank <= rank)))
    else:
        selected = valid & (shifted >= prefix)
    local_output_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    tl.store(
        routes_ptr + output_base + local_output_rank * route_route_stride,
        key_offsets,
        mask=selected,
    )


def tiled_radix_select_packed_routes(
    scores: torch.Tensor,
    routes: torch.Tensor,
    head_keep_blocks: torch.Tensor,
    route_head_offsets: torch.Tensor,
    *,
    query_block_offset: int,
    force_tile_mask: torch.Tensor | None = None,
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
    if force_tile_mask is not None and (
        force_tile_mask.shape != (routes.shape[1], triton.cdiv(sparse_key_blocks, 32))
        or force_tile_mask.dtype is not torch.int32
        or force_tile_mask.device != scores.device
        or not force_tile_mask.is_contiguous()
    ):
        raise ValueError("forced membership must use contiguous INT32 [query_blocks,key_words]")
    common_args = (
        scores,
        routes,
        head_keep_blocks,
        route_head_offsets,
        head_keep_blocks if force_tile_mask is None else force_tile_mask,
        sparse_key_blocks,
        query_block_offset,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        routes.stride(0),
        routes.stride(1),
        routes.stride(2),
        0 if force_tile_mask is None else force_tile_mask.stride(0),
    )
    has_force = force_tile_mask is not None
    single_load_block_k = _select_single_load_block_k(sparse_key_blocks)
    with device_context(scores.device):
        if single_load_block_k is not None:
            _single_load_radix_select_packed_routes_kernel[(query_blocks, heads, batch)](
                *common_args,
                block_k=single_load_block_k,
                has_force=has_force,
                num_warps=4,
                num_stages=1,
            )
        else:
            _tiled_radix_select_packed_routes_kernel[(query_blocks, heads, batch)](
                *common_args,
                selector_tile=_SELECTOR_TILE,
                has_force=has_force,
                num_warps=4,
                num_stages=1,
            )


__all__ = ["tiled_radix_select_packed_routes"]
