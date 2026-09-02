"""FP32 mean-pool routing for sparse Piper Attention."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from ._block_layout import validate_block_lengths
from ._budget import _ResolvedRouteLayout
from ._routes import (
    _MEAN_POOL_ROUTING,
    PackedRouteAndCoarseBuilder,
    PackedRouteBuilder,
    PackedRoutes,
    PackedRoutesAndCoarseOutput,
)
from .coarse import (
    _apply_chunked_coarse_residual,
    _mean_pool_head_major_blocks,
    _mean_pool_token_blocks,
    _preserve_coarse_residual_in_graph,
    validate_coarse_residual_inputs,
    validate_coarse_scale,
)

_QUERY_CHUNK_BLOCKS = 384


def _mean_pool_coarse_residual_impl(
    fine_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compression_gate: torch.Tensor,
    *,
    sparse_key_blocks: int,
    coarse_key_blocks: int,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    coarse_key_blocks = validate_coarse_residual_inputs(
        fine_output,
        query,
        key,
        value,
        compression_gate,
        sparse_key_blocks,
        coarse_key_blocks,
        coarse_scale,
        block_lengths,
        routing_label="mean-pool",
    )

    query_mean = _mean_pool_token_blocks(query, block_lengths)
    key_mean = _mean_pool_token_blocks(key, block_lengths)[:, :, :coarse_key_blocks]
    pooled_value = _mean_pool_token_blocks(value, block_lengths)[:, :, :coarse_key_blocks]
    return _apply_chunked_coarse_residual(
        fine_output,
        pooled_value,
        compression_gate,
        _mean_pool_score_chunks(
            query_mean,
            key_mean,
            score_scale=coarse_scale,
        ),
    )


def mean_pool_coarse_residual(
    fine_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compression_gate: torch.Tensor,
    *,
    sparse_key_blocks: int,
    coarse_key_blocks: int | None = None,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a mean-pool coarse branch over a K64 prefix.

    All token tensors use ``[batch,sequence,heads,features]``. Compact storage
    may end in one partial block. With ``block_lengths``, storage instead holds
    complete physical K64 blocks whose valid rows occupy each block's prefix.
    ``coarse_key_blocks`` defaults to ``sparse_key_blocks`` for compatibility
    and may extend the coarse branch across the following dense-key blocks.
    Coarse scores are computed in bounded query-block chunks and the caller's
    compression gate is applied directly without an implicit activation.
    Compatible compiled inference graphs may fuse this with sparse attention.
    """
    resolved_coarse_key_blocks = (
        sparse_key_blocks if coarse_key_blocks is None else coarse_key_blocks
    )
    if _preserve_coarse_residual_in_graph(fine_output, query, key, value, compression_gate):
        return torch.ops.piper_kernels.sparse_piper_coarse_residual.default(
            fine_output,
            query,
            key,
            value,
            compression_gate,
            sparse_key_blocks,
            resolved_coarse_key_blocks,
            coarse_scale,
            _MEAN_POOL_ROUTING,
            block_lengths,
        )
    return _mean_pool_coarse_residual_impl(
        fine_output,
        query,
        key,
        value,
        compression_gate,
        sparse_key_blocks=sparse_key_blocks,
        coarse_key_blocks=resolved_coarse_key_blocks,
        coarse_scale=coarse_scale,
        block_lengths=block_lengths,
    )


def packed_mean_pool_routes_from_summaries(
    query_mean: torch.Tensor,
    key_mean: torch.Tensor,
    layout: _ResolvedRouteLayout,
) -> PackedRoutes:
    """Select routes from existing FP32 Q64/K64 valid-prefix means."""
    _validate_mean_summaries(query_mean, key_mean)
    batch, heads, query_blocks, _head_dim = query_mean.shape
    sparse_key_blocks = key_mean.shape[2]
    builder = PackedRouteBuilder(
        layout,
        batch=batch,
        heads=heads,
        query_blocks=query_blocks,
        sparse_key_blocks=sparse_key_blocks,
        device=query_mean.device,
    )
    for start, scores in _mean_pool_score_chunks(query_mean, key_mean):
        builder.write(scores, query_block_offset=start)
    return builder.routes


def packed_mean_pool_routes_and_coarse_from_summaries(
    query_mean: torch.Tensor,
    key_mean: torch.Tensor,
    pooled_value: torch.Tensor,
    layout: _ResolvedRouteLayout,
    *,
    sparse_key_blocks: int,
    coarse_scale: float,
) -> PackedRoutesAndCoarseOutput:
    """Route over a sparse prefix and attend coarsely over every supplied K block."""
    _validate_mean_summaries(query_mean, key_mean)
    validate_coarse_scale(coarse_scale)
    batch, heads, query_blocks, _head_dim = query_mean.shape
    builder = PackedRouteAndCoarseBuilder(
        layout,
        pooled_value,
        batch=batch,
        heads=heads,
        query_blocks=query_blocks,
        sparse_key_blocks=sparse_key_blocks,
        device=query_mean.device,
    )
    for start, scores in _mean_pool_score_chunks(
        query_mean,
        key_mean,
        score_scale=coarse_scale,
    ):
        builder.write(scores, query_block_offset=start)
    return builder.finish()


def packed_mean_pool_routes_from_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: _ResolvedRouteLayout,
    block_lengths: torch.Tensor | None = None,
) -> PackedRoutes:
    """Route compact or valid-front padded Q and sparse-prefix K64 blocks."""
    _validate_mean_sequences(query, key, block_lengths)
    query_mean = _sequence_block_means(query, block_lengths)
    key_mean = _sequence_block_means(
        key,
        None if block_lengths is None else block_lengths[: key.shape[2] // _BLOCK_ROWS],
    )
    return packed_mean_pool_routes_from_summaries(query_mean, key_mean, layout)


def _sequence_block_means(
    sequence: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    return _mean_pool_head_major_blocks(sequence, block_lengths)


def _mean_pool_score_chunks(
    query_mean: torch.Tensor,
    key_mean: torch.Tensor,
    *,
    score_scale: float | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    batch, heads, query_blocks, head_dim = query_mean.shape
    key_blocks = key_mean.shape[2]
    flat_key = key_mean.reshape(batch * heads, key_blocks, head_dim).transpose(1, 2)
    for start in range(0, query_blocks, _QUERY_CHUNK_BLOCKS):
        stop = min(start + _QUERY_CHUNK_BLOCKS, query_blocks)
        flat_query = query_mean[:, :, start:stop].reshape(
            batch * heads,
            stop - start,
            head_dim,
        )
        scores = (
            torch.bmm(flat_query, flat_key)
            if score_scale is None
            else torch.baddbmm(
                flat_query[:, :1, :1],
                flat_query,
                flat_key,
                beta=0,
                alpha=score_scale,
            )
        )
        yield (
            start,
            scores.reshape(batch, heads, stop - start, key_blocks),
        )


def _validate_mean_summaries(query_mean: torch.Tensor, key_mean: torch.Tensor) -> None:
    if query_mean.ndim != 4 or key_mean.ndim != 4:
        raise ValueError("mean-pool summaries must use rank-four Q/K tensors")
    if query_mean.shape[:2] != key_mean.shape[:2] or query_mean.shape[-1] != key_mean.shape[-1]:
        raise ValueError("mean-pool summary batch/head/feature dimensions must match")
    if query_mean.shape[2] < 1 or key_mean.shape[2] < 1:
        raise ValueError("mean-pool summaries must contain query and key blocks")
    if query_mean.dtype is not torch.float32 or key_mean.dtype is not torch.float32:
        raise ValueError("mean-pool summaries must use FP32")
    if query_mean.device != key_mean.device:
        raise ValueError("mean-pool summaries must share a device")
    if not query_mean.is_contiguous():
        raise ValueError("mean-pool query summaries must be contiguous")
    if key_mean.stride(-1) != 1 or key_mean.stride(-2) != key_mean.shape[-1]:
        raise ValueError("mean-pool key summaries must have contiguous block features")


def _validate_mean_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> None:
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("mean-pool query and key sequences must be rank-four tensors")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("mean-pool query and key batch/head/feature dimensions must match")
    if query.shape[2] < 1 or key.shape[2] < _BLOCK_ROWS:
        raise ValueError("mean-pool routing requires nonempty Q and at least one K64 block")
    if query.device != key.device:
        raise ValueError("mean-pool query and key sequences must share a device")
    if not query.is_floating_point() or not key.is_floating_point():
        raise TypeError("mean-pool query and key sequences must be floating-point tensors")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("mean-pool query and key features must be contiguous")
    if key.shape[2] % _BLOCK_ROWS:
        raise ValueError("mean-pool sparse keys require complete K64 blocks")
    if block_lengths is not None:
        block_count = validate_block_lengths(
            block_lengths,
            sequence_length=query.shape[2],
            device=query.device,
            context="padded mean-pool routing",
            require_contiguous=True,
        )
        if key.shape[2] // _BLOCK_ROWS > block_count:
            raise ValueError("padded mean-pool keys cannot exceed the block-length layout")


__all__ = [
    "mean_pool_coarse_residual",
    "packed_mean_pool_routes_and_coarse_from_summaries",
    "packed_mean_pool_routes_from_sequences",
    "packed_mean_pool_routes_from_summaries",
]
