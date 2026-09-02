"""FP32 mean-pool routing for sparse Piper Attention."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ._budget import _ResolvedRouteLayout
from ._routes import (
    PackedRouteAndCoarseBuilder,
    PackedRouteBuilder,
    PackedRoutes,
    PackedRoutesAndCoarseOutput,
)

_BLOCK_ROWS = 64
_QUERY_CHUNK_BLOCKS = 384


def packed_mean_pool_routes_from_summaries(
    query_mean: torch.Tensor,
    key_mean: torch.Tensor,
    layout: _ResolvedRouteLayout,
) -> PackedRoutes:
    """Select routes from existing FP32 Q64/K64 valid-prefix means."""
    _validate_mean_summaries(query_mean, key_mean)
    batch, heads, query_blocks, _head_dim = query_mean.shape
    key_blocks = key_mean.shape[2]
    builder = PackedRouteBuilder(
        layout,
        batch=batch,
        heads=heads,
        query_blocks=query_blocks,
        key_blocks=key_blocks,
        device=query_mean.device,
    )
    for start, scores in _mean_pool_score_chunks(query_mean, key_mean):
        builder.write(scores, route_query_offset=start)
    return builder.routes


def packed_mean_pool_routes_and_coarse_from_summaries(
    query_mean: torch.Tensor,
    key_mean: torch.Tensor,
    pooled_value: torch.Tensor,
    layout: _ResolvedRouteLayout,
    *,
    coarse_scale: float,
) -> PackedRoutesAndCoarseOutput:
    """Select fine routes and apply coarse attention from each mean-score chunk."""
    _validate_mean_summaries(query_mean, key_mean)
    batch, heads, query_blocks, _head_dim = query_mean.shape
    key_blocks = key_mean.shape[2]
    builder = PackedRouteAndCoarseBuilder(
        layout,
        pooled_value,
        batch=batch,
        heads=heads,
        query_blocks=query_blocks,
        key_blocks=key_blocks,
        device=query_mean.device,
        coarse_scale=coarse_scale,
    )
    for start, scores in _mean_pool_score_chunks(query_mean, key_mean):
        builder.write(scores, route_query_offset=start)
    return builder.finish()


def packed_mean_pool_routes_from_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: _ResolvedRouteLayout,
) -> PackedRoutes:
    """Portable eager fallback for compact Q and complete sparse-prefix K64 blocks."""
    _validate_mean_sequences(query, key)
    query_mean = _sequence_block_means(query)
    key_mean = _sequence_block_means(key)
    return packed_mean_pool_routes_from_summaries(query_mean, key_mean, layout)


def _sequence_block_means(
    sequence: torch.Tensor,
) -> torch.Tensor:
    rows = sequence.shape[2]
    full_rows = rows // _BLOCK_ROWS * _BLOCK_ROWS
    summaries = []
    if full_rows:
        blocks = sequence[:, :, :full_rows].unflatten(
            2,
            (full_rows // _BLOCK_ROWS, _BLOCK_ROWS),
        )
        summaries.append(blocks.float().mean(dim=3))
    if full_rows != rows:
        summaries.append(sequence[:, :, full_rows:].float().mean(dim=2, keepdim=True))
    return summaries[0] if len(summaries) == 1 else torch.cat(summaries, dim=2)


def _mean_pool_score_chunks(
    query_mean: torch.Tensor,
    key_mean: torch.Tensor,
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
        yield (
            start,
            torch.bmm(flat_query, flat_key).reshape(
                batch,
                heads,
                stop - start,
                key_blocks,
            ),
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


__all__ = [
    "packed_mean_pool_routes_and_coarse_from_summaries",
    "packed_mean_pool_routes_from_sequences",
    "packed_mean_pool_routes_from_summaries",
]
