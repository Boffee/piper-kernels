"""Shared summary-to-route and summary-to-coarse orchestration."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from . import _backend
from ._budget import _ResolvedRouteLayout
from ._contracts import (
    _MEAN_ROUTING,
    validate_routing_mode,
)
from ._routes import (
    PackedRouteAndCoarseBuilder,
    PackedRouteBuilder,
    PackedRoutes,
    PackedRoutesAndCoarseOutput,
)
from ._summaries import _validate_sequences, sequence_block_summaries
from .coarse import validate_coarse_scale

_QUERY_CHUNK_BLOCKS = 384


def packed_routes_from_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: _ResolvedRouteLayout,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    *,
    skip_dense_routing: bool = False,
    force_tile_mask: torch.Tensor | None = None,
    validate: bool = True,
) -> PackedRoutes:
    """Select routes for compact or valid-front padded Q and K64 sequences."""
    if validate and key.ndim == 4 and (key.shape[2] < _BLOCK_ROWS or key.shape[2] % _BLOCK_ROWS):
        raise ValueError("sparse routing requires complete sparse-prefix K64 blocks")
    if key.ndim == 4 and layout.keeps_all_blocks(key.shape[2] // _BLOCK_ROWS):
        if validate:
            validate_routing_mode(routing_mode)
            _validate_sequences(query, key, block_lengths)
        return PackedRouteBuilder(
            layout,
            batch=query.shape[0],
            heads=query.shape[1],
            query_blocks=(query.shape[2] + _BLOCK_ROWS - 1) // _BLOCK_ROWS,
            sparse_key_blocks=key.shape[2] // _BLOCK_ROWS,
            device=query.device,
            skip_dense_routing=skip_dense_routing,
            force_tile_mask=force_tile_mask,
        ).routes
    query_summary, key_primary, key_aux = sequence_block_summaries(
        query,
        key,
        routing_mode,
        block_lengths,
        validate=validate,
    )
    return _packed_routes_from_validated_summaries(
        query_summary,
        key_primary,
        key_aux,
        layout,
        routing_mode,
        skip_dense_routing=skip_dense_routing,
        force_tile_mask=force_tile_mask,
    )


def _packed_routes_from_validated_summaries(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    layout: _ResolvedRouteLayout,
    routing_mode: int,
    *,
    skip_dense_routing: bool = False,
    force_tile_mask: torch.Tensor | None = None,
) -> PackedRoutes:
    """Select routes from already-validated Q/K summaries."""
    batch, heads, query_blocks, _head_dim = query_summary.shape
    builder = PackedRouteBuilder(
        layout,
        batch=batch,
        heads=heads,
        query_blocks=query_blocks,
        sparse_key_blocks=key_primary.shape[2],
        device=query_summary.device,
        skip_dense_routing=skip_dense_routing,
        force_tile_mask=force_tile_mask,
    )
    if layout.keeps_all_blocks(key_primary.shape[2]):
        return builder.routes
    for start, scores in score_chunks(
        query_summary,
        key_primary,
        key_aux,
        routing_mode,
    ):
        builder.write(scores, query_block_offset=start)
    return builder.routes


def packed_routes_from_summaries(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    layout: _ResolvedRouteLayout,
    routing_mode: int,
    *,
    skip_dense_routing: bool = False,
    force_tile_mask: torch.Tensor | None = None,
) -> PackedRoutes:
    """Select routes through the fixed policy-independent summary contract."""
    _validate_summaries(query_summary, key_primary, key_aux, routing_mode)
    return _packed_routes_from_validated_summaries(
        query_summary,
        key_primary,
        key_aux,
        layout,
        routing_mode,
        skip_dense_routing=skip_dense_routing,
        force_tile_mask=force_tile_mask,
    )


def packed_routes_and_coarse_from_summaries(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    pooled_value: torch.Tensor,
    layout: _ResolvedRouteLayout,
    *,
    sparse_key_blocks: int,
    coarse_scale: float,
    routing_mode: int,
    skip_dense_routing: bool = False,
    force_tile_mask: torch.Tensor | None = None,
) -> PackedRoutesAndCoarseOutput:
    """Route sparsely and attend coarsely from the same summary score chunks."""
    _validate_summaries(query_summary, key_primary, key_aux, routing_mode)
    validate_coarse_scale(coarse_scale)
    batch, heads, query_blocks, _head_dim = query_summary.shape
    builder = PackedRouteAndCoarseBuilder(
        layout,
        pooled_value,
        batch=batch,
        heads=heads,
        query_blocks=query_blocks,
        sparse_key_blocks=sparse_key_blocks,
        device=query_summary.device,
        skip_dense_routing=skip_dense_routing,
        force_tile_mask=force_tile_mask,
    )
    for start, scores in score_chunks(
        query_summary,
        key_primary,
        key_aux,
        routing_mode,
        score_scale=coarse_scale,
    ):
        builder.write(scores, query_block_offset=start)
    return builder.finish()


def routing_scores(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    routing_mode: int,
    *,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Contract one validated summary chunk using the selected routing policy."""
    if routing_mode != _MEAN_ROUTING:
        score = _backend.select_minmax_scores(query_summary, key_primary, key_aux)
        if score is not None:
            return score(query_summary, key_primary, key_aux, score_scale=score_scale)
    batch, heads, query_blocks, head_dim = query_summary.shape
    key_blocks = key_primary.shape[2]
    kv_heads = key_primary.shape[1]
    flat_query = query_summary.reshape(
        batch * kv_heads, (heads // kv_heads) * query_blocks, head_dim
    )
    flat_key_primary = key_primary.reshape(batch * kv_heads, key_blocks, head_dim)
    if score_scale is None:
        scores = torch.bmm(flat_query, flat_key_primary.transpose(1, 2))
    else:
        scores = torch.baddbmm(
            flat_query[:, :1, :1],
            flat_query,
            flat_key_primary.transpose(1, 2),
            beta=0,
            alpha=score_scale,
        )
    if routing_mode != _MEAN_ROUTING:
        flat_key_aux = key_aux.reshape(batch * kv_heads, key_blocks, head_dim)
        if score_scale is None:
            auxiliary_scores = torch.bmm(flat_query, flat_key_aux.transpose(1, 2))
        else:
            auxiliary_scores = torch.baddbmm(
                flat_query[:, :1, :1],
                flat_query,
                flat_key_aux.transpose(1, 2),
                beta=0,
                alpha=score_scale,
            )
        if scores.requires_grad or auxiliary_scores.requires_grad:
            scores = torch.maximum(scores, auxiliary_scores)
        else:
            torch.maximum(scores, auxiliary_scores, out=scores)
    return scores.reshape(batch, heads, query_blocks, key_blocks)


def score_chunks(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    routing_mode: int,
    *,
    score_scale: float | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield bounded query-block score chunks for either routing policy."""
    for start in range(0, query_summary.shape[2], _QUERY_CHUNK_BLOCKS):
        stop = min(start + _QUERY_CHUNK_BLOCKS, query_summary.shape[2])
        yield (
            start,
            routing_scores(
                query_summary[:, :, start:stop],
                key_primary,
                key_aux,
                routing_mode,
                score_scale=score_scale,
            ),
        )


def _validate_summaries(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    routing_mode: int,
) -> None:
    validate_routing_mode(routing_mode)
    if query_summary.ndim != 4 or key_primary.ndim != 4:
        raise ValueError("routing summaries must use rank-four Q/K tensors")
    if (
        query_summary.shape[0] != key_primary.shape[0]
        or key_primary.shape[1] < 1
        or query_summary.shape[1] < 1
        or query_summary.shape[1] % key_primary.shape[1]
        or query_summary.shape[-1] != key_primary.shape[-1]
    ):
        raise ValueError("routing summary batch/head/feature dimensions must match")
    if query_summary.shape[2] < 1 or key_primary.shape[2] < 1:
        raise ValueError("routing summaries must contain query and key blocks")
    if query_summary.dtype is not torch.float32 or key_primary.dtype is not torch.float32:
        raise ValueError("routing summaries must use FP32")
    if query_summary.device != key_primary.device:
        raise ValueError("routing summaries must share a device")
    if not query_summary.is_contiguous():
        raise ValueError("routing query summaries must be contiguous")
    if key_primary.stride(-1) != 1 or key_primary.stride(-2) != key_primary.shape[-1]:
        raise ValueError("routing key summaries must have contiguous block features")
    if key_aux.dtype is not torch.float32 or key_aux.device != query_summary.device:
        raise ValueError("routing auxiliary summaries must be FP32 on the summary device")
    if routing_mode == _MEAN_ROUTING:
        expected_aux_shape = (*key_primary.shape[:2], 0, key_primary.shape[-1])
        if key_aux.shape != expected_aux_shape:
            raise ValueError("mean-pool routing requires a block-empty auxiliary summary")
    elif key_aux.shape != key_primary.shape or (
        key_aux.stride(-1) != 1 or key_aux.stride(-2) != key_aux.shape[-1]
    ):
        raise ValueError("minmax-pool auxiliary summaries must match primary key summaries")


__all__ = [
    "packed_routes_and_coarse_from_summaries",
    "packed_routes_from_sequences",
    "packed_routes_from_summaries",
    "routing_scores",
    "score_chunks",
]
