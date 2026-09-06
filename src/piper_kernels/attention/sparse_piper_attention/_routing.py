"""Shared summary-to-route and summary-to-coarse orchestration."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from ._budget import _ResolvedRouteLayout
from ._routes import (
    PackedRouteAndCoarseBuilder,
    PackedRouteBuilder,
    PackedRoutes,
    PackedRoutesAndCoarseOutput,
)
from ._routing_modes import (
    _MEAN_ROUTING,
    _ROUTING_NAME_BY_MODE,
    validate_routing_mode,
)
from ._summaries import sequence_block_summaries
from .coarse import (
    _apply_chunked_coarse_residual,
    _mean_pool_token_blocks,
    _preserve_coarse_residual_in_graph,
    validate_coarse_residual_inputs,
    validate_coarse_scale,
)

_QUERY_CHUNK_BLOCKS = 384


def _coarse_residual_from_mode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coarse_gate: torch.Tensor,
    *,
    coarse_key_blocks: int | None,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one policy-selected coarse branch while preserving its graph boundary."""
    validate_routing_mode(routing_mode)
    if _preserve_coarse_residual_in_graph(query, key, value, coarse_gate):
        resolved_coarse_key_blocks = (
            (
                block_lengths.numel()
                if block_lengths is not None
                else (query.shape[1] + _BLOCK_ROWS - 1) // _BLOCK_ROWS
            )
            if coarse_key_blocks is None
            else coarse_key_blocks
        )
        return torch.ops.piper_kernels.sparse_piper_coarse_residual.default(
            query,
            key,
            value,
            coarse_gate,
            resolved_coarse_key_blocks,
            coarse_scale,
            routing_mode,
            block_lengths,
        )
    return coarse_residual_impl(
        query,
        key,
        value,
        coarse_gate,
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        routing_mode=routing_mode,
        block_lengths=block_lengths,
    )


def coarse_residual_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coarse_gate: torch.Tensor,
    *,
    coarse_key_blocks: int | None,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the common Q/K/V-derived coarse residual implementation."""
    validate_routing_mode(routing_mode)
    coarse_key_blocks = validate_coarse_residual_inputs(
        query,
        key,
        value,
        coarse_gate,
        coarse_key_blocks,
        coarse_scale,
        block_lengths,
        routing_label=_ROUTING_NAME_BY_MODE[routing_mode],
    )
    query_head_major = query.transpose(1, 2)
    coarse_key = key.transpose(1, 2)[:, :, : coarse_key_blocks * _BLOCK_ROWS]
    query_summary, key_primary, key_aux = sequence_block_summaries(
        query_head_major,
        coarse_key,
        routing_mode,
        block_lengths,
    )
    pooled_value = _mean_pool_token_blocks(value, block_lengths)[:, :, :coarse_key_blocks]
    return _apply_chunked_coarse_residual(
        pooled_value,
        coarse_gate,
        score_chunks(
            query_summary,
            key_primary,
            key_aux,
            routing_mode,
            score_scale=coarse_scale,
        ),
    )


def packed_routes_from_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: _ResolvedRouteLayout,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> PackedRoutes:
    """Select routes for compact or valid-front padded Q and K64 sequences."""
    if key.ndim == 4 and (key.shape[2] < _BLOCK_ROWS or key.shape[2] % _BLOCK_ROWS):
        raise ValueError("sparse routing requires complete sparse-prefix K64 blocks")
    query_summary, key_primary, key_aux = sequence_block_summaries(
        query,
        key,
        routing_mode,
        block_lengths,
    )
    return packed_routes_from_summaries(
        query_summary,
        key_primary,
        key_aux,
        layout,
        routing_mode,
    )


def packed_routes_from_summaries(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    layout: _ResolvedRouteLayout,
    routing_mode: int,
) -> PackedRoutes:
    """Select routes through the fixed policy-independent summary contract."""
    _validate_summaries(query_summary, key_primary, key_aux, routing_mode)
    batch, heads, query_blocks, _head_dim = query_summary.shape
    builder = PackedRouteBuilder(
        layout,
        batch=batch,
        heads=heads,
        query_blocks=query_blocks,
        sparse_key_blocks=key_primary.shape[2],
        device=query_summary.device,
    )
    for start, scores in score_chunks(
        query_summary,
        key_primary,
        key_aux,
        routing_mode,
    ):
        builder.write(scores, query_block_offset=start)
    return builder.routes


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
    """Contract one summary chunk using the selected routing policy."""
    validate_routing_mode(routing_mode)
    batch, heads, query_blocks, head_dim = query_summary.shape
    key_blocks = key_primary.shape[2]
    flat_query = query_summary.reshape(batch * heads, query_blocks, head_dim)
    flat_key_primary = key_primary.reshape(batch * heads, key_blocks, head_dim)
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
        flat_key_aux = key_aux.reshape(batch * heads, key_blocks, head_dim)
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
        query_summary.shape[:2] != key_primary.shape[:2]
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
    "coarse_residual_impl",
    "packed_routes_and_coarse_from_summaries",
    "packed_routes_from_sequences",
    "packed_routes_from_summaries",
    "routing_scores",
    "score_chunks",
]
