"""Exact DSA routing for sparse Piper Attention."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._budget import _ResolvedRouteLayout
from ._routes import (
    PackedRouteAndCoarseBuilder,
    PackedRouteBuilder,
    PackedRoutes,
    PackedRoutesAndCoarseOutput,
)

try:
    from .dsa_triton import sequence_block_summaries as _sm120_sequence_block_summaries
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("triton"):
        raise
    _sm120_sequence_block_summaries = None

_QUERY_CHUNK_BLOCKS = 384
_BLOCK_ROWS = 64


def packed_dsa_routes_from_summaries(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    layout: _ResolvedRouteLayout,
) -> PackedRoutes:
    """Select routes from existing exact Q64/K64 extrema summaries."""
    _validate_dsa_summaries(query_summary, key_max, key_min)
    heads = query_summary.shape[1]
    key_block_count = key_max.shape[2]
    builder = PackedRouteBuilder(
        layout,
        batch=query_summary.shape[0],
        heads=heads,
        query_blocks=query_summary.shape[2],
        key_blocks=key_block_count,
        device=query_summary.device,
    )
    for start, scores in _dsa_score_chunks(query_summary, key_max, key_min):
        builder.write(scores, route_query_offset=start)
    return builder.routes


def packed_dsa_routes_and_coarse_from_summaries(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    pooled_value: torch.Tensor,
    layout: _ResolvedRouteLayout,
    *,
    coarse_scale: float,
) -> PackedRoutesAndCoarseOutput:
    """Select fine routes and apply coarse attention from each DSA-score chunk."""
    _validate_dsa_summaries(query_summary, key_max, key_min)
    batch, heads, query_blocks, _head_dim = query_summary.shape
    key_blocks = key_max.shape[2]
    builder = PackedRouteAndCoarseBuilder(
        layout,
        pooled_value,
        batch=batch,
        heads=heads,
        query_blocks=query_blocks,
        key_blocks=key_blocks,
        device=query_summary.device,
        coarse_scale=coarse_scale,
    )
    for start, scores in _dsa_score_chunks(query_summary, key_max, key_min):
        builder.write(scores, route_query_offset=start)
    return builder.finish()


def packed_dsa_routes_from_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: _ResolvedRouteLayout,
) -> PackedRoutes:
    """Select routes for a ragged Q sequence and complete sparse-prefix K64 blocks."""
    _validate_dsa_sequences(query, key)
    query_summary, key_max, key_min = _sequence_block_summaries(query, key)
    return packed_dsa_routes_from_summaries(query_summary, key_max, key_min, layout)


def _validate_dsa_summaries(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
) -> None:
    if query_summary.ndim != 4 or key_max.ndim != 4 or key_min.shape != key_max.shape:
        raise ValueError("DSA summaries must use rank-four Q/max/min tensors")
    if query_summary.shape[:2] != key_max.shape[:2]:
        raise ValueError("DSA summary batch/head dimensions must match")
    if query_summary.shape[-1] != key_max.shape[-1]:
        raise ValueError("DSA summary feature dimensions must match")
    if query_summary.shape[2] < 1 or key_max.shape[2] < 1:
        raise ValueError("DSA summaries must contain query and key blocks")
    if query_summary.dtype is not torch.float32 or any(
        summary.dtype is not query_summary.dtype for summary in (key_max, key_min)
    ):
        raise ValueError("DSA summaries must use FP32")
    if any(summary.device != query_summary.device for summary in (key_max, key_min)):
        raise ValueError("DSA summaries must share a device")
    if not query_summary.is_contiguous():
        raise ValueError("DSA query summaries must be contiguous")
    if any(
        summary.stride(-1) != 1 or summary.stride(-2) != summary.shape[-1]
        for summary in (key_max, key_min)
    ):
        raise ValueError("DSA key summaries must have contiguous block features")


def _sequence_block_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _supports_sm120_sequence_summaries(query, key):
        assert _sm120_sequence_block_summaries is not None
        return _sm120_sequence_block_summaries(query, key)
    query_summaries = []
    for start in range(0, query.shape[2], _BLOCK_ROWS):
        block = query[:, :, start : start + _BLOCK_ROWS].float()
        query_summaries.append(block.amax(dim=2) + block.amin(dim=2))
    query_summary = torch.stack(query_summaries, dim=2)
    key_blocks = key.unflatten(2, (key.shape[2] // _BLOCK_ROWS, _BLOCK_ROWS)).float()
    return query_summary, key_blocks.amax(dim=3), key_blocks.amin(dim=3)


def _dsa_scores(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
) -> torch.Tensor:
    """Contract exact FP32 scores while keeping only two score buffers."""
    batch, heads, query_count, head_dim = query_summary.shape
    key_count = key_max.shape[2]
    flat_query = query_summary.reshape(batch * heads, query_count, head_dim)
    flat_key_max = key_max.reshape(batch * heads, key_count, head_dim)
    flat_key_min = key_min.reshape(batch * heads, key_count, head_dim)
    scores = torch.bmm(flat_query, flat_key_max.transpose(1, 2))
    minimum_scores = torch.bmm(flat_query, flat_key_min.transpose(1, 2))
    torch.maximum(scores, minimum_scores, out=scores)
    return scores.reshape(batch, heads, query_count, key_count)


def _dsa_score_chunks(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
) -> Iterator[tuple[int, torch.Tensor]]:
    for start in range(0, query_summary.shape[2], _QUERY_CHUNK_BLOCKS):
        stop = min(start + _QUERY_CHUNK_BLOCKS, query_summary.shape[2])
        yield start, _dsa_scores(query_summary[:, :, start:stop], key_max, key_min)


def _supports_sm120_sequence_summaries(query: torch.Tensor, key: torch.Tensor) -> bool:
    target = AcceleratorTarget.from_device(query.device)
    return (
        _sm120_sequence_block_summaries is not None
        and target.is_cuda_capability(12, 0)
        and query.shape[-1] == 128
        and key.shape[-1] == 128
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype
    )


def _validate_dsa_sequences(query: torch.Tensor, key: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("DSA query and key sequences must be rank-four tensors")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("DSA query and key batch/head/feature dimensions must match")
    if query.shape[2] < 1 or key.shape[2] < _BLOCK_ROWS or key.shape[2] % _BLOCK_ROWS:
        raise ValueError("DSA requires a nonempty query and complete sparse-prefix K64 blocks")
    if query.device != key.device:
        raise ValueError("DSA query and key sequences must share a device")
    if not query.is_floating_point() or not key.is_floating_point():
        raise TypeError("DSA query and key sequences must be floating-point tensors")
