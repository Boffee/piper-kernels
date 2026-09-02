"""Exact DSA routing for sparse Piper Attention."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from ._budget import _ResolvedRouteLayout
from ._routes import (
    _DSA_ROUTING,
    PackedRouteAndCoarseBuilder,
    PackedRouteBuilder,
    PackedRoutes,
    PackedRoutesAndCoarseOutput,
)
from .coarse import (
    _apply_chunked_coarse_residual,
    _mean_pool_token_blocks,
    _preserve_coarse_residual_in_graph,
    validate_coarse_residual_inputs,
    validate_coarse_scale,
)

try:
    from .dsa_triton import sequence_block_summaries as _sm120_sequence_block_summaries
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("triton"):
        raise
    _sm120_sequence_block_summaries = None

_QUERY_CHUNK_BLOCKS = 384


def _dsa_coarse_residual_impl(
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
        routing_label="DSA",
    )

    query_head_major = query.transpose(1, 2)
    coarse_key = key.transpose(1, 2)[:, :, : coarse_key_blocks * _BLOCK_ROWS]
    query_summary, key_max, key_min = _sequence_block_summaries(
        query_head_major,
        coarse_key,
        block_lengths,
    )
    pooled_value = _mean_pool_token_blocks(value, block_lengths)[:, :, :coarse_key_blocks]
    return _apply_chunked_coarse_residual(
        fine_output,
        pooled_value,
        compression_gate,
        _dsa_score_chunks(
            query_summary,
            key_max,
            key_min,
            score_scale=coarse_scale,
        ),
    )


def dsa_coarse_residual(
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
    """Add a DSA-score coarse branch over a K64 prefix.

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
            _DSA_ROUTING,
            block_lengths,
        )
    return _dsa_coarse_residual_impl(
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


def packed_dsa_routes_from_summaries(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    layout: _ResolvedRouteLayout,
) -> PackedRoutes:
    """Select routes from existing exact Q64/K64 extrema summaries."""
    _validate_dsa_summaries(query_summary, key_max, key_min)
    heads = query_summary.shape[1]
    sparse_key_blocks = key_max.shape[2]
    builder = PackedRouteBuilder(
        layout,
        batch=query_summary.shape[0],
        heads=heads,
        query_blocks=query_summary.shape[2],
        sparse_key_blocks=sparse_key_blocks,
        device=query_summary.device,
    )
    for start, scores in _dsa_score_chunks(query_summary, key_max, key_min):
        builder.write(scores, query_block_offset=start)
    return builder.routes


def packed_dsa_routes_and_coarse_from_summaries(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    pooled_value: torch.Tensor,
    layout: _ResolvedRouteLayout,
    *,
    sparse_key_blocks: int,
    coarse_scale: float,
) -> PackedRoutesAndCoarseOutput:
    """Route over a sparse prefix and attend coarsely over every supplied K block."""
    _validate_dsa_summaries(query_summary, key_max, key_min)
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
    for start, scores in _dsa_score_chunks(
        query_summary,
        key_max,
        key_min,
        score_scale=coarse_scale,
    ):
        builder.write(scores, query_block_offset=start)
    return builder.finish()


def packed_dsa_routes_from_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: _ResolvedRouteLayout,
    block_lengths: torch.Tensor | None = None,
) -> PackedRoutes:
    """Select routes for compact or valid-front padded Q and sparse-prefix K64."""
    _validate_dsa_sequences(query, key, block_lengths)
    if block_lengths is not None:
        torch._assert_async(
            torch.all((block_lengths >= 1) & (block_lengths <= _BLOCK_ROWS)),
            f"padded DSA block lengths must lie in [1, {_BLOCK_ROWS}]",
        )
    query_summary, key_max, key_min = _sequence_block_summaries(query, key, block_lengths)
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
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_lengths is not None:
        _validate_dsa_block_lengths(query, key, block_lengths)
    if _supports_sm120_sequence_summaries(query, key):
        assert _sm120_sequence_block_summaries is not None
        return _sm120_sequence_block_summaries(query, key, block_lengths)
    if block_lengths is not None:
        query_max, query_min = _padded_block_extrema(query, block_lengths)
        key_blocks = key.shape[2] // _BLOCK_ROWS
        key_max, key_min = _padded_block_extrema(key, block_lengths[:key_blocks])
        return query_max + query_min, key_max, key_min
    query_max, query_min = _compact_block_extrema(query)
    key_max, key_min = _compact_block_extrema(key)
    return query_max + query_min, key_max, key_min


def _compact_block_extrema(
    sequence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 extrema for compact blocks, including a ragged tail."""
    extrema = []
    for start in range(0, sequence.shape[2], _BLOCK_ROWS):
        block = sequence[:, :, start : start + _BLOCK_ROWS].float()
        extrema.append((block.amax(dim=2), block.amin(dim=2)))
    maxima, minima = zip(*extrema, strict=True)
    return torch.stack(maxima, dim=2), torch.stack(minima, dim=2)


def _padded_block_extrema(
    sequence: torch.Tensor,
    block_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = sequence.unflatten(2, (block_lengths.numel(), _BLOCK_ROWS)).float()
    valid_rows = torch.arange(_BLOCK_ROWS, device=sequence.device) < block_lengths[:, None]
    valid_rows = valid_rows[None, None, :, :, None]
    maximum = torch.where(valid_rows, blocks, -float("inf")).amax(dim=3)
    minimum = torch.where(valid_rows, blocks, float("inf")).amin(dim=3)
    return maximum, minimum


def _validate_dsa_block_lengths(
    query: torch.Tensor,
    key: torch.Tensor,
    block_lengths: torch.Tensor,
) -> None:
    query_rows = query.shape[2]
    if (
        query_rows % _BLOCK_ROWS
        or block_lengths.shape != (query_rows // _BLOCK_ROWS,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != query.device
        or not block_lengths.is_contiguous()
        or key.shape[2] % _BLOCK_ROWS
        or key.shape[2] // _BLOCK_ROWS > block_lengths.numel()
    ):
        raise ValueError("padded DSA requires one contiguous device INT32 length per query K64")


def _dsa_scores(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    *,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Contract exact FP32 scores while keeping only two score buffers."""
    batch, heads, query_blocks, head_dim = query_summary.shape
    key_blocks = key_max.shape[2]
    flat_query = query_summary.reshape(batch * heads, query_blocks, head_dim)
    flat_key_max = key_max.reshape(batch * heads, key_blocks, head_dim)
    flat_key_min = key_min.reshape(batch * heads, key_blocks, head_dim)
    if score_scale is None:
        scores = torch.bmm(flat_query, flat_key_max.transpose(1, 2))
        minimum_scores = torch.bmm(flat_query, flat_key_min.transpose(1, 2))
    else:
        broadcast_input = flat_query[:, :1, :1]
        scores = torch.baddbmm(
            broadcast_input,
            flat_query,
            flat_key_max.transpose(1, 2),
            beta=0,
            alpha=score_scale,
        )
        minimum_scores = torch.baddbmm(
            broadcast_input,
            flat_query,
            flat_key_min.transpose(1, 2),
            beta=0,
            alpha=score_scale,
        )
    if scores.requires_grad or minimum_scores.requires_grad:
        scores = torch.maximum(scores, minimum_scores)
    else:
        torch.maximum(scores, minimum_scores, out=scores)
    return scores.reshape(batch, heads, query_blocks, key_blocks)


def _dsa_score_chunks(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    *,
    score_scale: float | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    for start in range(0, query_summary.shape[2], _QUERY_CHUNK_BLOCKS):
        stop = min(start + _QUERY_CHUNK_BLOCKS, query_summary.shape[2])
        yield (
            start,
            _dsa_scores(
                query_summary[:, :, start:stop],
                key_max,
                key_min,
                score_scale=score_scale,
            ),
        )


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


def _validate_dsa_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> None:
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
    if block_lengths is not None:
        _validate_dsa_block_lengths(query, key, block_lengths)
