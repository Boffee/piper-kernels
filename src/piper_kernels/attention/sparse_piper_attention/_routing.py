"""Shared block-summary, packed-route, and coarse-score orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from . import _backend
from ._budget import _UINT16_ROUTE_CAPACITY, _ResolvedRouteLayout
from ._contracts import (
    _MEAN_ROUTING,
    valid_block_rows,
    validate_block_lengths,
    validate_routing_mode,
)
from .coarse import (
    _coarse_attention_unchecked,
    _mean_pool_head_major_blocks,
    validate_coarse_scale,
)

_QUERY_CHUNK_BLOCKS = 384


@dataclass(frozen=True, slots=True)
class PackedRoutes:
    """UINT16 routes packed by head; an empty last axis means dense routing is skipped."""

    indices: torch.Tensor
    route_head_offsets: torch.Tensor
    head_keep_blocks: torch.Tensor


@dataclass(frozen=True, slots=True)
class PackedRoutesAndCoarseOutput:
    """Packed fine routes and the coarse output derived from the same scores."""

    routes: PackedRoutes
    coarse_output: torch.Tensor


class PackedRouteBuilder:
    """Build packed routes while resolving shared layout state only once.

    Optional membership is a prebuilt contiguous INT32 bitset for this query
    window: [query_blocks, ceil(sparse_key_blocks / 32)]. Its invocation owner
    validates the caller's IDs/counts and shares the bitset across windows.
    """

    def __init__(
        self,
        layout: _ResolvedRouteLayout,
        *,
        batch: int,
        heads: int,
        query_blocks: int,
        sparse_key_blocks: int,
        device: torch.device,
        skip_dense_routing: bool = False,
        force_tile_mask: torch.Tensor | None = None,
    ) -> None:
        _validate_route_layout(layout, heads, sparse_key_blocks, device)
        self._layout = layout
        self._full_keep = layout.keeps_all_blocks(sparse_key_blocks)
        skip_dense_routing = self._full_keep and skip_dense_routing
        self.routes = PackedRoutes(
            indices=torch.empty(
                (batch, query_blocks, 0 if skip_dense_routing else layout.routes_per_query),
                dtype=torch.uint16,
                device=device,
            ),
            route_head_offsets=layout.route_head_offsets,
            head_keep_blocks=layout.head_keep_blocks,
        )
        if self._full_keep:
            if not skip_dense_routing:
                _backend.fill_full_keep_routes(self.routes.indices, sparse_key_blocks)
            return
        self._select_routes = _backend.select_route_selector(self.routes.indices)
        self._force_tile_mask = force_tile_mask
        self._route_head_offsets = (
            None
            if self._select_routes is not None
            else layout.route_head_offsets.detach().cpu().tolist()
        )
        self._head_keep_block_values = (
            None
            if self._select_routes is not None
            else layout.head_keep_blocks.detach().cpu().tolist()
        )

    def write(
        self,
        scores: torch.Tensor,
        *,
        query_block_offset: int,
    ) -> None:
        """Select stable top-k routes from one dense-key score chunk."""
        if self._full_keep:
            return
        if self._select_routes is not None:
            self._select_routes(
                scores,
                self.routes.indices,
                self._layout.head_keep_blocks,
                self._layout.route_head_offsets,
                query_block_offset=query_block_offset,
                force_tile_mask=self._force_tile_mask,
            )
            return
        assert self._route_head_offsets is not None
        assert self._head_keep_block_values is not None
        _select_portable_routes(
            scores,
            self.routes.indices,
            self._route_head_offsets,
            self._head_keep_block_values,
            query_block_offset=query_block_offset,
            force_tile_mask=self._force_tile_mask,
        )


class PackedRouteAndCoarseBuilder:
    """Route over a sparse prefix while attending coarsely over a wider prefix."""

    def __init__(
        self,
        layout: _ResolvedRouteLayout,
        pooled_value: torch.Tensor,
        *,
        batch: int,
        heads: int,
        query_blocks: int,
        sparse_key_blocks: int,
        device: torch.device,
        skip_dense_routing: bool = False,
        force_tile_mask: torch.Tensor | None = None,
    ) -> None:
        if pooled_value.ndim != 4 or pooled_value.shape[-1] < 1:
            raise ValueError("pooled V must use [batch,heads,key blocks,features]")
        if pooled_value.shape[:2] != (batch, heads):
            raise ValueError("pooled V must match the routing batch and heads")
        coarse_key_blocks = pooled_value.shape[2]
        if not sparse_key_blocks <= coarse_key_blocks:
            raise ValueError("coarse attention must include every sparse key block")
        if pooled_value.dtype is not torch.float32:
            raise ValueError("pooled V must use FP32")
        if pooled_value.device != device:
            raise ValueError("pooled V and routing scores must share a device")
        self._route_builder = PackedRouteBuilder(
            layout,
            batch=batch,
            heads=heads,
            query_blocks=query_blocks,
            sparse_key_blocks=sparse_key_blocks,
            device=device,
            skip_dense_routing=skip_dense_routing,
            force_tile_mask=force_tile_mask,
        )
        self._pooled_value = pooled_value
        self._sparse_key_blocks = sparse_key_blocks
        self._coarse_output = pooled_value.new_empty(
            (batch, heads, query_blocks, pooled_value.shape[-1])
        )
        self._written_query_ranges: list[tuple[int, int]] = []

    def write(
        self,
        scores: torch.Tensor,
        *,
        query_block_offset: int,
    ) -> None:
        """Route over each score prefix without retaining the full coarse matrix."""
        if scores.ndim != 4 or scores.shape[-1] != self._pooled_value.shape[2]:
            raise ValueError("coarse route scores must match pooled V block dimensions")
        if isinstance(query_block_offset, bool) or not isinstance(query_block_offset, int):
            raise TypeError("coarse score query-block offset must be an integer")
        query_block_count = scores.shape[2]
        query_block_stop = query_block_offset + query_block_count
        if not 0 <= query_block_offset < query_block_stop <= self._coarse_output.shape[2]:
            raise ValueError("coarse score query-block range must fit the output")
        if any(
            query_block_offset < written_stop and written_start < query_block_stop
            for written_start, written_stop in self._written_query_ranges
        ):
            raise ValueError("coarse score query-block ranges must not overlap")
        self._route_builder.write(
            scores[..., : self._sparse_key_blocks],
            query_block_offset=query_block_offset,
        )
        self._coarse_output[:, :, query_block_offset:query_block_stop] = (
            _coarse_attention_unchecked(scores, self._pooled_value)
        )
        self._written_query_ranges.append((query_block_offset, query_block_stop))

    def finish(self) -> PackedRoutesAndCoarseOutput:
        """Return packed routes and their query-block-aligned coarse output."""
        next_query_block = 0
        for written_start, written_stop in sorted(self._written_query_ranges):
            if written_start != next_query_block:
                raise RuntimeError("coarse score chunks must cover every query block")
            next_query_block = written_stop
        if next_query_block != self._coarse_output.shape[2]:
            raise RuntimeError("coarse score chunks must cover every query block")
        return PackedRoutesAndCoarseOutput(
            routes=self._route_builder.routes,
            coarse_output=self._coarse_output,
        )


def _validate_route_layout(
    layout: _ResolvedRouteLayout,
    heads: int,
    sparse_key_blocks: int,
    device: torch.device,
) -> None:
    """Validate metadata shared by every packed block-routing policy."""
    if not 1 <= sparse_key_blocks <= _UINT16_ROUTE_CAPACITY:
        raise ValueError("sparse routing requires between 1 and 65,536 sparse key blocks")
    if layout.head_keep_blocks.shape != (heads,) or layout.route_head_offsets.shape != (heads + 1,):
        raise ValueError("sparse route layout does not match the attention head count")
    if layout.head_keep_blocks.device != device or layout.route_head_offsets.device != device:
        raise ValueError("sparse route layout must share the attention device")


def _select_portable_routes(
    scores: torch.Tensor,
    routes: torch.Tensor,
    route_head_offsets: list[int],
    head_keep_block_values: list[int],
    *,
    query_block_offset: int,
    force_tile_mask: torch.Tensor | None = None,
) -> None:
    query_stop = query_block_offset + scores.shape[2]
    forced = None
    if force_tile_mask is not None:
        key_ids = torch.arange(scores.shape[-1], device=scores.device)
        words = force_tile_mask[query_block_offset:query_stop, key_ids // 32]
        forced = ((words >> (key_ids % 32)) & 1).bool()
    for head, count in enumerate(head_keep_block_values):
        # Stable partition by priority after the score sort; scores remain intact
        # for callers that also use them for the coarse attention branch.
        selected = torch.argsort(
            scores[:, head],
            dim=-1,
            descending=True,
            stable=True,
        )
        if forced is not None:
            priority = forced[None].expand(scores.shape[0], -1, -1).gather(-1, selected)
            priority_order = torch.argsort(priority.int(), dim=-1, descending=True, stable=True)
            selected = selected.gather(-1, priority_order)
        selected = selected[..., :count]
        selected = selected.sort(dim=-1).values.to(torch.uint16)
        routes[
            :,
            query_block_offset:query_stop,
            route_head_offsets[head] : route_head_offsets[head + 1],
        ] = selected


def _sequence_block_summaries_validated(
    query: torch.Tensor,
    key: torch.Tensor,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute policy-specific Q/K summaries for already-validated operands."""
    summarize = _backend.select_sequence_summaries(query, key, block_lengths)
    if summarize is not None:
        return summarize(query, key, routing_mode, block_lengths)

    key_lengths = None if block_lengths is None else block_lengths[: key.shape[2] // _BLOCK_ROWS]
    if routing_mode == _MEAN_ROUTING:
        query_summary = _mean_pool_head_major_blocks(query, block_lengths)
        key_primary = _mean_pool_head_major_blocks(key, key_lengths)
        return query_summary, key_primary, key_primary[:, :, :0]

    if block_lengths is None:
        query_max, query_min = _compact_block_extrema(query)
        key_primary, key_aux = _compact_block_extrema(key)
    else:
        assert key_lengths is not None
        query_max, query_min = _padded_block_extrema(query, block_lengths)
        key_primary, key_aux = _padded_block_extrema(key, key_lengths)
    return query_max + query_min, key_primary, key_aux


def sequence_block_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    *,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy-specific Q/K summaries through one fixed tensor contract."""
    if validate:
        validate_routing_mode(routing_mode)
        _validate_sequences(query, key, block_lengths)
    return _sequence_block_summaries_validated(query, key, routing_mode, block_lengths)


def _compact_block_extrema(sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    valid_rows = valid_block_rows(block_lengths)[None, None, :, :, None]
    maximum = torch.where(valid_rows, blocks, -float("inf")).amax(dim=3)
    minimum = torch.where(valid_rows, blocks, float("inf")).amin(dim=3)
    return maximum, minimum


def _validate_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    block_lengths: torch.Tensor | None,
) -> None:
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("routing query and key sequences must be rank-four tensors")
    if (
        query.shape[0] != key.shape[0]
        or key.shape[1] < 1
        or query.shape[1] < 1
        or query.shape[1] % key.shape[1]
        or query.shape[-1] != key.shape[-1]
    ):
        raise ValueError("routing query and key batch/head/feature dimensions must match")
    if query.shape[2] < 1 or key.shape[2] < 1:
        raise ValueError("routing summaries require nonempty Q/K sequences")
    if query.device != key.device:
        raise ValueError("routing query and key sequences must share a device")
    if not query.is_floating_point() or not key.is_floating_point():
        raise TypeError("routing query and key sequences must be floating-point tensors")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("routing query and key features must be contiguous")
    if block_lengths is not None:
        block_count = validate_block_lengths(
            block_lengths,
            sequence_length=query.shape[2],
            device=query.device,
            context="padded sparse routing",
            require_contiguous=True,
        )
        if key.shape[2] % _BLOCK_ROWS:
            raise ValueError("padded routing keys must use complete physical K64 blocks")
        if key.shape[2] // _BLOCK_ROWS > block_count:
            raise ValueError("padded routing keys cannot exceed the block-length layout")


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
    return packed_routes_from_summaries(
        query_summary,
        key_primary,
        key_aux,
        layout,
        routing_mode,
        skip_dense_routing=skip_dense_routing,
        force_tile_mask=force_tile_mask,
        validate=False,
    )


def packed_routes_from_summaries(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    layout: _ResolvedRouteLayout,
    routing_mode: int,
    *,
    skip_dense_routing: bool = False,
    force_tile_mask: torch.Tensor | None = None,
    validate: bool = True,
) -> PackedRoutes:
    """Select routes through the fixed policy-independent summary contract."""
    if validate:
        _validate_summaries(query_summary, key_primary, key_aux, routing_mode)
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
    if key_primary.shape[2] != pooled_value.shape[2]:
        raise ValueError("coarse score key blocks must match pooled V blocks")
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


def _routing_scores_unchecked(
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
    return _routing_scores_unchecked(
        query_summary,
        key_primary,
        key_aux,
        routing_mode,
        score_scale=score_scale,
    )


def score_chunks(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    routing_mode: int,
    *,
    score_scale: float | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield bounded query-block score chunks for either routing policy."""
    validate_routing_mode(routing_mode)
    for start in range(0, query_summary.shape[2], _QUERY_CHUNK_BLOCKS):
        stop = min(start + _QUERY_CHUNK_BLOCKS, query_summary.shape[2])
        yield (
            start,
            _routing_scores_unchecked(
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
    "PackedRouteAndCoarseBuilder",
    "PackedRouteBuilder",
    "PackedRoutes",
    "PackedRoutesAndCoarseOutput",
    "packed_routes_and_coarse_from_summaries",
    "packed_routes_from_sequences",
    "packed_routes_from_summaries",
    "routing_scores",
    "score_chunks",
    "sequence_block_summaries",
]
