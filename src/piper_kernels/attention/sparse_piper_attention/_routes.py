"""Routing-policy-independent packed sparse route selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _backend
from ._budget import _ResolvedRouteLayout
from .coarse import _coarse_attention_unchecked


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
        if layout.head_keep_blocks.shape != (heads,):
            raise ValueError("sparse route layout does not match the attention head count")
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
        self._written_query_blocks = 0

    def write(
        self,
        scores: torch.Tensor,
        *,
        query_block_offset: int,
    ) -> None:
        """Route over each score prefix without retaining the full coarse matrix."""
        query_block_count = scores.shape[2]
        query_block_stop = query_block_offset + query_block_count
        self._route_builder.write(
            scores[..., : self._sparse_key_blocks],
            query_block_offset=query_block_offset,
        )
        self._coarse_output[:, :, query_block_offset:query_block_stop] = (
            _coarse_attention_unchecked(scores, self._pooled_value)
        )
        self._written_query_blocks += query_block_count

    def finish(self) -> PackedRoutesAndCoarseOutput:
        """Return packed routes and their query-block-aligned coarse output."""
        if self._written_query_blocks != self._coarse_output.shape[2]:
            raise RuntimeError("coarse score chunks must cover every query block")
        return PackedRoutesAndCoarseOutput(
            routes=self._route_builder.routes,
            coarse_output=self._coarse_output,
        )


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


__all__ = [
    "PackedRouteAndCoarseBuilder",
    "PackedRouteBuilder",
    "PackedRoutes",
    "PackedRoutesAndCoarseOutput",
]
