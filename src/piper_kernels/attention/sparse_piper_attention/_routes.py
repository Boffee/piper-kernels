"""Routing-policy-independent packed sparse route selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _backend
from ._budget import ResolvedRouteRegion, _ResolvedRouteLayout
from .coarse import coarse_attention


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
    window: [query_blocks, ceil(num_key_blocks / 32)]. Its invocation owner
    validates the caller's IDs/counts and shares the bitset across windows.
    """

    def __init__(
        self,
        layout: _ResolvedRouteLayout,
        *,
        batch: int,
        heads: int,
        query_blocks: int,
        device: torch.device,
        skip_dense_routing: bool = False,
        force_tile_mask: torch.Tensor | None = None,
    ) -> None:
        _validate_route_layout(layout, heads, device)
        self._layout = layout
        self._full_keep = layout.keeps_all_blocks
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
                _backend.fill_full_keep_routes(self.routes.indices, layout.num_key_blocks)
            return
        self._select_routes = _backend.select_route_selector(self.routes.indices)
        self._force_tile_mask = force_tile_mask
        if self._select_routes is None:
            # The native selector writes these rows itself, one query chunk at a time.
            for region in layout.regions:
                if region.keeps_all_blocks:
                    _write_retained_region(self.routes.indices, region)

    def write(
        self,
        scores: torch.Tensor,
        *,
        query_block_offset: int,
        compact_scores: bool = False,
    ) -> None:
        """Select one score chunk; compact scores omit regions that keep every block."""
        if self._full_keep:
            return
        if self._select_routes is not None:
            self._select_routes(
                scores,
                self.routes.indices,
                self._layout.selector_metadata,
                query_block_offset=query_block_offset,
                force_tile_mask=self._force_tile_mask,
                compact_scores=compact_scores,
            )
            return
        for region in self._layout.regions:
            if region.keeps_all_blocks:
                continue
            start = region.start_score_block if compact_scores else region.start_key_block
            _select_portable_routes(
                scores[..., start : start + region.num_key_blocks],
                self.routes.indices,
                region,
                query_block_offset=query_block_offset,
                force_tile_mask=self._force_tile_mask,
            )


class PackedRouteAndCoarseBuilder:
    """Route the complete key blocks while attending coarsely over the coarse scope."""

    def __init__(
        self,
        layout: _ResolvedRouteLayout,
        pooled_value: torch.Tensor,
        *,
        batch: int,
        heads: int,
        query_blocks: int,
        device: torch.device,
        skip_dense_routing: bool = False,
        force_tile_mask: torch.Tensor | None = None,
    ) -> None:
        if pooled_value.ndim != 4 or pooled_value.shape[-1] < 1:
            raise ValueError("pooled V must use [batch,heads,key blocks,features]")
        if pooled_value.shape[:2] != (batch, heads):
            raise ValueError("pooled V must match the routing batch and heads")
        coarse_key_blocks = pooled_value.shape[2]
        if not layout.num_key_blocks <= coarse_key_blocks:
            raise ValueError("coarse attention must include every routed key block")
        if pooled_value.dtype is not torch.float32:
            raise ValueError("pooled V must use FP32")
        if pooled_value.device != device:
            raise ValueError("pooled V and routing scores must share a device")
        self._route_builder = PackedRouteBuilder(
            layout,
            batch=batch,
            heads=heads,
            query_blocks=query_blocks,
            device=device,
            skip_dense_routing=skip_dense_routing,
            force_tile_mask=force_tile_mask,
        )
        self._pooled_value = pooled_value
        self._num_key_blocks = layout.num_key_blocks
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
        """Route from each chunk's complete-block scores without retaining the coarse matrix."""
        if scores.ndim != 4:
            raise ValueError("coarse route scores must use rank-four block tensors")
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
            scores[..., : self._num_key_blocks],
            query_block_offset=query_block_offset,
        )
        self._coarse_output[:, :, query_block_offset:query_block_stop] = coarse_attention(
            scores,
            self._pooled_value,
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
    device: torch.device,
) -> None:
    """Validate that resolved metadata belongs to these attention operands."""
    if layout.head_keep_blocks.shape != (heads,) or layout.route_head_offsets.shape != (heads + 1,):
        raise ValueError("sparse route layout does not match the attention head count")
    if layout.head_keep_blocks.device != device or layout.route_head_offsets.device != device:
        raise ValueError("sparse route layout must share the attention device")


def _write_retained_region(routes: torch.Tensor, region: ResolvedRouteRegion) -> None:
    blocks = torch.arange(
        region.start_key_block,
        region.start_key_block + region.num_key_blocks,
        device=routes.device,
        dtype=torch.int32,
    ).to(torch.uint16)
    for start in region.route_head_offset_values:
        routes[:, :, start : start + region.num_key_blocks] = blocks


def _select_portable_routes(
    scores: torch.Tensor,
    routes: torch.Tensor,
    region: ResolvedRouteRegion,
    *,
    query_block_offset: int,
    force_tile_mask: torch.Tensor | None = None,
) -> None:
    """Select one region's routes from its score columns, writing absolute IDs."""
    query_stop = query_block_offset + scores.shape[2]
    forced = None
    if force_tile_mask is not None:
        key_ids = region.start_key_block + torch.arange(scores.shape[-1], device=scores.device)
        words = force_tile_mask[query_block_offset:query_stop, key_ids // 32]
        forced = ((words >> (key_ids % 32)) & 1).bool()
    for head, (start, count) in enumerate(
        zip(region.route_head_offset_values, region.head_keep_block_values, strict=True)
    ):
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
        selected = (region.start_key_block + selected.sort(dim=-1).values).to(torch.uint16)
        routes[:, query_block_offset:query_stop, start : start + count] = selected


__all__ = [
    "PackedRouteAndCoarseBuilder",
    "PackedRouteBuilder",
    "PackedRoutes",
    "PackedRoutesAndCoarseOutput",
]
