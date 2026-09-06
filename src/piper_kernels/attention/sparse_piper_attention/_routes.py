"""Routing-policy-independent packed sparse route selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _backend
from ._budget import _UINT16_ROUTE_CAPACITY, _ResolvedRouteLayout
from .coarse import coarse_attention


@dataclass(frozen=True, slots=True)
class PackedRoutes:
    """UINT16 routes packed by physical head for every query block."""

    indices: torch.Tensor
    route_head_offsets: torch.Tensor
    head_keep_blocks: torch.Tensor


@dataclass(frozen=True, slots=True)
class PackedRoutesAndCoarseOutput:
    """Packed fine routes and the coarse output derived from the same scores."""

    routes: PackedRoutes
    coarse_output: torch.Tensor


class PackedRouteBuilder:
    """Build packed routes while resolving shared layout state only once."""

    def __init__(
        self,
        layout: _ResolvedRouteLayout,
        *,
        batch: int,
        heads: int,
        query_blocks: int,
        sparse_key_blocks: int,
        device: torch.device,
    ) -> None:
        _validate_route_layout(layout, heads, sparse_key_blocks, device)
        self._layout = layout
        self.routes = PackedRoutes(
            indices=torch.empty(
                (batch, query_blocks, layout.routes_per_query),
                dtype=torch.uint16,
                device=device,
            ),
            route_head_offsets=layout.route_head_offsets,
            head_keep_blocks=layout.head_keep_blocks,
        )
        self._select_routes = _backend.select_route_selector(self.routes.indices)
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
        if self._select_routes is not None:
            self._select_routes(
                scores,
                self.routes.indices,
                self._layout.head_keep_blocks,
                self._layout.route_head_offsets,
                query_block_offset=query_block_offset,
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
            scores[..., : self._sparse_key_blocks],
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
) -> None:
    query_stop = query_block_offset + scores.shape[2]
    for head, count in enumerate(head_keep_block_values):
        selected = torch.argsort(
            scores[:, head],
            dim=-1,
            descending=True,
            stable=True,
        )[..., :count]
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
