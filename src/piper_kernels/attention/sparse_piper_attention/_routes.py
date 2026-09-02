"""Routing-policy-independent packed sparse route selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._budget import _UINT16_ROUTE_CAPACITY, _ResolvedRouteLayout
from .coarse import coarse_attention

try:
    from .dsa_triton import tiled_radix_select_packed_routes as _sm120_select_routes
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("triton"):
        raise
    _sm120_select_routes = None

_DSA_ROUTING = 0
_MEAN_POOL_ROUTING = 1
_ROUTING_MODE_BY_NAME = {"dsa": _DSA_ROUTING, "mean_pool": _MEAN_POOL_ROUTING}
_ROUTING_NAME_BY_MODE = {mode: name for name, mode in _ROUTING_MODE_BY_NAME.items()}


@dataclass(frozen=True, slots=True)
class PackedRoutes:
    """UINT16 routes packed by physical head for every query block."""

    indices: torch.Tensor
    head_offsets: torch.Tensor
    keep_blocks: torch.Tensor


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
        key_blocks: int,
        device: torch.device,
    ) -> None:
        _validate_route_layout(layout, heads, key_blocks, device)
        self._layout = layout
        self.routes = PackedRoutes(
            indices=torch.empty(
                (batch, query_blocks, layout.routes_per_query),
                dtype=torch.uint16,
                device=device,
            ),
            head_offsets=layout.head_offsets,
            keep_blocks=layout.keep_blocks,
        )
        self._use_sm120 = _supports_sm120_selector(device)
        self._offsets = None if self._use_sm120 else layout.head_offsets.detach().cpu().tolist()
        self._keep_values = None if self._use_sm120 else layout.keep_blocks.detach().cpu().tolist()

    def write(
        self,
        scores: torch.Tensor,
        *,
        route_query_offset: int,
    ) -> None:
        """Select stable top-k routes from one contiguous score chunk."""
        if self._use_sm120:
            assert _sm120_select_routes is not None
            _sm120_select_routes(
                scores,
                self.routes.indices,
                self._layout.keep_blocks,
                self._layout.head_offsets,
                route_query_offset=route_query_offset,
            )
            return
        assert self._offsets is not None
        assert self._keep_values is not None
        _select_portable_routes(
            scores,
            self.routes.indices,
            self._offsets,
            self._keep_values,
            route_query_offset=route_query_offset,
        )


class PackedRouteAndCoarseBuilder:
    """Consume each score chunk once for fine routing and coarse attention."""

    def __init__(
        self,
        layout: _ResolvedRouteLayout,
        pooled_value: torch.Tensor,
        *,
        batch: int,
        heads: int,
        query_blocks: int,
        key_blocks: int,
        device: torch.device,
        coarse_scale: float,
    ) -> None:
        if not math.isfinite(coarse_scale) or coarse_scale <= 0:
            raise ValueError("coarse attention scale must be finite and positive")
        if pooled_value.shape[:3] != (batch, heads, key_blocks):
            raise ValueError("pooled V must match the routing batch, heads, and key blocks")
        if pooled_value.ndim != 4 or pooled_value.shape[-1] < 1:
            raise ValueError("pooled V must use [batch,heads,key blocks,features]")
        if pooled_value.dtype is not torch.float32:
            raise ValueError("pooled V must use FP32")
        if pooled_value.device != device:
            raise ValueError("pooled V and routing scores must share a device")
        self._route_builder = PackedRouteBuilder(
            layout,
            batch=batch,
            heads=heads,
            query_blocks=query_blocks,
            key_blocks=key_blocks,
            device=device,
        )
        self._pooled_value = pooled_value
        self._coarse_scale = coarse_scale
        self._coarse_chunks: list[torch.Tensor] = []

    def write(
        self,
        scores: torch.Tensor,
        *,
        route_query_offset: int,
    ) -> None:
        """Consume one raw score chunk without retaining its score matrix."""
        self._route_builder.write(scores, route_query_offset=route_query_offset)
        scores.mul_(self._coarse_scale)
        self._coarse_chunks.append(coarse_attention(scores, self._pooled_value))

    def finish(self) -> PackedRoutesAndCoarseOutput:
        """Return packed routes and the concatenated query-block output."""
        coarse_output = (
            self._coarse_chunks[0]
            if len(self._coarse_chunks) == 1
            else torch.cat(self._coarse_chunks, dim=2)
        )
        return PackedRoutesAndCoarseOutput(
            routes=self._route_builder.routes,
            coarse_output=coarse_output,
        )


def _validate_route_layout(
    layout: _ResolvedRouteLayout,
    heads: int,
    key_blocks: int,
    device: torch.device,
) -> None:
    """Validate metadata shared by every packed block-routing policy."""
    if not 1 <= key_blocks <= _UINT16_ROUTE_CAPACITY:
        raise ValueError("sparse routing requires between 1 and 65,536 sparse key blocks")
    if layout.keep_blocks.shape != (heads,) or layout.head_offsets.shape != (heads + 1,):
        raise ValueError("sparse route layout does not match the attention head count")
    if layout.keep_blocks.device != device or layout.head_offsets.device != device:
        raise ValueError("sparse route layout must share the attention device")


def _select_portable_routes(
    scores: torch.Tensor,
    output: torch.Tensor,
    offsets: list[int],
    keep_values: list[int],
    *,
    route_query_offset: int,
) -> None:
    query_stop = route_query_offset + scores.shape[2]
    for head, count in enumerate(keep_values):
        selected = torch.argsort(
            scores[:, head],
            dim=-1,
            descending=True,
            stable=True,
        )[..., :count]
        selected = selected.sort(dim=-1).values.to(torch.uint16)
        output[:, route_query_offset:query_stop, offsets[head] : offsets[head + 1]] = selected


def _supports_sm120_selector(device: torch.device) -> bool:
    target = AcceleratorTarget.from_device(device)
    return _sm120_select_routes is not None and target.is_cuda_capability(12, 0)


def validate_routing_mode(routing_mode: int) -> None:
    """Reject routing modes outside the internal static operator contract."""
    if not is_valid_routing_mode(routing_mode):
        raise ValueError("sparse Piper routing mode must be DSA or mean_pool")


def is_valid_routing_mode(routing_mode: int) -> bool:
    """Return whether a value names a supported static routing policy."""
    return routing_mode in _ROUTING_NAME_BY_MODE


__all__ = [
    "_DSA_ROUTING",
    "_MEAN_POOL_ROUTING",
    "_ROUTING_MODE_BY_NAME",
    "_ROUTING_NAME_BY_MODE",
    "PackedRouteAndCoarseBuilder",
    "PackedRouteBuilder",
    "PackedRoutes",
    "PackedRoutesAndCoarseOutput",
    "is_valid_routing_mode",
    "validate_routing_mode",
]
