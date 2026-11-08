"""Key layouts: ordered K64 regions with independent per-head keep budgets.

A layout covers every complete physical K64 block of the key sequence, in order.
Sparse regions apply the module's per-query-head keep ratios independently, and
dense regions keep every block, within one softmax. A compact partial final
block is always attended.

Layouts are runtime configuration. Compiled graphs pass them opaquely to the
operators that read them, so changing a layout, including its total block
count, reuses the graph. Compiler passes and model code must not read layout
contents; operators check coverage on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from operator import add

import torch
from torch._custom_class_base import CustomClassBase
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import register_custom_class

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS

from ._budget import (
    _UINT16_ROUTE_CAPACITY,
    ResolvedRouteRegion,
    _resolve_head_keep_blocks,
    _ResolvedRouteLayout,
)


@dataclass(frozen=True, slots=True)
class SparsePiperKeyRegion:
    """Describe one run of consecutive physical 64-token key blocks.

    ``num_blocks`` is a positive Python integer. Sparse regions independently
    apply the module's per-query-head keep ratios; ``sparse=False`` keeps every
    block. Pad a region's storage and use ``block_lengths`` when its valid token
    count is not a multiple of 64.
    """

    num_blocks: int
    sparse: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.num_blocks, bool) or not isinstance(self.num_blocks, int):
            raise TypeError("region num_blocks must be an integer")
        if not 1 <= self.num_blocks <= _UINT16_ROUTE_CAPACITY:
            raise ValueError("region num_blocks must lie in [1, 65536]")
        if not isinstance(self.sparse, bool):
            raise TypeError("region sparse must be a bool")


@dataclass(frozen=True, slots=True)
class SparsePiperKeyLayout(CustomClassBase):
    """Partition key storage into independently budgeted sparse and dense regions.

    ``regions`` is a nonempty tuple of ``SparsePiperKeyRegion`` values in physical
    key order, with at least one sparse region. Together they cover every
    complete 64-token block of the key sequence, at most 65536 blocks; a compact
    partial final block is always attended. The partition is shared by every
    batch, query, and head.

    Construct and inspect layouts outside compiled forward calls. Compiled code
    passes them opaquely, so changing region count, sizes, sparse/dense flags,
    or total block count reuses the graph; mark sequence dimensions dynamic to
    reuse it across sequence lengths. Explicit CUDA graphs require a fixed
    layout; recapture when it changes.
    """

    regions: tuple[SparsePiperKeyRegion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.regions, tuple):
            raise TypeError("layout regions must be a tuple of SparsePiperKeyRegion values")
        if not self.regions:
            raise ValueError("layout regions must be nonempty")
        if any(not isinstance(region, SparsePiperKeyRegion) for region in self.regions):
            raise TypeError("layout regions must contain SparsePiperKeyRegion values")
        if not any(region.sparse for region in self.regions):
            raise ValueError("key layout must contain a sparse region")
        if self.num_blocks > _UINT16_ROUTE_CAPACITY:
            raise ValueError("key layout must cover at most 65536 physical blocks")

    @property
    def num_blocks(self) -> int:
        """Return the total number of physical 64-token blocks in the partition."""
        return sum(region.num_blocks for region in self.regions)


# Consumers carry cudagraph_unsafe: automatic CUDA graph replay does not guard
# symbolic custom-class values and would reuse routes and budgets from capture.
register_custom_class(SparsePiperKeyLayout, typ="symbolic")


def validate_key_layout(key_layout: SparsePiperKeyLayout) -> None:
    """Check the public argument type; construction owns partition validation."""
    if not isinstance(key_layout, SparsePiperKeyLayout):
        raise TypeError("key_layout must be a SparsePiperKeyLayout")


def validate_key_layout_storage(
    # Operator fakes receive Torch's opaque stand-in during compiler tracing.
    key_layout: SparsePiperKeyLayout | FakeScriptObject,
    sequence_length: int,
) -> None:
    """Check that the layout covers the sequence's complete physical K64 blocks.

    Operators call this at their real and fake boundaries. During compiler fake
    execution the layout is opaque, so coverage is checked on each real call;
    reading it there would specialize the graph on the layout.
    """
    if isinstance(key_layout, FakeScriptObject):
        return
    if key_layout.num_blocks != sequence_length // TILE_ROWS:
        raise ValueError("key_layout must cover exactly the sequence's complete K64 blocks")


def resolve_keep_blocks(
    ratio_units: tuple[int, ...],
    key_layout: SparsePiperKeyLayout,
) -> tuple[tuple[int, ...], ...]:
    """Resolve per-head budgets in region order, rounding across heads per region."""
    return tuple(
        _resolve_head_keep_blocks(ratio_units, region.num_blocks)
        if region.sparse
        else (region.num_blocks,) * len(ratio_units)
        for region in key_layout.regions
    )


@dataclass(frozen=True, slots=True)
class _RoutePlan:
    """Host-side route layout for one profile, key layout, and head range."""

    num_heads: int
    routes_per_query: int
    regions: tuple[ResolvedRouteRegion, ...]
    metadata_values: tuple[int, ...]


# Profiles and layouts are immutable configuration that models reuse on every
# layer and step, so plan each combination once. Each call still stages its own
# device metadata, as explicit CUDA graph capture requires.
@lru_cache(maxsize=256)
def _build_route_plan(
    ratio_units: tuple[int, ...],
    key_layout: SparsePiperKeyLayout,
    start_head: int,
    end_head: int | None,
) -> _RoutePlan:
    budgets = [
        counts[start_head:end_head] for counts in resolve_keep_blocks(ratio_units, key_layout)
    ]
    totals = [sum(column) for column in zip(*budgets, strict=True)]
    offsets = list(accumulate(totals, initial=0))
    if offsets[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("sparse Piper packed routes exceed INT32 offset capacity")

    regions = []
    selector_values: list[int] = []
    next_offsets = tuple(offsets[:-1])
    start_key_block = start_score_block = 0
    for key_region, counts in zip(key_layout.regions, budgets, strict=True):
        region = ResolvedRouteRegion(
            start_key_block=start_key_block,
            num_key_blocks=key_region.num_blocks,
            start_score_block=start_score_block,
            head_keep_block_values=counts,
            route_head_offset_values=next_offsets,
        )
        regions.append(region)
        selector_values.extend((start_key_block, region.num_key_blocks, start_score_block))
        selector_values.extend(counts)
        selector_values.extend(next_offsets)
        next_offsets = tuple(map(add, next_offsets, counts))
        start_key_block += region.num_key_blocks
        if not region.keeps_all_blocks:
            start_score_block += region.num_key_blocks
    return _RoutePlan(
        num_heads=len(totals),
        routes_per_query=offsets[-1],
        regions=tuple(regions),
        metadata_values=(*totals, *offsets, *selector_values),
    )


def resolve_route_layout(
    ratio_units: tuple[int, ...],
    key_layout: SparsePiperKeyLayout,
    device: torch.device,
    *,
    start_head: int = 0,
    end_head: int | None = None,
) -> _ResolvedRouteLayout:
    """Lay out routes for profile heads ``[start_head, end_head)`` in one transfer.

    Each region rounds across every profile head before the slice, so a head range
    keeps the blocks its heads keep when all heads run together. Routes cover
    every complete block, dense regions included.
    """
    plan = _build_route_plan(ratio_units, key_layout, start_head, end_head)
    # Pinned staging makes the metadata copy legal during CUDA graph capture.
    metadata = torch.tensor(
        plan.metadata_values,
        dtype=torch.int32,
        device="cpu",
        pin_memory=device.type == "cuda",
    ).to(device, non_blocking=True)
    num_heads = plan.num_heads
    selector_start = 2 * num_heads + 1
    return _ResolvedRouteLayout(
        head_keep_blocks=metadata[:num_heads],
        route_head_offsets=metadata[num_heads:selector_start],
        routes_per_query=plan.routes_per_query,
        num_key_blocks=key_layout.num_blocks,
        regions=plan.regions,
        selector_metadata=metadata[selector_start:].view(len(plan.regions), selector_start + 2),
    )
