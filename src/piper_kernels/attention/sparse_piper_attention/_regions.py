"""Ordered key partitions and independent regional route budgets.

Regions cover complete physical K64 blocks. Their selections are packed in
physical key order within each query head, so attention needs only the total
head budgets and absolute block IDs, never region metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """Describe one consecutive region of physical 64-token key blocks.

    ``num_blocks`` is a positive Python integer. Sparse regions independently
    apply the module's per-query-head keep ratios; ``sparse=False`` keeps every
    block. Regions share one softmax and the same partition for every batch,
    query block, and head. Pad region boundaries using ``block_lengths`` when
    a region's valid token count is not a multiple of 64.
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
    key order, covering at most 65536 complete 64-token blocks. The partition is
    immutable and shared by every batch, query, and head. Attention checks that
    its total block count covers the supplied sequence exactly.

    Construct and inspect layouts outside compiled forward calls; compiled code
    passes them opaquely to attention. Region count, boundaries,
    sparse/dense flags, and total block count are runtime configuration. Mark
    sequence dimensions dynamic to reuse compiled graphs across sequence lengths.
    Other tensor shapes and kernel policies can still introduce specializations.
    Explicit CUDA graphs require fixed shapes, layout, and ratio profile;
    recapture when any changes.
    """

    regions: tuple[SparsePiperKeyRegion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.regions, tuple):
            raise TypeError("layout regions must be a tuple of SparsePiperKeyRegion values")
        if not self.regions:
            raise ValueError("layout regions must be nonempty")
        if any(not isinstance(region, SparsePiperKeyRegion) for region in self.regions):
            raise TypeError("layout regions must contain SparsePiperKeyRegion values")
        if self.num_blocks > _UINT16_ROUTE_CAPACITY:
            raise ValueError("key layout must cover at most 65536 physical blocks")

    @property
    def num_blocks(self) -> int:
        """Return the total number of physical 64-token blocks in the partition."""
        return sum(region.num_blocks for region in self.regions)


# No layout values are graph metadata, including its total block count. Exact
# coverage is checked inside the operator on every execution, so dynamic sequence
# lengths do not specialize the enclosing graph through the opaque object.
# Consumers must remain cudagraph_unsafe because replay would retain the layout
# and budgets used at capture.
register_custom_class(SparsePiperKeyLayout, typ="symbolic")


def validate_key_layout(key_layout: SparsePiperKeyLayout) -> None:
    """Check the public argument type; construction owns partition validation."""
    if not isinstance(key_layout, SparsePiperKeyLayout):
        raise TypeError("key_layout must be a SparsePiperKeyLayout")


def validate_key_layout_storage(key_layout: SparsePiperKeyLayout, sequence_length: int) -> None:
    """Validate storage at the real/fake operator boundary before preparing routes.

    Common input validation already requires a positive sequence. During compiler
    fake execution, partition contents are opaque: check the physical block shape
    and defer exact coverage to each runtime call. Concrete fake/meta calls can
    also check coverage directly without inspecting any tensor contents.
    """
    if sequence_length % TILE_ROWS or sequence_length > _UINT16_ROUTE_CAPACITY * TILE_ROWS:
        raise ValueError("key_layout must cover exactly the sequence's complete K64 blocks")
    if (
        not isinstance(key_layout, FakeScriptObject)
        and sequence_length != key_layout.num_blocks * TILE_ROWS
    ):
        raise ValueError("key_layout must cover exactly the sequence's complete K64 blocks")


def resolve_region_keep_blocks(
    ratio_units: tuple[int, ...],
    key_layout: SparsePiperKeyLayout,
) -> tuple[tuple[int, ...], ...]:
    """Resolve validated configuration, rounding across heads within each region."""
    return tuple(
        _resolve_head_keep_blocks(ratio_units, region.num_blocks)
        if region.sparse
        else (region.num_blocks,) * len(ratio_units)
        for region in key_layout.regions
    )


def resolve_region_route_layout(
    ratio_units: tuple[int, ...],
    key_layout: SparsePiperKeyLayout,
    device: torch.device,
) -> _ResolvedRouteLayout:
    """Build total head budgets and region destinations with one metadata transfer."""
    budgets = resolve_region_keep_blocks(ratio_units, key_layout)
    num_heads = len(ratio_units)
    totals = tuple(sum(counts[head] for counts in budgets) for head in range(num_heads))
    offsets = [0]
    for count in totals:
        offsets.append(offsets[-1] + count)
    if offsets[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("sparse Piper packed routes exceed INT32 offset capacity")
    next_offsets = offsets.copy()
    values = [*totals, *offsets]
    regions = []
    start_key_block = start_score_block = 0
    for key_region, counts in zip(key_layout.regions, budgets, strict=True):
        num_blocks = key_region.num_blocks
        region = ResolvedRouteRegion(
            start_key_block=start_key_block,
            num_key_blocks=num_blocks,
            start_score_block=start_score_block,
            head_keep_block_values=counts,
            route_head_offset_values=tuple(next_offsets[:-1]),
        )
        regions.append(region)
        # One row per region: physical start, length, compact score start,
        # H keep counts, H packed destinations. Dense rows never read scores.
        values.extend((start_key_block, num_blocks, start_score_block, *counts, *next_offsets[:-1]))
        for head, count in enumerate(counts):
            next_offsets[head] += count
        start_key_block += num_blocks
        if not region.keeps_all_blocks:
            start_score_block += num_blocks
    # Pinned staging follows the existing explicit CUDA-graph capture contract.
    metadata = torch.tensor(
        values, dtype=torch.int32, device="cpu", pin_memory=device.type == "cuda"
    ).to(device, non_blocking=True)
    stride = 2 * num_heads + 1
    return _ResolvedRouteLayout(
        head_keep_blocks=metadata[:num_heads],
        route_head_offsets=metadata[num_heads:stride],
        routes_per_query=offsets[-1],
        regions=tuple(regions),
        region_metadata=metadata[stride:].view(len(regions), 2 * num_heads + 3),
    )
