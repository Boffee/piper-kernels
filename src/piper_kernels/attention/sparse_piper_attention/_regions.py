"""Ordered key partitions and independent regional route budgets.

Regions cover complete physical K64 blocks. Their selections are packed in
physical key order within each query head, so attention needs only the total
head budgets and absolute block IDs, never region metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

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


def validate_key_regions(regions: tuple[SparsePiperKeyRegion, ...]) -> int:
    """Validate a caller's partition and return its physical key block count."""
    if not isinstance(regions, tuple) or not regions:
        raise ValueError("key_regions must be a nonempty tuple of SparsePiperKeyRegion values")
    if any(not isinstance(region, SparsePiperKeyRegion) for region in regions):
        raise TypeError("key_regions must contain SparsePiperKeyRegion values")
    num_blocks = sum(region.num_blocks for region in regions)
    if num_blocks > _UINT16_ROUTE_CAPACITY:
        raise ValueError("key_regions must cover at most 65536 physical blocks")
    return num_blocks


def resolve_region_keep_blocks(
    ratio_units: tuple[int, ...],
    region_blocks: Sequence[int],
    region_sparse: Sequence[bool],
) -> tuple[tuple[int, ...], ...]:
    """Resolve validated configuration, rounding across heads within each region."""
    return tuple(
        _resolve_head_keep_blocks(ratio_units, num_blocks)
        if sparse
        else (num_blocks,) * len(ratio_units)
        for num_blocks, sparse in zip(region_blocks, region_sparse, strict=True)
    )


def resolve_region_route_layout(
    ratio_units: tuple[int, ...],
    region_blocks: Sequence[int],
    region_sparse: Sequence[bool],
    device: torch.device,
) -> _ResolvedRouteLayout:
    """Build total head budgets and region destinations with one metadata transfer."""
    budgets = resolve_region_keep_blocks(ratio_units, region_blocks, region_sparse)
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
    for num_blocks, counts in zip(region_blocks, budgets, strict=True):
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
