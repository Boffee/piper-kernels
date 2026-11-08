"""Semantic head budgets and the resolved packed-route layout."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

_RATIO_SCALE = 1_000_000
_UINT16_ROUTE_CAPACITY = 1 << 16


@dataclass(frozen=True, slots=True)
class ResolvedRouteRegion:
    """One routed key interval and its destinations in the packed route list."""

    start_key_block: int
    num_key_blocks: int
    start_score_block: int
    head_keep_block_values: tuple[int, ...]
    route_head_offset_values: tuple[int, ...]

    @property
    def keeps_all_blocks(self) -> bool:
        return min(self.head_keep_block_values) == self.num_key_blocks


@dataclass(frozen=True, slots=True)
class _ResolvedRouteLayout:
    """Route metadata for one key layout and query-head range.

    Routes cover the layout's ``num_key_blocks`` complete key blocks, which the
    attention kernels receive as their routed range. Only a compact partial final
    tile follows them without a route, so kernels mask that tile alone. Forced
    IDs share the same domain. ``regions`` partitions the blocks, and
    ``selector_metadata`` holds one device row per region: start block, length,
    compact score start, H keep counts, and H packed destinations. Compact scores
    omit regions that keep every block. Attention consumes only the aggregate
    counts and offsets.
    """

    head_keep_blocks: torch.Tensor
    route_head_offsets: torch.Tensor
    routes_per_query: int
    num_key_blocks: int
    regions: tuple[ResolvedRouteRegion, ...]
    selector_metadata: torch.Tensor

    @property
    def keeps_all_blocks(self) -> bool:
        """Recognize a full routed budget without inspecting device values."""
        return all(region.keeps_all_blocks for region in self.regions)


def _normalize_head_keep_ratios(
    head_keep_ratios: Sequence[float] | torch.Tensor,
) -> tuple[int, ...]:
    if isinstance(head_keep_ratios, torch.Tensor):
        if head_keep_ratios.ndim != 1 or head_keep_ratios.numel() < 1:
            raise ValueError("sparse Piper head keep ratios must be a nonempty vector")
        if head_keep_ratios.dtype is torch.bool or not head_keep_ratios.is_floating_point():
            raise TypeError("sparse Piper head keep ratios must use a floating-point dtype")
        ratios = tuple(float(value) for value in head_keep_ratios.detach().cpu().tolist())
    else:
        if isinstance(head_keep_ratios, (str, bytes)):
            raise TypeError("sparse Piper head keep ratios must be a numeric sequence")
        ratios = tuple(float(value) for value in head_keep_ratios)
        if not ratios:
            raise ValueError("sparse Piper head keep ratios must be a nonempty vector")

    if any(not math.isfinite(ratio) or not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("sparse Piper head keep ratios must be finite and lie in (0, 1]")
    units = tuple(round(ratio * _RATIO_SCALE) for ratio in ratios)
    if any(not 1 <= value <= _RATIO_SCALE for value in units):
        raise ValueError("sparse Piper head keep ratios exceed supported precision")
    return units


def _resolve_head_keep_blocks(
    head_keep_ratio_units: tuple[int, ...],
    num_blocks: int,
) -> tuple[int, ...]:
    """Round one region's ratios to the nearest feasible total without reducing any floor."""
    if not head_keep_ratio_units:
        raise ValueError("sparse Piper ratio profile must be nonempty")
    if any(
        isinstance(units, bool) or not isinstance(units, int) for units in head_keep_ratio_units
    ):
        raise TypeError("sparse Piper ratio profile must use integer fixed-point values")
    if any(not 1 <= units <= _RATIO_SCALE for units in head_keep_ratio_units):
        raise ValueError("sparse Piper ratio profile contains an invalid fixed-point value")

    numerators = tuple(num_blocks * units for units in head_keep_ratio_units)
    head_keep_blocks = [max(1, numerator // _RATIO_SCALE) for numerator in numerators]
    target_total = (sum(numerators) + _RATIO_SCALE // 2) // _RATIO_SCALE
    target_total = min(
        len(head_keep_blocks) * num_blocks,
        max(sum(head_keep_blocks), target_total),
    )
    remaining = target_total - sum(head_keep_blocks)
    fractional_order = sorted(
        range(len(head_keep_blocks)),
        key=lambda head: (-(numerators[head] % _RATIO_SCALE), head),
    )
    for head in fractional_order:
        if remaining == 0:
            break
        if head_keep_blocks[head] < num_blocks:
            head_keep_blocks[head] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("sparse Piper could not realize the requested physical head budget")
    return tuple(head_keep_blocks)
