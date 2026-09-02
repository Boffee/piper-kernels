"""Semantic head budgets and private packed-route layout resolution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

_RATIO_SCALE = 1_000_000
_UINT16_ROUTE_CAPACITY = 1 << 16


@dataclass(frozen=True, slots=True)
class _ResolvedRouteLayout:
    """Layout-specific integer metadata consumed by DSA and attention."""

    head_keep_blocks: torch.Tensor
    route_head_offsets: torch.Tensor
    routes_per_query: int


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
    sparse_key_blocks: int,
) -> tuple[int, ...]:
    """Round ratios to the nearest feasible aggregate without reducing any floor."""
    if not head_keep_ratio_units:
        raise ValueError("sparse Piper ratio profile must be nonempty")
    if any(
        isinstance(units, bool) or not isinstance(units, int) for units in head_keep_ratio_units
    ):
        raise TypeError("sparse Piper ratio profile must use integer fixed-point values")
    if any(not 1 <= units <= _RATIO_SCALE for units in head_keep_ratio_units):
        raise ValueError("sparse Piper ratio profile contains an invalid fixed-point value")
    if isinstance(sparse_key_blocks, bool) or not isinstance(sparse_key_blocks, int):
        raise TypeError("sparse Piper sparse key block count must be an integer")
    if not 1 <= sparse_key_blocks <= _UINT16_ROUTE_CAPACITY:
        raise ValueError("sparse Piper sparse key block count must lie in [1, 65,536]")

    numerators = tuple(sparse_key_blocks * units for units in head_keep_ratio_units)
    head_keep_blocks = [max(1, numerator // _RATIO_SCALE) for numerator in numerators]
    target_total = (sum(numerators) + _RATIO_SCALE // 2) // _RATIO_SCALE
    target_total = min(
        len(head_keep_blocks) * sparse_key_blocks,
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
        if head_keep_blocks[head] < sparse_key_blocks:
            head_keep_blocks[head] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("sparse Piper could not realize the requested physical head budget")
    return tuple(head_keep_blocks)


def _resolve_route_layout(
    head_keep_ratio_units: tuple[int, ...],
    sparse_key_blocks: int,
    device: torch.device,
) -> _ResolvedRouteLayout:
    head_keep_block_values = _resolve_head_keep_blocks(
        head_keep_ratio_units,
        sparse_key_blocks,
    )
    route_head_offset_values = [0]
    for count in head_keep_block_values:
        route_head_offset_values.append(route_head_offset_values[-1] + count)
    if route_head_offset_values[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("sparse Piper packed routes exceed INT32 offset capacity")

    metadata = torch.tensor(
        (*head_keep_block_values, *route_head_offset_values),
        device=device,
        dtype=torch.int32,
    )
    heads = len(head_keep_block_values)
    return _ResolvedRouteLayout(
        head_keep_blocks=metadata[:heads],
        route_head_offsets=metadata[heads:],
        routes_per_query=route_head_offset_values[-1],
    )
