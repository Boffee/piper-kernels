"""Resolved route layouts shared by sparse Piper routing tests."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from piper_kernels import SparsePiperKeyLayout, SparsePiperKeyRegion
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _ResolvedRouteLayout,
)
from piper_kernels.attention.sparse_piper_attention._key_layout import resolve_route_layout


def sparse_layout(num_blocks: int) -> SparsePiperKeyLayout:
    """Return one sparse region over every block."""
    return SparsePiperKeyLayout((SparsePiperKeyRegion(num_blocks),))


def route_layout(
    ratios: Sequence[float],
    key_layout: SparsePiperKeyLayout | int,
    device: torch.device | str = "cpu",
) -> _ResolvedRouteLayout:
    """Resolve ratios over a layout, or over one sparse region of that many blocks."""
    if isinstance(key_layout, int):
        key_layout = sparse_layout(key_layout)
    return resolve_route_layout(
        _normalize_head_keep_ratios(ratios), key_layout, torch.device(device)
    )


def counted_route_layout(
    counts: Sequence[int],
    num_blocks: int,
    device: torch.device | str = "cpu",
) -> _ResolvedRouteLayout:
    """Resolve one sparse region whose heads keep exactly ``counts`` blocks."""
    layout = route_layout([count / num_blocks for count in counts], num_blocks, device)
    # Oracles read budgets from the layout; pin the budgets each test names.
    assert layout.regions[0].head_keep_block_values == tuple(counts)
    return layout
