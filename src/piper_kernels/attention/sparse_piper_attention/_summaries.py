"""Shared block-summary orchestration for sparse Piper routing."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from . import _backend
from ._block_layout import valid_block_rows, validate_block_lengths
from ._routing_modes import (
    _MEAN_ROUTING,
    validate_routing_mode,
)
from .coarse import _mean_pool_head_major_blocks


def sequence_block_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    *,
    global_block_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy-specific Q/K summaries through one fixed tensor contract.

    ``block_lengths`` describes the global K64 layout. K is its leading prefix,
    and Q holds the blocks that start at global block ``global_block_offset``.
    """
    validate_routing_mode(routing_mode)
    query_block_lengths, key_block_lengths = _validate_sequences(
        query, key, block_lengths, global_block_offset=global_block_offset
    )
    summarize = _backend.select_sequence_summaries(query, key)
    if summarize is not None:
        return summarize(query, key, routing_mode, query_block_lengths, key_block_lengths)

    if routing_mode == _MEAN_ROUTING:
        query_summary = _mean_pool_head_major_blocks(query, query_block_lengths)
        key_primary = _mean_pool_head_major_blocks(key, key_block_lengths)
        return query_summary, key_primary, key_primary[:, :, :0]

    if query_block_lengths is None or key_block_lengths is None:
        query_max, query_min = _compact_block_extrema(query)
        key_primary, key_aux = _compact_block_extrema(key)
    else:
        query_max, query_min = _padded_block_extrema(query, query_block_lengths)
        key_primary, key_aux = _padded_block_extrema(key, key_block_lengths)
    return query_max + query_min, key_primary, key_aux


def _compact_block_extrema(sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 extrema for compact blocks, including a ragged tail."""
    extrema = []
    for start in range(0, sequence.shape[2], _BLOCK_ROWS):
        block = sequence[:, :, start : start + _BLOCK_ROWS].float()
        extrema.append((block.amax(dim=2), block.amin(dim=2)))
    maxima, minima = zip(*extrema, strict=True)
    return torch.stack(maxima, dim=2), torch.stack(minima, dim=2)


def _padded_block_extrema(
    sequence: torch.Tensor,
    block_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = sequence.unflatten(2, (block_lengths.numel(), _BLOCK_ROWS)).float()
    valid_rows = valid_block_rows(block_lengths)[None, None, :, :, None]
    maximum = torch.where(valid_rows, blocks, -float("inf")).amax(dim=3)
    minimum = torch.where(valid_rows, blocks, float("inf")).amin(dim=3)
    return maximum, minimum


def _validate_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    global_block_offset: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate Q/K metadata and return the Q and K block lengths, if padded."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("routing query and key sequences must be rank-four tensors")
    if (
        query.shape[0] != key.shape[0]
        or key.shape[1] < 1
        or query.shape[1] < 1
        or query.shape[1] % key.shape[1]
        or query.shape[-1] != key.shape[-1]
    ):
        raise ValueError("routing query and key batch/head/feature dimensions must match")
    if query.shape[2] < 1 or key.shape[2] < 1:
        raise ValueError("routing summaries require nonempty Q/K sequences")
    if query.device != key.device:
        raise ValueError("routing query and key sequences must share a device")
    if not query.is_floating_point() or not key.is_floating_point():
        raise TypeError("routing query and key sequences must be floating-point tensors")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("routing query and key features must be contiguous")
    if block_lengths is None:
        return None, None
    block_count = validate_block_lengths(
        block_lengths,
        sequence_length=None,
        device=query.device,
        context="padded sparse routing",
        require_contiguous=True,
    )
    if query.shape[2] % _BLOCK_ROWS or key.shape[2] % _BLOCK_ROWS:
        raise ValueError("padded routing Q/K must use complete physical K64 blocks")
    end_block = global_block_offset + query.shape[2] // _BLOCK_ROWS
    if global_block_offset < 0 or end_block > block_count:
        raise ValueError("padded routing queries must lie within the block-length layout")
    if key.shape[2] // _BLOCK_ROWS > block_count:
        raise ValueError("padded routing keys cannot exceed the block-length layout")
    return (
        block_lengths[global_block_offset:end_block],
        block_lengths[: key.shape[2] // _BLOCK_ROWS],
    )


__all__ = ["sequence_block_summaries"]
