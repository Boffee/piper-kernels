"""Shared ConvRot sparse-Piper operand layout and metadata validation."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import (
    HEAD_DIM,
    QUERY_SCALE_ROWS,
    TILE_ROWS,
    padded_sequence_length,
)


def validate_block_lengths(
    block_lengths: torch.Tensor | None,
    sequence_length: int,
    device: torch.device,
) -> None:
    """Validate trusted valid-prefix metadata for aligned physical storage."""
    if block_lengths is None:
        return
    if (
        sequence_length % TILE_ROWS
        or block_lengths.shape != (sequence_length // TILE_ROWS,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != device
        or not block_lengths.is_contiguous()
    ):
        raise ValueError("block lengths must be one contiguous device INT32 value per K64")


__all__ = [
    "HEAD_DIM",
    "QUERY_SCALE_ROWS",
    "TILE_ROWS",
    "padded_sequence_length",
    "validate_block_lengths",
]
