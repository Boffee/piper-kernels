"""Shared ConvRot INT8 sparse-Piper operand layout and metadata validation."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import (
    QUERY_SCALE_ROWS,
    SUPPORTED_HEAD_DIMS,
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


def resolve_row_block_launches(
    sequence_rows: int,
    block_m: int,
    *,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> tuple[tuple[int, int, bool], ...]:
    """Return ``(row_blocks, row_block_offset, aligned_rows)`` launches for full or partial runs."""
    full_row_blocks = sequence_rows // block_m
    if start_row_block != 0 or num_row_blocks is not None:
        if start_row_block < 0:
            raise ValueError("start_row_block must be non-negative")
        if num_row_blocks is None:
            if sequence_rows % block_m != 0:
                raise ValueError("partial row-block launches require full block_m alignment")
            block_count = full_row_blocks - start_row_block
        else:
            block_count = num_row_blocks
        if block_count < 0 or start_row_block + block_count > full_row_blocks:
            raise ValueError(
                "partial row-block range must stay within full block_m blocks of sequence_length"
            )
        return ((block_count, start_row_block, True),) if block_count > 0 else ()

    launches: list[tuple[int, int, bool]] = []
    if full_row_blocks:
        launches.append((full_row_blocks, 0, True))
    if sequence_rows % block_m:
        launches.append((1, full_row_blocks, False))
    return tuple(launches)


__all__ = [
    "QUERY_SCALE_ROWS",
    "SUPPORTED_HEAD_DIMS",
    "TILE_ROWS",
    "padded_sequence_length",
    "resolve_row_block_launches",
    "validate_block_lengths",
]
