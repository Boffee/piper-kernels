"""Logical and storage layout shared by sparse-Piper operand producers."""

from __future__ import annotations

HEAD_DIM = 128
QUERY_SCALE_ROWS = 32
TILE_ROWS = 64


def padded_sequence_length(sequence_length: int) -> int:
    """Round a logical sequence length up to sparse Piper's K64 storage."""
    return (sequence_length + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS


__all__ = ["HEAD_DIM", "QUERY_SCALE_ROWS", "TILE_ROWS", "padded_sequence_length"]
