"""Compatibility imports for sparse-Piper's shared operand layout."""

from __future__ import annotations

from piper_kernels.attention.kernels.sparse_piper.layout import (
    HEAD_DIM,
    QUERY_SCALE_ROWS,
    TILE_ROWS,
    padded_sequence_length,
)

__all__ = ["HEAD_DIM", "QUERY_SCALE_ROWS", "TILE_ROWS", "padded_sequence_length"]
