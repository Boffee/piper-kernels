"""Logical and storage layout shared by sparse-Piper operand producers."""

from __future__ import annotations

from enum import IntEnum

SUPPORTED_HEAD_DIMS = (64, 128)
QUERY_SCALE_ROWS = 32
TILE_ROWS = 64


class ValueFormat(IntEnum):
    """Final INT8 V storage, independent of its K64 quantization groups.

    Q and K use contiguous [B,H,S,D] storage on every supported backend.
    TRANSPOSED is [B,H,D,S]. AMD_WMMA is [B,H,S/64,D,64], with token
    bits 3 and 4 exchanged within each tile for the WMMA reduction order.
    S is padded to 64; H is the KV head count, including for grouped queries.
    """

    TRANSPOSED = 0
    AMD_WMMA = 1

    def shape(self, batch: int, heads: int, sequence: int, head_dim: int) -> tuple[int, ...]:
        """Physical shape for a padded logical [B,H,S,D] value tensor."""
        if self is ValueFormat.AMD_WMMA:
            return batch, heads, sequence // TILE_ROWS, head_dim, TILE_ROWS
        return batch, heads, head_dim, sequence


def padded_sequence_length(sequence_length: int) -> int:
    """Round a logical sequence length up to sparse Piper's K64 storage."""
    return (sequence_length + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS


__all__ = [
    "QUERY_SCALE_ROWS",
    "SUPPORTED_HEAD_DIMS",
    "TILE_ROWS",
    "ValueFormat",
    "padded_sequence_length",
]
