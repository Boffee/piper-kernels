"""Shared K64 layout and query-scope helpers for sparse Piper attention."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS


def validate_block_lengths(
    block_lengths: torch.Tensor,
    *,
    sequence_length: int,
    device: torch.device,
    context: str,
    require_contiguous: bool = False,
    check_values: bool = True,
) -> int:
    """Validate one valid-prefix length per physical K64 block."""
    if (
        block_lengths.ndim != 1
        or block_lengths.numel() < 1
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != device
        or sequence_length != block_lengths.numel() * TILE_ROWS
        or (require_contiguous and not block_lengths.is_contiguous())
    ):
        contiguity = " contiguous" if require_contiguous else ""
        raise ValueError(
            f"{context} block lengths must be one{contiguity} device INT32 value per K64"
        )
    if check_values and not torch.compiler.is_compiling():
        torch._assert_async(
            torch.all((block_lengths >= 1) & (block_lengths <= TILE_ROWS)),
            f"{context} block lengths must lie in [1, {TILE_ROWS}]",
        )
    return block_lengths.numel()


def valid_block_rows(block_lengths: torch.Tensor) -> torch.Tensor:
    """Return the valid-prefix mask ``[blocks, 64]`` described by lengths."""
    return torch.arange(TILE_ROWS, device=block_lengths.device)[None, :] < block_lengths[:, None]


def validate_sparse_query_blocks(
    sparse_query_blocks: int | None,
    *,
    query_blocks: int,
    context: str,
) -> None:
    """Validate an optional leading sparse-query block count."""
    if sparse_query_blocks is None:
        return
    if isinstance(sparse_query_blocks, bool):
        raise TypeError(f"{context} sparse_query_blocks must be an integer or None")
    if torch.compiler.is_compiling():
        torch._check(
            sparse_query_blocks >= 0,
            lambda: f"{context} sparse_query_blocks must be nonnegative",
        )
        torch._check(
            sparse_query_blocks <= query_blocks,
            lambda: f"{context} sparse_query_blocks cannot exceed the query block count",
        )
        return
    if not isinstance(sparse_query_blocks, int):
        raise TypeError(f"{context} sparse_query_blocks must be an integer or None")
    if not 0 <= sparse_query_blocks <= query_blocks:
        raise ValueError(f"{context} sparse_query_blocks must fit the query block count")


__all__ = [
    "valid_block_rows",
    "validate_block_lengths",
    "validate_sparse_query_blocks",
]
