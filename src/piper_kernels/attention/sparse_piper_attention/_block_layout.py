"""Shared valid-prefix K64 layout helpers for sparse Piper attention."""

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


__all__ = [
    "valid_block_rows",
    "validate_block_lengths",
]
