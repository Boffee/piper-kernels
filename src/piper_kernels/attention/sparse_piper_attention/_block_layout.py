"""Shared K64 layout and query-scope helpers for sparse Piper attention."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS


def validate_block_lengths(
    block_lengths: torch.Tensor,
    *,
    sequence_length: int | None,
    device: torch.device,
    context: str,
    require_contiguous: bool = False,
    check_values: bool = True,
) -> int:
    """Validate one valid-prefix length per physical K64 block.

    ``sequence_length`` names the physical rows the layout must describe exactly;
    None accepts any nonempty layout, for callers that check block ranges themselves.
    """
    if (
        block_lengths.ndim != 1
        or block_lengths.numel() < 1
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != device
        or (sequence_length is not None and sequence_length != block_lengths.numel() * TILE_ROWS)
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


def validate_query_slice(
    global_row_offset: int,
    num_query_rows: int,
    *,
    sequence_length: int,
    has_block_lengths: bool,
    context: str,
) -> None:
    """Validate where a query slice's rows lie in the global sequence.

    The slice holds global rows ``[global_row_offset, global_row_offset + num_query_rows)``.
    ``sequence_length`` counts the global rows: logical rows for compact storage,
    physical rows with ``block_lengths``. The range starts on a K64 block and covers
    whole blocks; only a compact range that ends the sequence may end in a partial one.
    """
    if isinstance(global_row_offset, bool):
        raise TypeError(f"{context} global_row_offset must be an integer")
    if torch.compiler.is_compiling():
        end_row = global_row_offset + num_query_rows
        torch._check(
            global_row_offset % TILE_ROWS == 0,
            lambda: f"{context} global_row_offset must be a multiple of {TILE_ROWS}",
        )
        torch._check(
            global_row_offset >= 0,
            lambda: f"{context} global_row_offset must be nonnegative",
        )
        torch._check(
            num_query_rows >= 1,
            lambda: f"{context} query rows must be nonempty",
        )
        torch._check(
            end_row <= sequence_length,
            lambda: f"{context} query rows must fit the global sequence",
        )
        whole_blocks = num_query_rows % TILE_ROWS == 0
        torch._check(
            whole_blocks if has_block_lengths else whole_blocks | (end_row == sequence_length),
            lambda: f"{context} query rows must cover whole K64 blocks",
        )
        return
    if not isinstance(global_row_offset, int):
        raise TypeError(f"{context} global_row_offset must be an integer")
    if global_row_offset % TILE_ROWS:
        raise ValueError(f"{context} global_row_offset must be a multiple of {TILE_ROWS}")
    end_row = global_row_offset + num_query_rows
    if global_row_offset < 0 or num_query_rows < 1 or end_row > sequence_length:
        raise ValueError(
            f"{context} query rows [{global_row_offset}, {end_row}) must fit the "
            f"{sequence_length}-row global sequence"
        )
    if num_query_rows % TILE_ROWS and (has_block_lengths or end_row != sequence_length):
        raise ValueError(
            f"{context} query rows must cover whole K64 blocks; only compact rows that "
            "end the sequence may end in a partial block"
        )


def slice_query_block_lengths(
    block_lengths: torch.Tensor | None,
    global_block_offset: int,
    num_query_rows: int,
) -> torch.Tensor | None:
    """Return the block lengths of a validated query slice starting at a global block."""
    if block_lengths is None:
        return None
    return block_lengths[global_block_offset : global_block_offset + num_query_rows // TILE_ROWS]


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
    "slice_query_block_lengths",
    "valid_block_rows",
    "validate_block_lengths",
    "validate_query_slice",
    "validate_sparse_query_blocks",
]
