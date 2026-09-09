"""Shared validation of caller-owned sparse-attention launch ranges."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS, TILE_ROWS

from ._dtype import SUPPORTED_DTYPES
from ._prepared import _PreparedSparsePiperAttention


def _resolve_query_block_range(
    prepared: _PreparedSparsePiperAttention,
    query_block_offset: int,
    query_block_count: int | None,
) -> tuple[int, int, int]:
    """Validate a local launch range and return its size and global offset."""
    query_state = prepared.query
    stored_query_blocks = query_state.data.shape[2] // TILE_ROWS
    if isinstance(query_block_offset, bool) or not isinstance(query_block_offset, int):
        raise TypeError("sparse Piper query block offset must be an integer")
    if query_block_count is not None and (
        isinstance(query_block_count, bool) or not isinstance(query_block_count, int)
    ):
        raise TypeError("sparse Piper query block count must be an integer or None")
    if not 0 <= query_block_offset < stored_query_blocks:
        raise ValueError("sparse Piper query block offset must fit the prepared query storage")
    resolved_query_block_count = (
        stored_query_blocks - query_block_offset if query_block_count is None else query_block_count
    )
    if (
        resolved_query_block_count < 1
        or query_block_offset + resolved_query_block_count > stored_query_blocks
    ):
        raise ValueError("sparse Piper query block range must fit the prepared query storage")
    return (
        stored_query_blocks,
        resolved_query_block_count,
        query_state.global_block_offset + query_block_offset,
    )


def validate_attention_launch(
    prepared: _PreparedSparsePiperAttention,
    output: torch.Tensor,
    query_block_offset: int,
    query_block_count: int | None,
    coarse_output: torch.Tensor | None,
    coarse_gate: torch.Tensor | None,
) -> tuple[int, int]:
    """Validate output/coarse tensors; return block count and global offset."""
    query_state = prepared.query
    context = prepared.context
    query = query_state.data
    batch, heads, query_storage_sequence_length, head_dim = query.shape
    logical_sequence_length = context.logical_sequence_length
    storage_sequence_length = context.key.shape[2]
    has_block_lengths = context.block_lengths is not None
    if (
        head_dim not in SUPPORTED_HEAD_DIMS
        or query_storage_sequence_length < TILE_ROWS
        or query_storage_sequence_length % TILE_ROWS
        or storage_sequence_length < TILE_ROWS
        or storage_sequence_length % TILE_ROWS
        or (
            not has_block_lengths
            and (logical_sequence_length + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS
            != storage_sequence_length
        )
    ):
        raise ValueError("sparse Piper requires padded Q64/K64/D64/D128 storage")
    stored_query_blocks, resolved_query_block_count, global_query_block_offset = (
        _resolve_query_block_range(
            prepared,
            query_block_offset,
            query_block_count,
        )
    )
    output_sequence_length = (
        resolved_query_block_count * TILE_ROWS
        if has_block_lengths
        else min(
            resolved_query_block_count * TILE_ROWS,
            logical_sequence_length - global_query_block_offset * TILE_ROWS,
        )
    )
    if (
        output.shape != (batch, heads, output_sequence_length, head_dim)
        or output.dtype not in SUPPORTED_DTYPES
        or output.device != query.device
        or output.stride(-1) != 1
    ):
        raise ValueError("sparse Piper output must match the query block range")
    if context.value_scale_multiplier.shape[-1] != 1:
        raise ValueError("sparse Piper requires one folded scale per K64 tile")
    has_coarse_residual = coarse_output is not None or coarse_gate is not None
    if (coarse_output is None) != (coarse_gate is None):
        raise ValueError("coarse output and coarse gate must be supplied together")
    if has_coarse_residual:
        assert coarse_output is not None
        assert coarse_gate is not None
        if (
            coarse_output.shape != (batch, heads, stored_query_blocks, head_dim)
            or coarse_output.dtype is not torch.float32
            or coarse_output.device != query.device
            or coarse_output.stride(-1) != 1
        ):
            raise ValueError("sparse Piper coarse output must be FP32 [batch,heads,Q64,D64/D128]")
        if (
            coarse_gate.shape != (batch, output_sequence_length, heads, head_dim)
            or coarse_gate.dtype is not output.dtype
            or coarse_gate.device != query.device
            or coarse_gate.stride(-1) != 1
        ):
            raise ValueError("sparse Piper coarse gate must match the local attention output")
    return resolved_query_block_count, global_query_block_offset
