"""Shared validation of caller-owned sparse-attention launch ranges."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS, TILE_ROWS

from ._dtype import SUPPORTED_DTYPES
from ._prepared import _PreparedSparsePiperAttention

_DO_NOT_SPECIALIZE_ARGUMENTS = (
    "query_block_offset",
    "global_query_block_offset",
    "logical_sequence_length",
    "sparse_key_blocks",
    "sparse_query_blocks",
    "stride_rb",
    "stride_rq",
    "output_sequence_length",
)


@dataclass(frozen=True, slots=True)
class _ValidatedAttentionLaunch:
    """Backend-neutral kernel operands resolved by common launch validation."""

    batch: int
    heads: int
    head_dim: int
    query_block_offset: int
    query_block_count: int
    global_query_block_offset: int
    query_storage_sequence_length: int
    storage_sequence_length: int
    output_sequence_length: int
    logical_sequence_length: int
    sparse_key_blocks: int
    sparse_query_blocks: int
    route_strides: tuple[int, int, int]
    output_strides: tuple[int, int, int]
    coarse_strides: tuple[int, int, int]
    gate_strides: tuple[int, int, int]
    coarse_output: torch.Tensor
    coarse_gate: torch.Tensor
    block_lengths: torch.Tensor
    mask_block_lengths: bool
    has_dense_query_suffix: bool
    apply_coarse_residual: bool
    skip_dense_routing: bool

    @property
    def query_rows(self) -> int:
        """Return the number of output rows covered by this launch."""
        return self.query_block_count * TILE_ROWS


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
) -> _ValidatedAttentionLaunch:
    """Validate tensors and resolve the common metadata consumed by native kernels."""
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
    has_dense_query_suffix = context.sparse_query_blocks is not None
    sparse_query_blocks = (
        storage_sequence_length // TILE_ROWS
        if context.sparse_query_blocks is None
        else context.sparse_query_blocks
    )
    resolved_coarse_output = context.value_mean if coarse_output is None else coarse_output
    resolved_coarse_gate = output if coarse_gate is None else coarse_gate
    block_lengths_operand = (
        context.head_keep_blocks if context.block_lengths is None else context.block_lengths
    )
    return _ValidatedAttentionLaunch(
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        query_block_offset=query_block_offset,
        query_block_count=resolved_query_block_count,
        global_query_block_offset=global_query_block_offset,
        query_storage_sequence_length=query_storage_sequence_length,
        storage_sequence_length=storage_sequence_length,
        output_sequence_length=output_sequence_length,
        logical_sequence_length=logical_sequence_length,
        sparse_key_blocks=context.sparse_key_blocks,
        sparse_query_blocks=sparse_query_blocks,
        route_strides=(
            query_state.routes.stride(0),
            query_state.routes.stride(1),
            query_state.routes.stride(2),
        ),
        output_strides=(output.stride(0), output.stride(1), output.stride(2)),
        coarse_strides=(
            (0, 0, 0)
            if coarse_output is None
            else (
                coarse_output.stride(0),
                coarse_output.stride(1),
                coarse_output.stride(2),
            )
        ),
        gate_strides=(
            (0, 0, 0)
            if coarse_gate is None
            else (coarse_gate.stride(0), coarse_gate.stride(2), coarse_gate.stride(1))
        ),
        coarse_output=resolved_coarse_output,
        coarse_gate=resolved_coarse_gate,
        block_lengths=block_lengths_operand,
        mask_block_lengths=has_block_lengths,
        has_dense_query_suffix=has_dense_query_suffix,
        apply_coarse_residual=has_coarse_residual,
        skip_dense_routing=context.routes_per_query == 0,
    )
