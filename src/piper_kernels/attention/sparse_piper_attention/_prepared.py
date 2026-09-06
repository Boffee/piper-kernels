"""Shared quantized sparse-Piper storage and validation; no accelerator runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import (
    HEAD_DIM,
    QUERY_SCALE_ROWS,
    TILE_ROWS,
)

from ._block_layout import validate_sparse_query_blocks


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperContext:
    """Sequence-global K/V storage and sparse-attention policy metadata."""

    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_mean: torch.Tensor
    route_head_offsets: torch.Tensor
    head_keep_blocks: torch.Tensor
    routes_per_query: int
    block_lengths: torch.Tensor | None
    sparse_key_blocks: int
    sparse_query_blocks: int | None
    logical_sequence_length: int


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperQuery:
    """Query-local quantized storage and routes at one global block offset."""

    data: torch.Tensor
    scale: torch.Tensor
    routes: torch.Tensor
    global_block_offset: int = 0


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperAttention:
    """Sequence-global context paired with one full or local query range."""

    context: _PreparedSparsePiperContext
    query: _PreparedSparsePiperQuery


def _prepare_sparse_piper_context_from_quantized(  # noqa: PLR0912, PLR0913
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_blocks: torch.Tensor,
    route_head_offsets: torch.Tensor,
    *,
    sparse_key_blocks: int,
    routes_per_query: int,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
) -> _PreparedSparsePiperContext:
    """Validate sequence-global quantized K/V storage and routing policy.

    ``block_lengths`` opts into internally padded K64 storage. Each entry gives
    the valid prefix length in ``[1, 64]`` of one physical block and supersedes
    ``logical_sequence_length`` for masking. Without it, storage is the padded
    form of the compact logical sequence with at most one ragged tail.
    """
    if key.ndim != 4 or key.dtype is not torch.int8:
        raise ValueError(
            "quantized sparse Piper K must be [batch,heads,storage_sequence,D128] INT8"
        )
    batch, heads, storage_sequence_length, head_dim = key.shape
    if (
        head_dim != HEAD_DIM
        or storage_sequence_length < TILE_ROWS
        or storage_sequence_length % TILE_ROWS
    ):
        raise ValueError("quantized sparse Piper requires K64-aligned D128 key storage")
    if block_lengths is None:
        if (
            logical_sequence_length < TILE_ROWS
            or logical_sequence_length > storage_sequence_length
            or (logical_sequence_length + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS
            != storage_sequence_length
        ):
            raise ValueError("quantized sparse Piper storage must be the padded logical sequence")
    elif not 1 <= logical_sequence_length <= storage_sequence_length:
        raise ValueError("quantized sparse Piper logical length must fit block-length storage")
    if (
        value.shape != (batch, heads, head_dim, storage_sequence_length)
        or value.dtype is not torch.int8
    ):
        raise ValueError(
            "quantized sparse Piper V must be transposed INT8 [B,H,D,storage_sequence]"
        )
    tile_count = storage_sequence_length // TILE_ROWS
    if key_scale.shape != (batch, heads, tile_count):
        raise ValueError("quantized sparse Piper K scales must contain one value per K64")
    if value_scale_multiplier.shape != (batch, heads, tile_count, 1):
        raise ValueError("quantized sparse Piper V scales must contain one value per K64")
    if value_mean.shape != (batch, heads, head_dim):
        raise ValueError("quantized sparse Piper V mean must be [batch,heads,D128]")
    # Layout construction owns the value-range invariants documented
    # above. Inspecting device values here would add a validation kernel or a host
    # synchronization to every launch; only launch-critical tensor properties
    # are checked on this hot path.
    if block_lengths is not None and (
        block_lengths.shape != (tile_count,) or block_lengths.dtype is not torch.int32
    ):
        raise ValueError("quantized sparse Piper block lengths must be one INT32 value per K64")
    scales = key_scale, value_scale_multiplier, value_mean
    if any(scale.dtype is not torch.float32 for scale in scales):
        raise ValueError("quantized sparse Piper scales and V mean must use FP32")
    tensors = (
        key,
        value,
        *scales,
        head_keep_blocks,
        route_head_offsets,
        *((block_lengths,) if block_lengths is not None else ()),
    )
    if any(tensor.device != key.device for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must share a device")
    if any(tensor.layout is not torch.strided or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must be contiguous strided tensors")
    if not 1 <= sparse_key_blocks <= tile_count:
        raise ValueError("quantized sparse Piper prefix must fit the K64 tile count")
    total_query_blocks = (
        tile_count
        if block_lengths is not None
        else (logical_sequence_length + TILE_ROWS - 1) // TILE_ROWS
    )
    validate_sparse_query_blocks(
        sparse_query_blocks,
        query_blocks=total_query_blocks,
        context="quantized sparse Piper",
    )
    if head_keep_blocks.shape != (heads,) or head_keep_blocks.dtype is not torch.int32:
        raise ValueError("quantized sparse Piper head keep blocks must be one INT32 value per head")
    if route_head_offsets.shape != (heads + 1,) or route_head_offsets.dtype is not torch.int32:
        raise ValueError("quantized sparse Piper route offsets must be an INT32 head vector")

    return _PreparedSparsePiperContext(
        key=key,
        value=value,
        key_scale=key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_mean=value_mean,
        route_head_offsets=route_head_offsets,
        head_keep_blocks=head_keep_blocks,
        routes_per_query=routes_per_query,
        block_lengths=block_lengths,
        sparse_key_blocks=sparse_key_blocks,
        sparse_query_blocks=sparse_query_blocks,
        logical_sequence_length=logical_sequence_length,
    )


def _prepare_sparse_piper_query_from_quantized(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    routes: torch.Tensor,
    context: _PreparedSparsePiperContext,
    *,
    global_block_offset: int = 0,
) -> _PreparedSparsePiperQuery:
    """Validate query-local quantized storage and locate it globally."""
    if query.ndim != 4 or query.dtype is not torch.int8:
        raise ValueError(
            "quantized sparse Piper Q must be [batch,heads,storage_sequence,D128] INT8"
        )
    batch, heads, storage_sequence_length, head_dim = query.shape
    if (
        query.shape[:2] != context.key.shape[:2]
        or head_dim != HEAD_DIM
        or storage_sequence_length < TILE_ROWS
        or storage_sequence_length % TILE_ROWS
    ):
        raise ValueError("quantized sparse Piper requires compatible K64-aligned D128 Q storage")
    query_block_count = storage_sequence_length // TILE_ROWS
    total_query_blocks = context.key.shape[2] // TILE_ROWS
    if (
        isinstance(global_block_offset, bool)
        or not isinstance(global_block_offset, int)
        or global_block_offset < 0
        or global_block_offset + query_block_count > total_query_blocks
    ):
        raise ValueError("quantized sparse Piper Q storage must fit the global sequence")
    if query_scale.shape != (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS):
        raise ValueError("quantized sparse Piper Q scales must contain one value per Q32")
    if routes.shape != (batch, query_block_count, context.routes_per_query):
        raise ValueError("quantized sparse Piper routes must match batch/query/packed budgets")
    if query_scale.dtype is not torch.float32:
        raise ValueError("quantized sparse Piper Q scales must use FP32")
    if routes.dtype is not torch.uint16:
        raise ValueError("quantized sparse Piper routes must use UINT16")
    tensors = query, query_scale, routes
    if any(tensor.device != context.key.device for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must share a device")
    if any(tensor.layout is not torch.strided or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must be contiguous strided tensors")
    return _PreparedSparsePiperQuery(
        data=query,
        scale=query_scale,
        routes=routes,
        global_block_offset=global_block_offset,
    )
