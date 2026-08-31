"""Projection-independent bounded sparse-attention output orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import HEAD_DIM, TILE_ROWS
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch

if TYPE_CHECKING:
    from piper_kernels.attention.sparse_piper_attention.triton import (
        _PreparedSparsePiperAttention,
    )

DEFAULT_QUERY_CHUNK_ROWS = 4_096

type ChunkProjector = Callable[[torch.Tensor, torch.Tensor, int, int], None]


def validate_attention_output(
    query: torch.Tensor,
    logical_sequence_length: int,
    query_chunk_rows: int,
) -> int:
    """Validate the common boundary and return the flattened head width."""
    if query.ndim != 4:
        raise ValueError("fused sparse Piper output requires four-dimensional quantized queries")
    batch, heads, _storage_sequence_length, head_dim = query.shape
    if batch < 1 or head_dim != HEAD_DIM:
        raise ValueError("fused sparse Piper output requires nonempty batches with D128 heads")
    if isinstance(logical_sequence_length, bool) or not isinstance(logical_sequence_length, int):
        raise TypeError("fused sparse Piper logical sequence length must be an integer")
    if (
        isinstance(query_chunk_rows, bool)
        or not isinstance(query_chunk_rows, int)
        or query_chunk_rows < TILE_ROWS
        or query_chunk_rows % TILE_ROWS
    ):
        raise ValueError("fused sparse Piper query chunk rows must be a positive multiple of 64")
    if query.device.type != "cuda":
        raise ValueError("fused sparse Piper output currently requires CUDA")
    target = AcceleratorTarget.from_device(query.device)
    if not target.is_cuda_capability(12, 0):
        raise ValueError("fused sparse Piper output requires exact NVIDIA SM120")
    return heads * head_dim


def prepare_attention(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
) -> _PreparedSparsePiperAttention:
    """Prepare routing once for a format-specific chunked output projection."""
    return _quantized_dispatch._prepare_quantized_sparse_piper_attention(
        query,
        query_scale,
        query_summary,
        key,
        key_scale,
        key_max,
        key_min,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
    )


def run_chunked_attention_output(
    prepared_attention: _PreparedSparsePiperAttention,
    query: torch.Tensor,
    logical_sequence_length: int,
    output_features: int,
    query_chunk_rows: int,
    project_chunk: ChunkProjector,
    projector_tensors: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Pipeline bounded attention chunks into one format-specific projection."""
    chunk_blocks = query_chunk_rows // TILE_ROWS
    total_blocks = (logical_sequence_length + TILE_ROWS - 1) // TILE_ROWS
    chunk_count = (total_blocks + chunk_blocks - 1) // chunk_blocks
    capacity = min(logical_sequence_length, query_chunk_rows)
    batch, heads, head_dim = query.shape[0], query.shape[1], query.shape[3]
    attention_buffers = torch.empty(
        (min(2, chunk_count), batch, capacity, heads, head_dim),
        device=query.device,
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (batch, logical_sequence_length, output_features),
        device=query.device,
        dtype=torch.bfloat16,
    )

    if chunk_count == 1:
        attention_chunk = attention_buffers[0]
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            prepared_attention,
            attention_chunk.transpose(1, 2),
        )
        project_chunk(attention_chunk, output, 0, logical_sequence_length)
        return output

    with torch.cuda.device(query.device):
        producer = torch.cuda.current_stream(query.device)
        consumer = torch.cuda.Stream(device=query.device)
        ready = (torch.cuda.Event(), torch.cuda.Event())
        reusable = (torch.cuda.Event(), torch.cuda.Event())
        last_slot = 0
        for chunk_index, block_start in enumerate(range(0, total_blocks, chunk_blocks)):
            slot = chunk_index % 2
            last_slot = slot
            if chunk_index >= 2:
                producer.wait_event(reusable[slot])
            block_count = min(chunk_blocks, total_blocks - block_start)
            start = block_start * TILE_ROWS
            rows = min(block_count * TILE_ROWS, logical_sequence_length - start)
            attention_chunk = attention_buffers[slot, :, :rows]
            _quantized_dispatch._launch_quantized_sparse_piper_attention(
                prepared_attention,
                attention_chunk.transpose(1, 2),
                query_block_offset=block_start,
                query_block_count=block_count,
            )
            ready[slot].record(producer)
            with torch.cuda.stream(consumer):
                consumer.wait_event(ready[slot])
                project_chunk(attention_chunk, output, start, rows)
                reusable[slot].record(consumer)
        producer.wait_event(reusable[last_slot])
        attention_buffers.record_stream(consumer)
        output.record_stream(consumer)
        for tensor in projector_tensors:
            tensor.record_stream(consumer)
    return output


__all__ = [
    "DEFAULT_QUERY_CHUNK_ROWS",
    "prepare_attention",
    "run_chunked_attention_output",
    "validate_attention_output",
]
