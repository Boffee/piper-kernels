"""Projection-independent bounded sparse-attention output orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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

_output_sequence_length = _quantized_dispatch._quantized_attention_output_sequence_length


@dataclass(frozen=True, slots=True)
class _PreparedAttentionOutput:
    """Shared launch state for one bounded attention-to-projection pipeline."""

    attention: _PreparedSparsePiperAttention
    sequence_length: int
    coarse_output: torch.Tensor | None
    compression_gate: torch.Tensor | None


def new_projected_output(
    query: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    output_features: int,
) -> torch.Tensor:
    """Allocate the projected compact or valid-front padded output shape."""
    return query.new_empty(
        (
            query.shape[0],
            _output_sequence_length(query, logical_sequence_length, block_lengths),
            output_features,
        ),
        dtype=torch.bfloat16,
    )


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
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    compression_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
) -> _PreparedAttentionOutput:
    """Prepare fine routing and, when requested, one shared coarse result."""
    if (block_mean is None) != (compression_gate is None):
        raise ValueError("block means and compression gate must be supplied together")
    if block_mean is None:
        if coarse_scale is not None or coarse_key_blocks is not None:
            raise ValueError("coarse output metadata requires block means")
        prepared = _quantized_dispatch._prepare_quantized_sparse_piper_attention(
            query,
            query_scale,
            query_summary,
            key,
            key_scale,
            key_summary,
            key_aux,
            value,
            value_scale_multiplier,
            value_mean,
            head_keep_ratio_units,
            sparse_key_blocks,
            logical_sequence_length,
            routing_mode,
            block_lengths,
            sparse_query_blocks,
        )
        coarse_output = None
    else:
        if coarse_scale is None:
            raise ValueError("coarse output requires a scale")
        prepared, coarse_output = (
            _quantized_dispatch._prepare_quantized_sparse_piper_attention_with_coarse(
                query,
                query_scale,
                query_summary,
                key,
                key_scale,
                key_summary,
                key_aux,
                value,
                value_scale_multiplier,
                value_mean,
                block_mean,
                head_keep_ratio_units,
                sparse_key_blocks,
                logical_sequence_length,
                routing_mode,
                coarse_scale,
                block_lengths,
                coarse_key_blocks,
                sparse_query_blocks,
            )
        )
    return _PreparedAttentionOutput(
        attention=prepared,
        sequence_length=_output_sequence_length(
            query,
            logical_sequence_length,
            block_lengths,
        ),
        coarse_output=coarse_output,
        compression_gate=compression_gate,
    )


def run_chunked_attention_output(
    prepared: _PreparedAttentionOutput,
    output_features: int,
    query_chunk_rows: int,
    project_chunk: ChunkProjector,
    projector_tensors: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Pipeline bounded attention chunks into one format-specific projection."""
    prepared_attention = prepared.attention
    query = prepared_attention.query
    chunk_blocks = query_chunk_rows // TILE_ROWS
    sequence_length = prepared.sequence_length
    total_blocks = (sequence_length + TILE_ROWS - 1) // TILE_ROWS
    chunk_count = (total_blocks + chunk_blocks - 1) // chunk_blocks
    capacity = min(sequence_length, query_chunk_rows)
    batch, heads, head_dim = query.shape[0], query.shape[1], query.shape[3]
    attention_buffers = torch.empty(
        (min(2, chunk_count), batch, capacity, heads, head_dim),
        device=query.device,
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (batch, sequence_length, output_features),
        device=query.device,
        dtype=torch.bfloat16,
    )

    if chunk_count == 1:
        attention_chunk = attention_buffers[0]
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            prepared_attention,
            attention_chunk.transpose(1, 2),
            coarse_output=prepared.coarse_output,
            compression_gate=prepared.compression_gate,
        )
        project_chunk(attention_chunk, output, 0, sequence_length)
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
            rows = min(block_count * TILE_ROWS, sequence_length - start)
            attention_chunk = attention_buffers[slot, :, :rows]
            _quantized_dispatch._launch_quantized_sparse_piper_attention(
                prepared_attention,
                attention_chunk.transpose(1, 2),
                query_block_offset=block_start,
                query_block_count=block_count,
                coarse_output=prepared.coarse_output,
                compression_gate=prepared.compression_gate,
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
    "new_projected_output",
    "prepare_attention",
    "run_chunked_attention_output",
    "validate_attention_output",
]
