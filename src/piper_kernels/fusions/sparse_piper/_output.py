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
_MIN_PROJECTED_GATE_PIPELINE_CHUNKS = 8

type ChunkProjector = Callable[[torch.Tensor, torch.Tensor, int, int], None]
type CoarseGateChunkProjector = Callable[[torch.Tensor, int, int], None]

output_sequence_length = _quantized_dispatch._quantized_attention_output_sequence_length


@dataclass(frozen=True, slots=True)
class _PreparedAttentionOutput:
    """Shared launch state for one bounded attention-to-projection pipeline."""

    attention: _PreparedSparsePiperAttention
    sequence_length: int
    coarse_output: torch.Tensor | None
    coarse_gate: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _PingPongBuffers:
    """Two reusable chunk slots synchronized by produced and consumed events."""

    buffers: torch.Tensor
    produced: tuple[torch.cuda.Event, torch.cuda.Event]
    consumed: tuple[torch.cuda.Event, torch.cuda.Event]

    def acquire_for_write(
        self,
        stream: torch.cuda.Stream,
        chunk_index: int,
        rows: int,
    ) -> torch.Tensor:
        """Wait until a slot's previous reader has released it."""
        slot = chunk_index % 2
        if chunk_index >= 2:
            stream.wait_event(self.consumed[slot])
        return self.buffers[slot, :, :rows]

    def publish(self, stream: torch.cuda.Stream, chunk_index: int) -> None:
        """Publish one completed write to readers."""
        self.produced[chunk_index % 2].record(stream)

    def acquire_for_read(
        self,
        stream: torch.cuda.Stream,
        chunk_index: int,
        rows: int,
    ) -> torch.Tensor:
        """Wait for a slot's current contents and return its local view."""
        slot = chunk_index % 2
        stream.wait_event(self.produced[slot])
        return self.buffers[slot, :, :rows]

    def release(self, stream: torch.cuda.Stream, chunk_index: int) -> None:
        """Mark one consumed slot reusable by its writer."""
        self.consumed[chunk_index % 2].record(stream)


def _enqueue_gate_chunk(
    slots: _PingPongBuffers,
    stream: torch.cuda.Stream,
    project_chunk: CoarseGateChunkProjector,
    chunk_index: int,
    start: int,
    rows: int,
) -> None:
    """Project and publish one gate chunk on its side stream."""
    with torch.cuda.stream(stream):
        chunk = slots.acquire_for_write(stream, chunk_index, rows)
        project_chunk(chunk, start, rows)
        slots.publish(stream, chunk_index)


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
            output_sequence_length(query, logical_sequence_length, block_lengths),
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
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    *,
    has_projected_coarse_gate: bool = False,
) -> _PreparedAttentionOutput:
    """Prepare fine routing and, when requested, one shared coarse result."""
    if coarse_gate is not None and has_projected_coarse_gate:
        raise ValueError("coarse gate cannot be both materialized and projected")
    has_coarse_gate = coarse_gate is not None or has_projected_coarse_gate
    if (block_mean is not None) != has_coarse_gate:
        raise ValueError("block means and coarse gate must be supplied together")
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
        sequence_length=output_sequence_length(
            query,
            logical_sequence_length,
            block_lengths,
        ),
        coarse_output=coarse_output,
        coarse_gate=coarse_gate,
    )


def run_chunked_attention_output(  # noqa: PLR0915
    prepared: _PreparedAttentionOutput,
    output_features: int,
    query_chunk_rows: int,
    project_chunk: ChunkProjector,
    projector_tensors: Sequence[torch.Tensor],
    *,
    project_coarse_gate_chunk: CoarseGateChunkProjector | None = None,
) -> torch.Tensor:
    """Pipeline bounded attention chunks into one format-specific projection."""
    prepared_attention = prepared.attention
    query = prepared_attention.query
    chunk_blocks = query_chunk_rows // TILE_ROWS
    sequence_length = prepared.sequence_length
    total_blocks = (sequence_length + TILE_ROWS - 1) // TILE_ROWS
    chunk_ranges: list[tuple[int, int, int, int]] = []
    for block_start in range(0, total_blocks, chunk_blocks):
        block_count = min(chunk_blocks, total_blocks - block_start)
        start = block_start * TILE_ROWS
        rows = min(block_count * TILE_ROWS, sequence_length - start)
        chunk_ranges.append((block_start, block_count, start, rows))
    chunk_count = len(chunk_ranges)
    pipeline_projected_gate = (
        project_coarse_gate_chunk is not None and chunk_count >= _MIN_PROJECTED_GATE_PIPELINE_CHUNKS
    )
    capacity = min(sequence_length, query_chunk_rows)
    batch, heads, head_dim = query.shape[0], query.shape[1], query.shape[3]
    attention_buffers = torch.empty(
        (min(2, chunk_count), batch, capacity, heads, head_dim),
        device=query.device,
        dtype=torch.bfloat16,
    )
    has_coarse_residual = prepared.coarse_output is not None
    if has_coarse_residual and (prepared.coarse_gate is None) == (
        project_coarse_gate_chunk is None
    ):
        raise ValueError("coarse attention requires exactly one coarse gate source")
    if not has_coarse_residual and (
        prepared.coarse_gate is not None or project_coarse_gate_chunk is not None
    ):
        raise ValueError("coarse gate source requires coarse attention")
    coarse_gate_buffers = (
        torch.empty(
            (2 if pipeline_projected_gate else 1, batch, capacity, heads, head_dim),
            device=query.device,
            dtype=torch.bfloat16,
        )
        if project_coarse_gate_chunk is not None
        else None
    )
    output = torch.empty(
        (batch, sequence_length, output_features),
        device=query.device,
        dtype=torch.bfloat16,
    )

    def coarse_gate_chunk(start: int, rows: int) -> torch.Tensor | None:
        if prepared.coarse_gate is not None:
            return prepared.coarse_gate[:, start : start + rows]
        if project_coarse_gate_chunk is None:
            return None
        assert coarse_gate_buffers is not None
        chunk = coarse_gate_buffers[0, :, :rows]
        project_coarse_gate_chunk(chunk, start, rows)
        return chunk

    if chunk_count == 1:
        attention_chunk = attention_buffers[0]
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            prepared_attention,
            attention_chunk.transpose(1, 2),
            coarse_output=prepared.coarse_output,
            coarse_gate=coarse_gate_chunk(0, sequence_length),
        )
        project_chunk(attention_chunk, output, 0, sequence_length)
        return output

    with torch.cuda.device(query.device):
        producer = torch.cuda.current_stream(query.device)
        consumer = torch.cuda.Stream(device=query.device)
        attention_slots = _PingPongBuffers(
            attention_buffers,
            (torch.cuda.Event(), torch.cuda.Event()),
            (torch.cuda.Event(), torch.cuda.Event()),
        )
        gate_slots = None
        gate_stream = None
        if pipeline_projected_gate:
            assert coarse_gate_buffers is not None
            assert project_coarse_gate_chunk is not None
            gate_stream = torch.cuda.Stream(device=query.device)
            gate_slots = _PingPongBuffers(
                coarse_gate_buffers,
                (torch.cuda.Event(), torch.cuda.Event()),
                attention_slots.produced,
            )
            gate_stream.wait_stream(producer)
            _enqueue_gate_chunk(
                gate_slots,
                gate_stream,
                project_coarse_gate_chunk,
                0,
                0,
                capacity,
            )

        for chunk_index, (block_start, block_count, start, rows) in enumerate(chunk_ranges):
            attention_chunk = attention_slots.acquire_for_write(producer, chunk_index, rows)
            if gate_slots is None:
                gate_chunk = coarse_gate_chunk(start, rows)
            else:
                gate_chunk = gate_slots.acquire_for_read(producer, chunk_index, rows)
            _quantized_dispatch._launch_quantized_sparse_piper_attention(
                prepared_attention,
                attention_chunk.transpose(1, 2),
                query_block_offset=block_start,
                query_block_count=block_count,
                coarse_output=prepared.coarse_output,
                coarse_gate=gate_chunk,
            )
            attention_slots.publish(producer, chunk_index)
            next_chunk_index = chunk_index + 1
            if gate_slots is not None and next_chunk_index < chunk_count:
                _, _, next_start, next_rows = chunk_ranges[next_chunk_index]
                assert gate_stream is not None
                assert project_coarse_gate_chunk is not None
                _enqueue_gate_chunk(
                    gate_slots,
                    gate_stream,
                    project_coarse_gate_chunk,
                    next_chunk_index,
                    next_start,
                    next_rows,
                )
            with torch.cuda.stream(consumer):
                ready_attention = attention_slots.acquire_for_read(
                    consumer,
                    chunk_index,
                    rows,
                )
                project_chunk(ready_attention, output, start, rows)
                attention_slots.release(consumer, chunk_index)
        producer.wait_event(attention_slots.consumed[(chunk_count - 1) % 2])
        attention_slots.buffers.record_stream(consumer)
        output.record_stream(consumer)
        for tensor in projector_tensors:
            tensor.record_stream(consumer)
        if gate_slots is not None:
            assert gate_stream is not None
            gate_slots.buffers.record_stream(gate_stream)
    return output


__all__ = [
    "DEFAULT_QUERY_CHUNK_ROWS",
    "new_projected_output",
    "output_sequence_length",
    "prepare_attention",
    "run_chunked_attention_output",
    "validate_attention_output",
]
