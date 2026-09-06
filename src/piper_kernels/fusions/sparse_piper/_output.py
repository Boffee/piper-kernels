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
    from piper_kernels.attention.sparse_piper_attention._prepared import (
        _PreparedSparsePiperAttention,
    )

DEFAULT_QUERY_CHUNK_ROWS = 4_096
_MIN_PROJECTED_GATE_PIPELINE_CHUNKS = 8

type ChunkProjector = Callable[[torch.Tensor, torch.Tensor, int, int], None]
type CoarseGateChunkProjector = Callable[[torch.Tensor, int, int], None]
type QueryChunkProjector = Callable[[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
type AttentionChunkLauncher = Callable[
    [torch.Tensor, int, int, int, int, torch.Tensor | None], None
]

output_sequence_length = _quantized_dispatch._quantized_attention_output_sequence_length


@dataclass(frozen=True, slots=True)
class _PreparedAttentionOutput:
    """Shared launch state for one bounded attention-to-projection pipeline."""

    attention: _PreparedSparsePiperAttention
    sequence_length: int
    coarse_output: torch.Tensor | None
    coarse_gate: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _PreparedAttentionContext:
    """Global launch state for query chunks that have not been projected yet."""

    quantized_context: _quantized_dispatch._PreparedQuantizedSparsePiperContext
    sequence_length: int
    coarse_gate: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _PingPongSlots:
    """Synchronize two reusable caller-owned chunk slots."""

    produced: tuple[torch.cuda.Event, torch.cuda.Event]
    consumed: tuple[torch.cuda.Event, torch.cuda.Event]

    def acquire_for_write(
        self,
        stream: torch.cuda.Stream,
        chunk_index: int,
    ) -> int:
        """Wait until a slot's previous reader has released it."""
        slot = chunk_index % 2
        if chunk_index >= 2:
            stream.wait_event(self.consumed[slot])
        return slot

    def publish(self, stream: torch.cuda.Stream, chunk_index: int) -> None:
        """Publish one completed write to readers."""
        self.produced[chunk_index % 2].record(stream)

    def acquire_for_read(
        self,
        stream: torch.cuda.Stream,
        chunk_index: int,
    ) -> int:
        """Wait for a slot's current contents and return its index."""
        slot = chunk_index % 2
        stream.wait_event(self.produced[slot])
        return slot

    def release(self, stream: torch.cuda.Stream, chunk_index: int) -> None:
        """Mark one consumed slot reusable by its writer."""
        self.consumed[chunk_index % 2].record(stream)


def _enqueue_gate_chunk(
    slots: _PingPongSlots,
    buffers: torch.Tensor,
    stream: torch.cuda.Stream,
    project_chunk: CoarseGateChunkProjector,
    chunk_index: int,
    start: int,
    rows: int,
) -> None:
    """Project and publish one gate chunk on its side stream."""
    with torch.cuda.stream(stream):
        slot = slots.acquire_for_write(stream, chunk_index)
        chunk = buffers[slot, :, :rows]
        project_chunk(chunk, start, rows)
        slots.publish(stream, chunk_index)


def new_projected_output(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    output_features: int,
) -> torch.Tensor:
    """Allocate the projected compact or valid-front padded output shape."""
    return attention_storage.new_empty(
        (
            attention_storage.shape[0],
            output_sequence_length(
                attention_storage,
                logical_sequence_length,
                block_lengths,
            ),
            output_features,
        ),
        dtype=torch.bfloat16,
    )


def validate_attention_output(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    query_chunk_rows: int,
) -> int:
    """Validate the common boundary and return the flattened head width."""
    if attention_storage.ndim != 4:
        raise ValueError("fused sparse Piper output requires four-dimensional quantized storage")
    batch, heads, _storage_sequence_length, head_dim = attention_storage.shape
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
    if attention_storage.device.type != "cuda":
        raise ValueError("fused sparse Piper output currently requires CUDA")
    target = AcceleratorTarget.from_device(attention_storage.device)
    if not target.is_cuda_capability(12, 0):
        raise ValueError("fused sparse Piper output requires exact NVIDIA SM120")
    return heads * head_dim


def prepare_attention_context(  # noqa: PLR0913, PLR0917
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
) -> _PreparedAttentionContext:
    """Prepare global K/V state for bounded, independently projected Q chunks."""
    if coarse_gate is not None and has_projected_coarse_gate:
        raise ValueError("coarse gate cannot be both materialized and projected")
    has_coarse_gate = coarse_gate is not None or has_projected_coarse_gate
    if (block_mean is not None) != has_coarse_gate:
        raise ValueError("block means and coarse gate must be supplied together")
    quantized_context = _quantized_dispatch._prepare_quantized_sparse_piper_context(
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
        block_mean,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
    )
    return _PreparedAttentionContext(
        quantized_context=quantized_context,
        sequence_length=output_sequence_length(
            key,
            logical_sequence_length,
            block_lengths,
        ),
        coarse_gate=coarse_gate,
    )


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
    context = prepare_attention_context(
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
        block_mean,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        has_projected_coarse_gate=has_projected_coarse_gate,
    )
    prepared, coarse_output = _quantized_dispatch._prepare_quantized_sparse_piper_query(
        context.quantized_context,
        query,
        query_scale,
        query_summary,
        global_block_offset=0,
    )
    return _PreparedAttentionOutput(
        attention=prepared,
        sequence_length=context.sequence_length,
        coarse_output=coarse_output,
        coarse_gate=coarse_gate,
    )


def _query_chunk_ranges(
    sequence_length: int,
    query_chunk_rows: int,
) -> list[tuple[int, int, int, int]]:
    """Return global block and row coordinates for each bounded Q window."""
    chunk_blocks = query_chunk_rows // TILE_ROWS
    total_blocks = (sequence_length + TILE_ROWS - 1) // TILE_ROWS
    ranges = []
    for block_start in range(0, total_blocks, chunk_blocks):
        block_count = min(chunk_blocks, total_blocks - block_start)
        start = block_start * TILE_ROWS
        rows = min(block_count * TILE_ROWS, sequence_length - start)
        ranges.append((block_start, block_count, start, rows))
    return ranges


def _run_chunked_attention_pipeline(  # noqa: PLR0913, PLR0915
    attention_storage: torch.Tensor,
    sequence_length: int,
    has_coarse_residual: bool,
    coarse_gate: torch.Tensor | None,
    output_features: int,
    query_chunk_rows: int,
    launch_chunk: AttentionChunkLauncher,
    project_chunk: ChunkProjector,
    projector_tensors: Sequence[torch.Tensor],
    *,
    project_coarse_gate_chunk: CoarseGateChunkProjector | None = None,
    share_projection_stream: bool = False,
) -> torch.Tensor:
    """Share buffering, gate, and stream ordering across both Q lifetimes."""
    chunk_ranges = _query_chunk_ranges(sequence_length, query_chunk_rows)
    chunk_count = len(chunk_ranges)
    pipeline_projected_gate = (
        project_coarse_gate_chunk is not None
        and chunk_count > 1
        and (share_projection_stream or chunk_count >= _MIN_PROJECTED_GATE_PIPELINE_CHUNKS)
    )
    capacity = min(sequence_length, query_chunk_rows)
    batch, heads, head_dim = (
        attention_storage.shape[0],
        attention_storage.shape[1],
        attention_storage.shape[3],
    )
    attention_buffers = torch.empty(
        (min(2, chunk_count), batch, capacity, heads, head_dim),
        device=attention_storage.device,
        dtype=torch.bfloat16,
    )
    if has_coarse_residual and (coarse_gate is None) == (project_coarse_gate_chunk is None):
        raise ValueError("coarse attention requires exactly one coarse gate source")
    if not has_coarse_residual and (
        coarse_gate is not None or project_coarse_gate_chunk is not None
    ):
        raise ValueError("coarse gate source requires coarse attention")
    coarse_gate_buffers = (
        torch.empty(
            (2 if pipeline_projected_gate else 1, batch, capacity, heads, head_dim),
            device=attention_storage.device,
            dtype=torch.bfloat16,
        )
        if project_coarse_gate_chunk is not None
        else None
    )
    output = torch.empty(
        (batch, sequence_length, output_features),
        device=attention_storage.device,
        dtype=torch.bfloat16,
    )

    def coarse_gate_chunk(start: int, rows: int) -> torch.Tensor | None:
        if coarse_gate is not None:
            return coarse_gate[:, start : start + rows]
        if project_coarse_gate_chunk is None:
            return None
        assert coarse_gate_buffers is not None
        chunk = coarse_gate_buffers[0, :, :rows]
        project_coarse_gate_chunk(chunk, start, rows)
        return chunk

    if chunk_count == 1:
        block_start, block_count, start, rows = chunk_ranges[0]
        attention_chunk = attention_buffers[0, :, :rows]
        launch_chunk(
            attention_chunk,
            block_start,
            block_count,
            start,
            rows,
            coarse_gate_chunk(start, rows),
        )
        project_chunk(attention_chunk, output, start, rows)
        return output

    with torch.cuda.device(attention_storage.device):
        producer = torch.cuda.current_stream(attention_storage.device)
        consumer = torch.cuda.Stream(device=attention_storage.device)
        attention_slots = _PingPongSlots(
            (torch.cuda.Event(), torch.cuda.Event()),
            (torch.cuda.Event(), torch.cuda.Event()),
        )
        gate_slots = None
        gate_stream = None
        if pipeline_projected_gate:
            assert coarse_gate_buffers is not None
            assert project_coarse_gate_chunk is not None
            gate_stream = (
                consumer
                if share_projection_stream
                else torch.cuda.Stream(device=attention_storage.device)
            )
            gate_slots = _PingPongSlots(
                (torch.cuda.Event(), torch.cuda.Event()),
                attention_slots.produced,
            )
            gate_stream.wait_stream(producer)
            _enqueue_gate_chunk(
                gate_slots,
                coarse_gate_buffers,
                gate_stream,
                project_coarse_gate_chunk,
                0,
                0,
                capacity,
            )

        for chunk_index, (block_start, block_count, start, rows) in enumerate(chunk_ranges):
            attention_slot = attention_slots.acquire_for_write(producer, chunk_index)
            attention_chunk = attention_buffers[attention_slot, :, :rows]
            if gate_slots is None:
                gate_chunk = coarse_gate_chunk(start, rows)
            else:
                gate_slot = gate_slots.acquire_for_read(producer, chunk_index)
                assert coarse_gate_buffers is not None
                gate_chunk = coarse_gate_buffers[gate_slot, :, :rows]
            launch_chunk(
                attention_chunk,
                block_start,
                block_count,
                start,
                rows,
                gate_chunk,
            )
            attention_slots.publish(producer, chunk_index)
            next_chunk_index = chunk_index + 1
            if gate_slots is not None and next_chunk_index < chunk_count:
                _, _, next_start, next_rows = chunk_ranges[next_chunk_index]
                assert gate_stream is not None
                assert coarse_gate_buffers is not None
                assert project_coarse_gate_chunk is not None
                _enqueue_gate_chunk(
                    gate_slots,
                    coarse_gate_buffers,
                    gate_stream,
                    project_coarse_gate_chunk,
                    next_chunk_index,
                    next_start,
                    next_rows,
                )
            with torch.cuda.stream(consumer):
                ready_slot = attention_slots.acquire_for_read(consumer, chunk_index)
                ready_attention = attention_buffers[ready_slot, :, :rows]
                project_chunk(ready_attention, output, start, rows)
                attention_slots.release(consumer, chunk_index)
        producer.wait_event(attention_slots.consumed[(chunk_count - 1) % 2])
        attention_buffers.record_stream(consumer)
        output.record_stream(consumer)
        for tensor in projector_tensors:
            tensor.record_stream(consumer)
        if gate_slots is not None:
            assert gate_stream is not None
            assert coarse_gate_buffers is not None
            coarse_gate_buffers.record_stream(gate_stream)
    return output


def run_chunked_attention_output(
    prepared: _PreparedAttentionOutput,
    output_features: int,
    query_chunk_rows: int,
    project_chunk: ChunkProjector,
    projector_tensors: Sequence[torch.Tensor],
    *,
    project_coarse_gate_chunk: CoarseGateChunkProjector | None = None,
    share_projection_stream: bool = False,
) -> torch.Tensor:
    """Pipeline a materialized Q boundary through bounded attention output."""
    prepared_attention = prepared.attention

    def launch_chunk(
        attention_chunk: torch.Tensor,
        block_start: int,
        block_count: int,
        _start: int,
        _rows: int,
        gate_chunk: torch.Tensor | None,
    ) -> None:
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            prepared_attention,
            attention_chunk.transpose(1, 2),
            query_block_offset=block_start,
            query_block_count=block_count,
            coarse_output=prepared.coarse_output,
            coarse_gate=gate_chunk,
        )

    return _run_chunked_attention_pipeline(
        prepared_attention.query.data,
        prepared.sequence_length,
        prepared.coarse_output is not None,
        prepared.coarse_gate,
        output_features,
        query_chunk_rows,
        launch_chunk,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=project_coarse_gate_chunk,
        share_projection_stream=share_projection_stream,
    )


def run_chunked_projected_query_attention_output(
    prepared: _PreparedAttentionContext,
    output_features: int,
    query_chunk_rows: int,
    project_query_chunk: QueryChunkProjector,
    project_chunk: ChunkProjector,
    projector_tensors: Sequence[torch.Tensor],
    *,
    project_coarse_gate_chunk: CoarseGateChunkProjector | None = None,
    share_projection_stream: bool = False,
) -> torch.Tensor:
    """Project, route, attend, and consume one bounded Q window at a time."""

    def launch_chunk(
        attention_chunk: torch.Tensor,
        block_start: int,
        _block_count: int,
        start: int,
        rows: int,
        gate_chunk: torch.Tensor | None,
    ) -> None:
        query, query_scale, query_summary = project_query_chunk(start, rows)
        local_attention, coarse_output = _quantized_dispatch._prepare_quantized_sparse_piper_query(
            prepared.quantized_context,
            query,
            query_scale,
            query_summary,
            global_block_offset=block_start,
        )
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            local_attention,
            attention_chunk.transpose(1, 2),
            coarse_output=coarse_output,
            coarse_gate=gate_chunk,
        )

    return _run_chunked_attention_pipeline(
        prepared.quantized_context.kernel_context.key,
        prepared.sequence_length,
        prepared.quantized_context.pooled_value is not None,
        prepared.coarse_gate,
        output_features,
        query_chunk_rows,
        launch_chunk,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=project_coarse_gate_chunk,
        share_projection_stream=share_projection_stream,
    )


__all__ = [
    "DEFAULT_QUERY_CHUNK_ROWS",
    "new_projected_output",
    "output_sequence_length",
    "prepare_attention",
    "prepare_attention_context",
    "run_chunked_attention_output",
    "run_chunked_projected_query_attention_output",
    "validate_attention_output",
]
