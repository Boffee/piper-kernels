"""Bounded attention-to-projection buffering and stream orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

type AttentionProjector = Callable[[torch.Tensor], torch.Tensor]
type ChunkProjector = Callable[[torch.Tensor, torch.Tensor, int, int], None]
type AuxiliaryChunkProjector = Callable[[torch.Tensor, int, int], None]
type AttentionChunkLauncher = Callable[[torch.Tensor, int, int, torch.Tensor | None], None]


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


def _enqueue_auxiliary_chunk(
    slots: _PingPongSlots,
    buffers: torch.Tensor,
    stream: torch.cuda.Stream,
    project_chunk: AuxiliaryChunkProjector,
    chunk_index: int,
    start: int,
    rows: int,
) -> None:
    """Project and publish one auxiliary chunk on its side stream."""
    with torch.cuda.stream(stream):
        slot = slots.acquire_for_write(stream, chunk_index)
        chunk = buffers[slot, :, :rows]
        project_chunk(chunk, start, rows)
        slots.publish(stream, chunk_index)


def run_chunked_attention_output(  # noqa: PLR0912, PLR0913, PLR0915
    attention_shape: tuple[int, int, int, int],
    device: torch.device,
    output_features: int,
    query_chunk_rows: int,
    launch_chunk: AttentionChunkLauncher,
    project_chunk: ChunkProjector | None,
    projector_tensors: Sequence[torch.Tensor],
    *,
    auxiliary_input: torch.Tensor | None = None,
    project_auxiliary_chunk: AuxiliaryChunkProjector | None = None,
    auxiliary_pipeline_min_chunks: int = 8,
    output_dtype: torch.dtype = torch.bfloat16,
    project_attention: AttentionProjector | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run validated row windows through attention and its output projection.

    Attention chunks use [batch, rows, heads, head_dim] storage. The caller owns
    attention semantics, supported dtypes, positive dimensions, and chunk alignment.
    Launchers receive global row coordinates and an optional auxiliary input in
    the same layout. Supply at most one auxiliary source. An auxiliary producer
    fills bounded buffers; its consumer must enqueue all reads on the attention
    stream before returning.

    Chunk projectors fill the final output rows. Multiple chunks overlap projection
    on a consumer stream; a single chunk uses the current stream. Include every
    tensor retained by the chunk projector in projector_tensors.

    A whole-output projector materializes attention first, for example to derive a
    global activation scale. It owns its output allocation; out applies only to
    chunk projection.
    """
    batch, sequence_length, heads, head_dim = attention_shape
    chunk_ranges = [
        (start, min(query_chunk_rows, sequence_length - start))
        for start in range(0, sequence_length, query_chunk_rows)
    ]
    chunk_count = len(chunk_ranges)
    pipeline_auxiliary = (
        project_attention is None
        and project_auxiliary_chunk is not None
        and chunk_count >= auxiliary_pipeline_min_chunks
    )
    capacity = min(sequence_length, query_chunk_rows)
    chunk_shape = (batch, capacity, heads, head_dim)
    auxiliary_buffers = (
        torch.empty(
            (2 if pipeline_auxiliary else 1, *chunk_shape),
            device=device,
            dtype=output_dtype,
        )
        if project_auxiliary_chunk is not None
        else None
    )

    def get_auxiliary_chunk(start: int, rows: int) -> torch.Tensor | None:
        if auxiliary_input is not None:
            return auxiliary_input[:, start : start + rows]
        if project_auxiliary_chunk is None:
            return None
        assert auxiliary_buffers is not None
        chunk = auxiliary_buffers[0, :, :rows]
        project_auxiliary_chunk(chunk, start, rows)
        return chunk

    if project_attention is not None:
        # A global activation scale requires every attention row before projection.
        attention = torch.empty(attention_shape, device=device, dtype=output_dtype)
        for start, rows in chunk_ranges:
            launch_chunk(
                attention[:, start : start + rows],
                start,
                rows,
                get_auxiliary_chunk(start, rows),
            )
        return project_attention(attention)

    assert project_chunk is not None
    attention_buffers = torch.empty(
        (min(2, chunk_count), *chunk_shape),
        device=device,
        dtype=output_dtype,
    )
    output_shape = (batch, sequence_length, output_features)
    if out is None:
        output = torch.empty(output_shape, device=device, dtype=output_dtype)
    else:
        if (
            out.shape != output_shape
            or out.device != device
            or out.dtype is not output_dtype
            or not out.is_contiguous()
        ):
            raise ValueError("attention output buffer must match the projected output")
        output = out

    if chunk_count == 1:
        start, rows = chunk_ranges[0]
        attention_chunk = attention_buffers[0, :, :rows]
        launch_chunk(attention_chunk, start, rows, get_auxiliary_chunk(start, rows))
        project_chunk(attention_chunk, output, start, rows)
        return output

    with torch.cuda.device(device):
        producer = torch.cuda.current_stream(device)
        consumer = torch.cuda.Stream(device=device)
        attention_slots = _PingPongSlots(
            produced=(torch.cuda.Event(), torch.cuda.Event()),
            consumed=(torch.cuda.Event(), torch.cuda.Event()),
        )
        auxiliary_slots = None
        auxiliary_stream = None
        if pipeline_auxiliary:
            assert auxiliary_buffers is not None
            assert project_auxiliary_chunk is not None
            auxiliary_stream = torch.cuda.Stream(device=device)
            auxiliary_slots = _PingPongSlots(
                produced=(torch.cuda.Event(), torch.cuda.Event()),
                # Attention completion releases its auxiliary input slot.
                consumed=attention_slots.produced,
            )
            auxiliary_stream.wait_stream(producer)
            _enqueue_auxiliary_chunk(
                auxiliary_slots,
                auxiliary_buffers,
                auxiliary_stream,
                project_auxiliary_chunk,
                0,
                0,
                capacity,
            )

        for chunk_index, (start, rows) in enumerate(chunk_ranges):
            attention_slot = attention_slots.acquire_for_write(producer, chunk_index)
            attention_chunk = attention_buffers[attention_slot, :, :rows]
            if auxiliary_slots is None:
                auxiliary_chunk = get_auxiliary_chunk(start, rows)
            else:
                auxiliary_slot = auxiliary_slots.acquire_for_read(producer, chunk_index)
                assert auxiliary_buffers is not None
                auxiliary_chunk = auxiliary_buffers[auxiliary_slot, :, :rows]
            launch_chunk(attention_chunk, start, rows, auxiliary_chunk)
            attention_slots.publish(producer, chunk_index)
            next_chunk_index = chunk_index + 1
            if auxiliary_slots is not None and next_chunk_index < chunk_count:
                next_start, next_rows = chunk_ranges[next_chunk_index]
                assert auxiliary_stream is not None
                assert auxiliary_buffers is not None
                assert project_auxiliary_chunk is not None
                _enqueue_auxiliary_chunk(
                    auxiliary_slots,
                    auxiliary_buffers,
                    auxiliary_stream,
                    project_auxiliary_chunk,
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
        if auxiliary_slots is not None:
            assert auxiliary_stream is not None
            assert auxiliary_buffers is not None
            auxiliary_buffers.record_stream(auxiliary_stream)
    return output
