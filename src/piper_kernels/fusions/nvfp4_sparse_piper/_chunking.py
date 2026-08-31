"""Double-buffered prepared-NVFP4 row projection for fused consumers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from piper_kernels.linear.nvfp4._projection import matmul_prepared_chunk_out

DEFAULT_CHUNK_ROWS = 4096


@dataclass(frozen=True, slots=True)
class PreparedProjection:
    """Raw storage required to multiply chunks from one prepared activation."""

    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor


type ChunkConsumer = Callable[[torch.Tensor, int], None]


def run_chunked_projection(
    operands: PreparedProjection,
    chunk_rows: int,
    consume: ChunkConsumer,
    consumer_tensors: Sequence[torch.Tensor],
) -> None:
    """Overlap prepared FP4 GEMMs with a consumer using two reusable BF16 buffers."""
    rows = operands.input_qdata.shape[0]
    output_features = operands.weight_qdata.shape[0]
    chunk_count = (rows + chunk_rows - 1) // chunk_rows
    capacity = min(rows, chunk_rows)
    buffers = torch.empty(
        (min(2, chunk_count), capacity, output_features),
        device=operands.input_qdata.device,
        dtype=torch.bfloat16,
    )

    def project(slot: int, start: int, end: int) -> torch.Tensor:
        return matmul_prepared_chunk_out(
            operands.input_qdata,
            operands.input_scale,
            operands.weight_qdata,
            operands.weight_scale,
            start,
            end,
            buffers[slot],
        )

    if chunk_count == 1:
        consume(project(0, 0, rows), 0)
        return

    with torch.cuda.device(operands.input_qdata.device):
        producer = torch.cuda.current_stream(operands.input_qdata.device)
        consumer = torch.cuda.Stream(device=operands.input_qdata.device)
        ready = (torch.cuda.Event(), torch.cuda.Event())
        reusable = (torch.cuda.Event(), torch.cuda.Event())
        last_slot = 0
        for chunk_index, start in enumerate(range(0, rows, chunk_rows)):
            slot = chunk_index % 2
            last_slot = slot
            if chunk_index >= 2:
                producer.wait_event(reusable[slot])
            end = min(start + chunk_rows, rows)
            projection = project(slot, start, end)
            ready[slot].record(producer)
            with torch.cuda.stream(consumer):
                consumer.wait_event(ready[slot])
                consume(projection, start)
                reusable[slot].record(consumer)
        producer.wait_event(reusable[last_slot])
        buffers.record_stream(consumer)
        for tensor in consumer_tensors:
            tensor.record_stream(consumer)


__all__ = ["DEFAULT_CHUNK_ROWS", "PreparedProjection", "run_chunked_projection"]
