"""Shared bounded peer-exchange reduce-scatter helpers for distributed fusions."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from piper_kernels._residual import ResidualAdd, validate_residual_add
from piper_kernels._residual.triton import apply_residual_add

from . import _comm as distributed_comm

type PeerExchangeHandle = tuple[dist.Work, int, int, torch.Tensor, tuple[torch.Tensor, ...]]
type ApplyReceived = Callable[[torch.Tensor, int, int], None]

_COMM_WORKSPACE_LIMIT_BYTES = 256 * 1024 * 1024


def contiguous_chunk_view(
    buffer: torch.Tensor,
    batch: int,
    rows: int,
    features: int,
) -> torch.Tensor:
    """Slice a prefix of ``buffer`` as ``[batch, rows, features]``."""
    return buffer.view(-1)[: batch * rows * features].view(batch, rows, features)


def validate_distributed_sequence(
    sequence_length: int,
    comm_group_size: int,
    comm_group_name: str,
    *,
    name: str = "distributed Piper",
) -> int:
    """Validate process-group metadata and return one rank's local sequence length."""
    if (
        isinstance(comm_group_size, bool)
        or not isinstance(comm_group_size, int)
        or comm_group_size < 2
    ):
        raise ValueError(f"{name} attention requires comm_group_size >= 2")
    if not isinstance(comm_group_name, str):
        raise TypeError(f"{name} attention requires string comm_group_name")
    if not isinstance(sequence_length, torch.SymInt) and sequence_length % comm_group_size:
        raise ValueError(f"{name} sequence length must divide evenly across ranks")
    return sequence_length // comm_group_size


def resolve_validated_process_group(
    comm_group_name: str,
    comm_group_size: int,
    *,
    name: str = "distributed Piper",
) -> dist.ProcessGroup:
    """Resolve ``comm_group_name`` and check that its world size matches."""
    process_group = distributed_comm.resolve_process_group(comm_group_name)
    if process_group.size() != comm_group_size:
        raise ValueError(f"{name} comm_group_size must match resolved group")
    return process_group


def validate_distributed_residual_add(
    residual_add: ResidualAdd,
    *,
    batch: int,
    local_sequence_length: int,
    output_features: int,
    storage_device: torch.device,
    name: str = "distributed Piper",
    storage_label: str = "input",
) -> None:
    """Validate that a local residual base matches one rank's share and device."""
    base = residual_add.base
    if base.ndim != 3 or base.shape[:2] != (batch, local_sequence_length):
        raise ValueError(f"{name} residual base must have one local rank share of rows")
    if base.device != storage_device:
        raise ValueError(f"{name} residual base must share the {storage_label} storage device")
    validate_residual_add(base, residual_add, output_features)


def choose_distributed_chunk_rows(
    local_sequence_length: int,
    query_chunk_rows: int,
    *,
    tile_rows: int,
    batch: int,
    output_features: int,
    element_size: int,
    world_size: int,
    reuse_direct_receive: bool = False,
    max_comm_bytes: int = _COMM_WORKSPACE_LIMIT_BYTES,
) -> int:
    """Choose a balanced, tile-aligned peer exchange window bounded by comm VRAM."""
    if local_sequence_length < tile_rows:
        return local_sequence_length
    peer_count = max(1, world_size - 1)
    fixed_buffers = (0 if reuse_direct_receive else 1) + (0 if world_size <= 2 else 1)
    row_bytes = max(1, batch * output_features * element_size)
    single_buffers = peer_count + fixed_buffers
    if (
        local_sequence_length <= query_chunk_rows
        and local_sequence_length % tile_rows == 0
        and local_sequence_length * row_bytes * single_buffers <= max_comm_bytes
    ):
        return local_sequence_length
    multi_buffers = 2 * peer_count + fixed_buffers
    budget_rows = max_comm_bytes // (row_bytes * multi_buffers)
    cap = min(local_sequence_length, query_chunk_rows, budget_rows)
    cap = max(tile_rows, (cap // tile_rows) * tile_rows)
    chunks = (local_sequence_length + cap - 1) // cap
    rows = (local_sequence_length + chunks - 1) // chunks
    aligned = ((rows + tile_rows - 1) // tile_rows) * tile_rows
    return min(cap, max(tile_rows, aligned))


def apply_received_chunk(
    receive_tensor: torch.Tensor,
    output: torch.Tensor,
    local_offset: int,
    rows: int,
    *,
    residual_add: ResidualAdd | None,
    reuse_direct_receive: bool,
) -> None:
    """Copy or residual-add a peer-received chunk into ``output``."""
    batch, local_sequence_length = output.shape[:2]
    if residual_add is not None:
        for batch_index in range(batch):
            apply_residual_add(
                receive_tensor[batch_index],
                residual_add,
                row_offset=batch_index * local_sequence_length + local_offset,
            )
    elif not reuse_direct_receive:
        output[:, local_offset : local_offset + rows].copy_(receive_tensor)


def run_distributed_fallback(
    process_group: dist.ProcessGroup,
    comm_group_size: int,
    local_sequence_length: int,
    output_features: int,
    output_dtype: torch.dtype,
    full_output: torch.Tensor,
    residual_add: ResidualAdd | None,
) -> torch.Tensor:
    """Reduce-scatter a full-sequence output when aligned chunking cannot run."""
    batch = full_output.shape[0]
    reduced = torch.empty(
        (batch, local_sequence_length, output_features),
        device=full_output.device,
        dtype=output_dtype,
    )
    reduce_scatter_input = (
        full_output.view(comm_group_size, 1, local_sequence_length, output_features)
        if batch == 1
        else torch.cat(list(full_output.split(local_sequence_length, dim=1)), dim=0).contiguous()
    )
    work = dist.reduce_scatter_tensor(
        reduced,
        reduce_scatter_input,
        op=dist.ReduceOp.SUM,
        group=process_group,
        async_op=False,
    )
    if work is not None:
        work.wait()
    if residual_add is not None:
        apply_residual_add(reduced, residual_add, row_offset=0)
        return residual_add.base
    return reduced


def allocate_peer_comm_slots(
    world_size: int,
    peer_ranks: tuple[int, ...],
    batch: int,
    capacity: int,
    output_features: int,
    *,
    slots: int,
    device: torch.device,
    output_dtype: torch.dtype,
    reuse_direct_receive: bool,
) -> tuple[list[list[torch.Tensor]], torch.Tensor | None, torch.Tensor | None]:
    """Allocate double-buffered send slots and a single reused receive scratch buffer."""
    send_slots: list[list[torch.Tensor]] = [
        [
            torch.empty((batch, capacity, output_features), device=device, dtype=output_dtype)
            for _ in peer_ranks
        ]
        for _ in range(slots)
    ]
    # Safe to reuse one receive buffer across chunks: each loop waits on the previous
    # collective and consumes its received chunk on the current stream before issuing the next.
    receive_buffer = (
        None
        if reuse_direct_receive
        else torch.empty((batch, capacity, output_features), device=device, dtype=output_dtype)
    )
    zero_local_buffer = (
        None
        if world_size == 2
        else torch.zeros((batch, capacity, output_features), device=device, dtype=output_dtype)
    )
    return send_slots, receive_buffer, zero_local_buffer


def run_peer_chunk_exchange(
    process_group: dist.ProcessGroup,
    output: torch.Tensor,
    comm_chunk_rows: int,
    run_peer_window: Callable[[int, int, torch.Tensor], None],
    apply_received: ApplyReceived,
    *,
    reuse_direct_receive: bool,
) -> PeerExchangeHandle:
    """Compute peer windows and ping-pong async exchanges, returning the final in-flight chunk."""
    world_size = process_group.size()
    batch, local_sequence_length, output_features = output.shape
    capacity = min(local_sequence_length, comm_chunk_rows)
    sub_ranges = [
        (offset, min(comm_chunk_rows, local_sequence_length - offset))
        for offset in range(0, local_sequence_length, comm_chunk_rows)
    ]
    slots = min(2, len(sub_ranges))
    peer_ranks = tuple(peer for peer in range(world_size) if peer != process_group.rank())

    send_slots, receive_buffer, zero_local_buffer = allocate_peer_comm_slots(
        world_size,
        peer_ranks,
        batch,
        capacity,
        output_features,
        slots=slots,
        device=output.device,
        output_dtype=output.dtype,
        reuse_direct_receive=reuse_direct_receive,
    )

    pending: tuple[dist.Work, int, int, int, torch.Tensor] | None = None
    try:
        for chunk_index, (local_offset, rows) in enumerate(sub_ranges):
            slot = chunk_index % slots
            send_views = [
                contiguous_chunk_view(send_slots[slot][peer_index], batch, rows, output_features)
                for peer_index in range(len(peer_ranks))
            ]
            for peer_rank, send_view in zip(peer_ranks, send_views, strict=True):
                run_peer_window(peer_rank * local_sequence_length + local_offset, rows, send_view)

            if pending is not None:
                prev_work, _prev_slot, prev_offset, prev_rows, prev_receive = pending
                prev_work.wait()
                apply_received(prev_receive, prev_offset, prev_rows)
                pending = None

            if reuse_direct_receive:
                receive_view = output[:, local_offset : local_offset + rows]
            else:
                assert receive_buffer is not None
                receive_view = contiguous_chunk_view(receive_buffer, batch, rows, output_features)
            zero_view = (
                None
                if zero_local_buffer is None
                else contiguous_chunk_view(zero_local_buffer, batch, rows, output_features)
            )
            work = distributed_comm.start_peer_exchange(
                process_group, send_views, receive_view, zero_view
            )
            pending = (work, slot, local_offset, rows, receive_view)
    except BaseException:
        if pending is not None:
            pending[0].wait()
        raise

    assert pending is not None
    last_work, last_slot, last_offset, last_rows, last_receive = pending
    keepalive: tuple[torch.Tensor, ...] = (
        *send_slots[last_slot],
        *(() if zero_local_buffer is None else (zero_local_buffer,)),
    )
    return last_work, last_offset, last_rows, last_receive, keepalive


def run_peer_attention_exchange(
    process_group: dist.ProcessGroup,
    output: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    comm_chunk_rows: int,
    run_attention: Callable[[int, int, torch.Tensor], None],
    peer_projector: Callable[[torch.Tensor, torch.Tensor, int, int], None],
    apply_received: ApplyReceived,
    reuse_direct_receive: bool,
) -> PeerExchangeHandle:
    """Run attention + output projection for each peer window and ping-pong exchange."""
    batch, local_sequence_length, output_features = output.shape
    capacity = min(local_sequence_length, comm_chunk_rows)
    attention_scratch: torch.Tensor | None = (
        None
        if output_features >= heads * head_dim
        else torch.empty(
            (batch, capacity, heads, head_dim),
            device=output.device,
            dtype=output.dtype,
        )
    )

    def run_peer_window(global_start: int, rows: int, output_buffer: torch.Tensor) -> None:
        # Pack attention contiguously within each batch's own output slice so
        # per-batch projection never overwrites later batches before reading them.
        attention_chunk = (
            output_buffer.view(batch, rows * output_features)[:, : rows * heads * head_dim].view(
                batch, rows, heads, head_dim
            )
            if attention_scratch is None
            else attention_scratch[:, :rows]
        )
        run_attention(global_start, rows, attention_chunk)
        peer_projector(attention_chunk, output_buffer, 0, rows)

    return run_peer_chunk_exchange(
        process_group,
        output,
        comm_chunk_rows,
        run_peer_window,
        apply_received,
        reuse_direct_receive=reuse_direct_receive,
    )


def finish_overlapped_exchange(
    last_exchange: PeerExchangeHandle,
    *,
    run_own: Callable[[], torch.Tensor | None],
    apply_received: ApplyReceived,
    output: torch.Tensor,
    residual_add: ResidualAdd | None,
) -> torch.Tensor:
    """Run own-rank work behind the last in-flight collective, wait, and combine."""
    last_work, last_offset, last_rows, last_receive, keepalive = last_exchange
    del last_exchange
    try:
        own_partial = run_own()
    except BaseException:
        last_work.wait()
        del keepalive, last_receive
        raise
    last_work.wait()
    del keepalive
    apply_received(last_receive, last_offset, last_rows)
    if residual_add is not None:
        return residual_add.base
    assert own_partial is not None
    output.add_(own_partial)
    return output


def run_overlapped_distributed_output(
    process_group: dist.ProcessGroup,
    *,
    output_shape: tuple[int, int, int],
    output_dtype: torch.dtype,
    device: torch.device,
    query_chunk_rows: int,
    tile_rows: int,
    residual_add: ResidualAdd | None,
    run_fallback: Callable[[], torch.Tensor],
    start_aligned: Callable[
        [int, torch.Tensor, ApplyReceived, bool, int],
        tuple[PeerExchangeHandle, Callable[[], torch.Tensor | None]],
    ],
) -> torch.Tensor:
    """Run aligned overlapped peer exchange + own-rank output, or reduce-scatter fallback."""
    batch, local_sequence_length, output_features = output_shape
    if local_sequence_length < tile_rows or local_sequence_length % tile_rows:
        return run_distributed_fallback(
            process_group,
            process_group.size(),
            local_sequence_length,
            output_features,
            output_dtype,
            run_fallback(),
            residual_add,
        )

    reuse_direct_receive = residual_add is None and batch == 1
    comm_chunk_rows = choose_distributed_chunk_rows(
        local_sequence_length,
        query_chunk_rows,
        tile_rows=tile_rows,
        batch=batch,
        output_features=output_features,
        element_size=output_dtype.itemsize,
        world_size=process_group.size(),
        reuse_direct_receive=reuse_direct_receive,
    )
    output = (
        residual_add.base
        if residual_add is not None
        else torch.empty(output_shape, device=device, dtype=output_dtype)
    )

    def apply_received(receive_tensor: torch.Tensor, local_offset: int, rows: int) -> None:
        apply_received_chunk(
            receive_tensor,
            output,
            local_offset,
            rows,
            residual_add=residual_add,
            reuse_direct_receive=reuse_direct_receive,
        )

    own_base = process_group.rank() * local_sequence_length
    last_exchange, run_own = start_aligned(
        comm_chunk_rows,
        output,
        apply_received,
        reuse_direct_receive,
        own_base,
    )
    try:
        return finish_overlapped_exchange(
            last_exchange,
            run_own=run_own,
            apply_received=apply_received,
            output=output,
            residual_add=residual_add,
        )
    finally:
        del last_exchange


__all__ = [
    "ApplyReceived",
    "PeerExchangeHandle",
    "allocate_peer_comm_slots",
    "apply_received_chunk",
    "choose_distributed_chunk_rows",
    "contiguous_chunk_view",
    "finish_overlapped_exchange",
    "resolve_validated_process_group",
    "run_distributed_fallback",
    "run_overlapped_distributed_output",
    "run_peer_attention_exchange",
    "run_peer_chunk_exchange",
    "validate_distributed_residual_add",
    "validate_distributed_sequence",
]
