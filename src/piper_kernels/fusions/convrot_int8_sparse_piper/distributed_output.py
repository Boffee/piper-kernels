"""Bounded peer-exchange reduce-scatter for distributed fused sparse Piper attention.

Overlaps bounded peer attention + output projection chunks and collective exchange
behind own-rank local sequence computation. Unlike eager reduce-scatter followed by
``residual_add_``, overlapped peer-chunk residual updates add peer and local rank
contributions into ``base`` in two separately rounded steps to keep peer scratch
memory bounded.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from piper_kernels._distributed import _comm as distributed_comm
from piper_kernels._residual import ResidualAdd, validate_residual_add
from piper_kernels._residual.triton import apply_residual_add
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile
from piper_kernels.fusions.convrot_int8_projection import output as projection_output
from piper_kernels.fusions.sparse_piper import _output as output_common
from piper_kernels.weights.convrot.int8._quantization import validate_activation_scale

from . import _backend as fusion_backend
from . import output as output_ops
from ._layout import TILE_ROWS

_DEFAULT_QUERY_CHUNK_ROWS = output_ops.DEFAULT_QUERY_CHUNK_ROWS


def _contiguous_chunk_view(
    buffer: torch.Tensor,
    batch: int,
    rows: int,
    features: int,
) -> torch.Tensor:
    return buffer.view(-1)[: batch * rows * features].view(batch, rows, features)


def _choose_distributed_chunk_rows(local_sequence_length: int, query_chunk_rows: int) -> int:
    if local_sequence_length < TILE_ROWS:
        return local_sequence_length
    target = min(local_sequence_length, query_chunk_rows, 4096)
    return max(TILE_ROWS, (target // TILE_ROWS) * TILE_ROWS)


def _validate_distributed_sequence(
    sequence_length: int,
    comm_group_size: int,
    comm_group_name: str,
) -> int:
    if (
        isinstance(comm_group_size, bool)
        or not isinstance(comm_group_size, int)
        or comm_group_size < 2
    ):
        raise ValueError("distributed sparse Piper attention requires comm_group_size >= 2")
    if not isinstance(comm_group_name, str):
        raise TypeError("distributed sparse Piper attention requires string comm_group_name")
    if not isinstance(sequence_length, torch.SymInt) and sequence_length % comm_group_size:
        raise ValueError("distributed sparse Piper sequence length must divide evenly across ranks")
    return sequence_length // comm_group_size


def _run_distributed_fallback(
    process_group: dist.ProcessGroup,
    comm_group_size: int,
    local_sequence_length: int,
    output_features: int,
    output_dtype: torch.dtype,
    full_output: torch.Tensor,
    residual_add: ResidualAdd | None,
) -> torch.Tensor:
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


def _allocate_peer_comm_slots(
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
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor | None], torch.Tensor | None]:
    send_slots: list[list[torch.Tensor]] = [
        [
            torch.empty((batch, capacity, output_features), device=device, dtype=output_dtype)
            for _ in peer_ranks
        ]
        for _ in range(slots)
    ]
    receive_slots: list[torch.Tensor | None] = [
        None
        if reuse_direct_receive
        else torch.empty((batch, capacity, output_features), device=device, dtype=output_dtype)
        for _ in range(slots)
    ]
    zero_local_buffer = (
        None
        if world_size == 2
        else torch.zeros((batch, capacity, output_features), device=device, dtype=output_dtype)
    )
    return send_slots, receive_slots, zero_local_buffer


def _run_peer_attention_exchange(
    process_group: dist.ProcessGroup,
    prepared: output_common._PreparedAttentionContext,
    key_dims: tuple[int, int, int],
    output: torch.Tensor,
    comm_chunk_rows: int,
    project_query_chunk: output_common.QueryChunkProjector,
    gate_projection: output_ops.GateProjection | None,
    apply_received: Callable[[torch.Tensor, int, int], None],
    peer_projector: output_common.ChunkProjector,
    *,
    reuse_direct_receive: bool,
) -> tuple[dist.Work, int, int, torch.Tensor, tuple[torch.Tensor, ...]]:
    world_size = process_group.size()
    batch, local_sequence_length, output_features = output.shape
    heads, head_dim = key_dims[1], key_dims[2]
    capacity = min(local_sequence_length, comm_chunk_rows)
    sub_ranges = [
        (offset, min(comm_chunk_rows, local_sequence_length - offset))
        for offset in range(0, local_sequence_length, comm_chunk_rows)
    ]
    slots = min(2, len(sub_ranges))
    peer_ranks = tuple(peer for peer in range(world_size) if peer != process_group.rank())
    attention_scratch: torch.Tensor | None = (
        None
        if output_features >= heads * head_dim
        else torch.empty(
            (batch, capacity, heads, head_dim), device=output.device, dtype=output.dtype
        )
    )
    gate_scratch: torch.Tensor | None = (
        torch.empty((batch, capacity, heads, head_dim), device=output.device, dtype=output.dtype)
        if gate_projection is not None
        else None
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
        gate_chunk: torch.Tensor | None = None
        if gate_projection is not None:
            assert gate_scratch is not None
            gate_chunk = gate_scratch[:, :rows]
            gate_projection.project(gate_chunk, global_start, rows)
        elif prepared.coarse_gate is not None:
            gate_chunk = prepared.coarse_gate[:, global_start : global_start + rows]
        query, query_scale, query_summary = project_query_chunk(global_start, rows)
        local_attention, coarse_output = _quantized_dispatch._prepare_quantized_sparse_piper_query(
            prepared.quantized_context,
            query,
            query_scale,
            query_summary,
            global_block_offset=global_start // TILE_ROWS,
        )
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            local_attention,
            attention_chunk.transpose(1, 2),
            coarse_output=coarse_output,
            coarse_gate=gate_chunk,
        )
        peer_projector(attention_chunk, output_buffer, 0, rows)

    send_slots, receive_slots, zero_local_buffer = _allocate_peer_comm_slots(
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
                _contiguous_chunk_view(send_slots[slot][peer_index], batch, rows, output_features)
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
                receive_slot = receive_slots[slot]
                assert receive_slot is not None
                receive_view = _contiguous_chunk_view(receive_slot, batch, rows, output_features)
            zero_view = (
                None
                if zero_local_buffer is None
                else _contiguous_chunk_view(zero_local_buffer, batch, rows, output_features)
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


def _run_distributed_projected_query_attention_output(  # noqa: PLR0913, PLR0917
    comm_group_size: int,
    comm_group_name: str,
    query_input: torch.Tensor,
    query_input_scale: torch.Tensor | None,
    query_weight_qdata: torch.Tensor,
    query_weight_scale: torch.Tensor,
    query_norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_norm_epsilon: float,
    softmax_scale: float,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    key_center_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_projection: output_ops.GateProjection | None = None,
    *,
    input_group_size: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    query_bias: torch.Tensor | None = None,
    output_input_scale: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    residual_add: ResidualAdd | None = None,
) -> torch.Tensor:
    """Overlap peer chunk reduce-scatter behind own-slice attention + residual epilogue."""
    projection_backend = fusion_backend.require_projection_backend(
        query_input, head_dim=key.shape[-1]
    )
    backend = fusion_backend.require_output_backend(key)
    prepared = output_common.prepare_attention_context(
        key,
        key_scale,
        key_center,
        key_center_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        ratio_profile,
        sparse_key_blocks,
        logical_sequence_length,
        routing_mode,
        block_lengths,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        has_projected_coarse_gate=gate_projection is not None,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    sequence_length = prepared.sequence_length
    local_sequence_length = _validate_distributed_sequence(
        sequence_length, comm_group_size, comm_group_name
    )
    process_group = distributed_comm.resolve_process_group(comm_group_name)
    rank = process_group.rank()
    world_size = process_group.size()
    if world_size != comm_group_size:
        raise ValueError("distributed sparse Piper comm_group_size must match resolved group")

    output_ops.validate_projected_query_input(
        query_input,
        query_input_scale,
        query_weight_qdata,
        query_weight_scale,
        key,
        sequence_length,
        query_norm_weight=query_norm_weight,
        cos=cos,
        sin=sin,
        query_norm_epsilon=query_norm_epsilon,
        softmax_scale=softmax_scale,
        input_group_size=input_group_size,
        query_bias=query_bias,
    )
    if prepared.coarse_gate is not None:
        output_common.validate_coarse_gate(
            prepared.coarse_gate,
            batch=key.shape[0],
            sequence_length=sequence_length,
            heads=key.shape[1],
            head_dim=key.shape[3],
            dtype=output_dtype,
            device=key.device,
        )
    validate_activation_scale(output_input_scale, key.device)
    _input_features, output_features = output_ops.validate_output_projection(
        key,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
        output_dtype=output_dtype,
    )

    if local_sequence_length < TILE_ROWS or local_sequence_length % TILE_ROWS:
        full_output = output_ops.run_projected_query_attention_output(
            query_input,
            query_input_scale,
            query_weight_qdata,
            query_weight_scale,
            query_norm_weight,
            cos,
            sin,
            query_norm_epsilon,
            softmax_scale,
            key,
            key_scale,
            key_center,
            key_center_scale,
            key_summary,
            key_aux,
            value,
            value_scale_multiplier,
            value_mean,
            ratio_profile,
            sparse_key_blocks,
            logical_sequence_length,
            routing_mode,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            query_chunk_rows,
            block_lengths,
            coarse_gate,
            coarse_scale,
            coarse_key_blocks,
            sparse_query_blocks,
            gate_projection,
            input_group_size=input_group_size,
            output_dtype=output_dtype,
            query_bias=query_bias,
            output_input_scale=output_input_scale,
            force_tile_ids=force_tile_ids,
            force_tile_counts=force_tile_counts,
        )
        return _run_distributed_fallback(
            process_group,
            comm_group_size,
            local_sequence_length,
            output_features,
            output_dtype,
            full_output,
            residual_add,
        )

    project_query_chunk = output_ops.build_query_chunk_projector(
        query_input,
        query_input_scale,
        query_weight_qdata,
        query_weight_scale,
        query_norm_weight,
        cos,
        sin,
        query_norm_epsilon,
        softmax_scale,
        routing_mode,
        block_lengths,
        backend=backend,
        projection_backend=projection_backend,
        head_dim=key.shape[-1],
        input_group_size=input_group_size,
        query_bias=query_bias,
    )

    comm_chunk_rows = _choose_distributed_chunk_rows(local_sequence_length, query_chunk_rows)
    batch, heads, _storage_seq, head_dim = key.shape
    peer_projector, _peer_retained = projection_output.prepare_chunk_projector(
        local_sequence_length,
        comm_chunk_rows,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        backend=backend,
        output_input_scale=output_input_scale,
        residual_add=None,
    )
    output = (
        residual_add.base
        if residual_add is not None
        else output_common.new_projected_output(
            key, local_sequence_length, output_features, output_dtype=output_dtype
        )
    )
    reuse_direct_receive = residual_add is None and batch == 1

    def apply_received(receive_tensor: torch.Tensor, local_offset: int, rows: int) -> None:
        if residual_add is not None:
            for batch_index in range(batch):
                apply_residual_add(
                    receive_tensor[batch_index],
                    residual_add,
                    row_offset=batch_index * local_sequence_length + local_offset,
                )
        elif not reuse_direct_receive:
            output[:, local_offset : local_offset + rows].copy_(receive_tensor)

    last_work, last_offset, last_rows, last_receive, last_comm_keepalive = (
        _run_peer_attention_exchange(
            process_group,
            prepared,
            (batch, heads, head_dim),
            output,
            comm_chunk_rows,
            project_query_chunk,
            gate_projection,
            apply_received,
            peer_projector,
            reuse_direct_receive=reuse_direct_receive,
        )
    )
    del peer_projector, _peer_retained

    own_partial: torch.Tensor | None = None
    try:
        own_base = rank * local_sequence_length
        own_gate_project = (
            None
            if gate_projection is None
            else (lambda buf, start, rows: gate_projection.project(buf, own_base + start, rows))
        )
        own_projector, own_retained = projection_output.prepare_chunk_projector(
            local_sequence_length,
            comm_chunk_rows,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            backend=backend,
            output_input_scale=output_input_scale,
            residual_add=residual_add,
        )
        if residual_add is not None:
            output_common.run_chunked_projected_query_attention_output(
                prepared,
                output_features,
                comm_chunk_rows,
                project_query_chunk,
                own_projector,
                own_retained,
                project_coarse_gate_chunk=own_gate_project,
                output_dtype=output_dtype,
                out=residual_add.base,
                reuse_output_for_attention=False,
                global_row_offset=own_base,
                sequence_length=local_sequence_length,
            )
        else:
            own_partial = output_common.run_chunked_projected_query_attention_output(
                prepared,
                output_features,
                comm_chunk_rows,
                project_query_chunk,
                own_projector,
                own_retained,
                project_coarse_gate_chunk=own_gate_project,
                output_dtype=output_dtype,
                reuse_output_for_attention=True,
                global_row_offset=own_base,
                sequence_length=local_sequence_length,
            )
    except BaseException:
        last_work.wait()
        del last_comm_keepalive, last_receive
        raise
    last_work.wait()
    del last_comm_keepalive
    apply_received(last_receive, last_offset, last_rows)
    if residual_add is not None:
        return residual_add.base
    assert own_partial is not None
    output.add_(own_partial)
    return output


def _validate_distributed_residual_add(
    comm_group_size: int,
    comm_group_name: str,
    residual_add: ResidualAdd,
    key: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    weight_qdata: torch.Tensor,
    force_tile_ids: torch.Tensor | None,
    force_tile_counts: torch.Tensor | None,
) -> None:
    """Check from metadata that ``residual_add.base`` has one rank's rows and dtype."""
    base = residual_add.base
    sequence_length = output_common.validate_output_metadata(
        key,
        logical_sequence_length,
        block_lengths,
        output_dtype=base.dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    local_sequence_length = _validate_distributed_sequence(
        sequence_length, comm_group_size, comm_group_name
    )
    if base.ndim != 3 or base.shape[:2] != (key.shape[0], local_sequence_length):
        raise ValueError(
            "distributed sparse Piper residual base must have one local rank share of rows"
        )
    if base.device != key.device:
        raise ValueError(
            "distributed sparse Piper residual base must share the attention storage device"
        )
    validate_residual_add(base, residual_add, weight_qdata.shape[0])


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_projected_query_attention_reduce_scatter_residual_add_",
    mutates_args=("base",),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _projected_query_attention_reduce_scatter_residual_add_op(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    comm_group_size: int,
    comm_group_name: str,
    query_input: torch.Tensor,
    query_input_scale: torch.Tensor | None,
    query_weight_qdata: torch.Tensor,
    query_weight_scale: torch.Tensor,
    query_norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_norm_epsilon: float,
    softmax_scale: float,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    key_center_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    query_bias: torch.Tensor | None = None,
    output_input_scale: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    input_group_size: int | None = None,
) -> None:
    """Compute distributed sparse attention output + reduce-scatter into ``base``.

    To bound peer exchange memory on sequence-aligned chunks, peer and local rank
    contributions are added to ``base`` in two rounded steps rather than combined
    into one temporary buffer before residual addition.
    """
    residual_add = ResidualAdd(
        base=base,
        gate=gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    _validate_distributed_residual_add(
        comm_group_size,
        comm_group_name,
        residual_add,
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata,
        force_tile_ids,
        force_tile_counts,
    )
    gate_projection = output_ops.prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        output_ops.resolve_projected_gate_input(
            query_input,
            gate_input,
            gate_weight_qdata,
            input_group_size,
        ),
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        input_group_size,
    )
    _run_distributed_projected_query_attention_output(
        comm_group_size,
        comm_group_name,
        query_input,
        query_input_scale,
        query_weight_qdata,
        query_weight_scale,
        query_norm_weight,
        cos,
        sin,
        query_norm_epsilon,
        softmax_scale,
        key,
        key_scale,
        key_center,
        key_center_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        ratio_profile,
        sparse_key_blocks,
        logical_sequence_length,
        routing_mode,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        query_chunk_rows,
        block_lengths,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
        input_group_size=input_group_size,
        output_dtype=base.dtype,
        query_bias=query_bias,
        output_input_scale=output_input_scale,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
        residual_add=residual_add,
    )


@_projected_query_attention_reduce_scatter_residual_add_op.register_fake
def _projected_query_attention_reduce_scatter_residual_add_op_fake(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    comm_group_size: int,
    comm_group_name: str,
    _query_input: torch.Tensor,
    _query_input_scale: torch.Tensor | None,
    _query_weight_qdata: torch.Tensor,
    _query_weight_scale: torch.Tensor,
    _query_norm_weight: torch.Tensor | None,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _query_norm_epsilon: float,
    _softmax_scale: float,
    key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_center: torch.Tensor,
    _key_center_scale: torch.Tensor,
    _key_summary: torch.Tensor,
    _key_aux: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _ratio_profile: _SparsePiperRatioProfile,
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _routing_mode: int,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    _query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
    _query_bias: torch.Tensor | None = None,
    _output_input_scale: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    input_group_size: int | None = None,  # noqa: ARG001 - custom-op keyword
) -> None:
    _validate_distributed_residual_add(
        comm_group_size,
        comm_group_name,
        ResidualAdd(
            base=base,
            gate=gate,
            gate_indices=gate_indices,
            python_indexing=python_indexing,
        ),
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata,
        force_tile_ids,
        force_tile_counts,
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_projected_query_attention_reduce_scatter",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _projected_query_attention_reduce_scatter_op(  # noqa: PLR0913, PLR0917
    comm_group_size: int,
    comm_group_name: str,
    query_input: torch.Tensor,
    query_input_scale: torch.Tensor | None,
    query_weight_qdata: torch.Tensor,
    query_weight_scale: torch.Tensor,
    query_norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_norm_epsilon: float,
    softmax_scale: float,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    key_center_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    query_bias: torch.Tensor | None = None,
    output_input_scale: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    input_group_size: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute distributed sparse attention output + reduce-scatter into this rank's rows."""
    gate_projection = output_ops.prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        output_ops.resolve_projected_gate_input(
            query_input,
            gate_input,
            gate_weight_qdata,
            input_group_size,
        ),
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        input_group_size,
    )
    return _run_distributed_projected_query_attention_output(
        comm_group_size,
        comm_group_name,
        query_input,
        query_input_scale,
        query_weight_qdata,
        query_weight_scale,
        query_norm_weight,
        cos,
        sin,
        query_norm_epsilon,
        softmax_scale,
        key,
        key_scale,
        key_center,
        key_center_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        ratio_profile,
        sparse_key_blocks,
        logical_sequence_length,
        routing_mode,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        query_chunk_rows,
        block_lengths,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
        input_group_size=input_group_size,
        output_dtype=output_dtype,
        query_bias=query_bias,
        output_input_scale=output_input_scale,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )


@_projected_query_attention_reduce_scatter_op.register_fake
def _projected_query_attention_reduce_scatter_op_fake(
    comm_group_size: int,
    comm_group_name: str,
    _query_input: torch.Tensor,
    _query_input_scale: torch.Tensor | None,
    _query_weight_qdata: torch.Tensor,
    _query_weight_scale: torch.Tensor,
    _query_norm_weight: torch.Tensor | None,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _query_norm_epsilon: float,
    _softmax_scale: float,
    key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_center: torch.Tensor,
    _key_center_scale: torch.Tensor,
    _key_summary: torch.Tensor,
    _key_aux: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _ratio_profile: _SparsePiperRatioProfile,
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _routing_mode: int,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    _query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
    _query_bias: torch.Tensor | None = None,
    _output_input_scale: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    input_group_size: int | None = None,  # noqa: ARG001 - custom-op keyword
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    sequence_length = output_common.validate_output_metadata(
        key,
        logical_sequence_length,
        block_lengths,
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    local_sequence_length = _validate_distributed_sequence(
        sequence_length, comm_group_size, comm_group_name
    )
    return output_common.new_projected_output(
        key, local_sequence_length, weight_qdata.shape[0], output_dtype=output_dtype
    )


__all__: list[str] = []
