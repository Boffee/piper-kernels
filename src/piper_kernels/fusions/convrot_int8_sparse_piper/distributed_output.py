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

from piper_kernels._distributed import _peer_exchange as peer_exchange
from piper_kernels._residual import ResidualAdd
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS
from piper_kernels.fusions.attention import _sparse_output as output_common
from piper_kernels.fusions.convrot_int8_projection import output as projection_output
from piper_kernels.weights.convrot.int8._quantization import validate_activation_scale

from . import output as output_ops


def _run_peer_attention_exchange(
    process_group: dist.ProcessGroup,
    prepared: output_common._PreparedAttentionContext,
    key_dims: tuple[int, int, int],
    output: torch.Tensor,
    comm_chunk_rows: int,
    project_query_chunk: output_common.QueryChunkProjector,
    gate_projection: output_ops.GateProjection | None,
    apply_received: peer_exchange.ApplyReceived,
    peer_projector: output_common.ChunkProjector,
    *,
    reuse_direct_receive: bool,
) -> peer_exchange.PeerExchangeHandle:
    batch, local_sequence_length, _output_features = output.shape
    heads, head_dim = key_dims[1], key_dims[2]
    capacity = min(local_sequence_length, comm_chunk_rows)
    gate_scratch: torch.Tensor | None = (
        torch.empty((batch, capacity, heads, head_dim), device=output.device, dtype=output.dtype)
        if gate_projection is not None
        else None
    )

    def run_attention(global_start: int, rows: int, attention_chunk: torch.Tensor) -> None:
        gate_chunk: torch.Tensor | None = None
        if gate_projection is not None:
            assert gate_scratch is not None
            gate_chunk = gate_scratch[:, :rows]
            gate_projection.project(gate_chunk, global_start, rows)
        elif prepared.coarse_gate is not None:
            gate_chunk = prepared.coarse_gate[:, global_start : global_start + rows]
        output_common.launch_projected_query_chunk(
            prepared,
            project_query_chunk,
            global_start,
            rows,
            attention_chunk,
            coarse_gate=gate_chunk,
        )

    return peer_exchange.run_peer_attention_exchange(
        process_group,
        output,
        heads=heads,
        head_dim=head_dim,
        comm_chunk_rows=comm_chunk_rows,
        run_attention=run_attention,
        peer_projector=peer_projector,
        apply_received=apply_received,
        reuse_direct_receive=reuse_direct_receive,
    )


def run_distributed_projected_query_attention_output(
    comm_group_size: int,
    comm_group_name: str,
    operands: output_ops.SparseProjectedQueryOperands,
    *,
    gate_projection: output_ops.GateProjection | None = None,
    residual_add: ResidualAdd | None = None,
) -> torch.Tensor:
    """Overlap peer chunk reduce-scatter behind own-slice attention + residual epilogue."""
    sequence_length = output_common.validate_output_metadata(
        operands.key,
        operands.logical_sequence_length,
        operands.block_lengths,
        output_dtype=operands.output_dtype,
        force_tile_ids=operands.force_tile_ids,
        force_tile_counts=operands.force_tile_counts,
    )
    local_sequence_length = peer_exchange.validate_distributed_sequence(
        sequence_length,
        comm_group_size,
        comm_group_name,
        name="distributed sparse Piper",
    )
    process_group = peer_exchange.resolve_validated_process_group(
        comm_group_name,
        comm_group_size,
        name="distributed sparse Piper",
    )
    backend, prepared, project_query_chunk = operands.prepare_context(
        has_projected_coarse_gate=gate_projection is not None,
    )
    if prepared.coarse_gate is not None:
        output_common.validate_coarse_gate(
            prepared.coarse_gate,
            batch=operands.key.shape[0],
            sequence_length=sequence_length,
            heads=operands.key.shape[1],
            head_dim=operands.key.shape[3],
            dtype=operands.output_dtype,
            device=operands.key.device,
        )
    validate_activation_scale(operands.output_input_scale, operands.key.device)
    _input_features, output_features = output_ops.validate_output_projection(
        operands.key,
        operands.weight_qdata,
        operands.weight_scale,
        operands.bias,
        operands.group_size,
        operands.logical_sequence_length,
        operands.query_chunk_rows,
        output_dtype=operands.output_dtype,
    )
    batch, heads, _storage_seq, head_dim = operands.key.shape

    def prepare_projector(
        seq_rows: int,
        chunk_rows: int,
        *,
        residual: ResidualAdd | None = None,
    ) -> tuple[output_common.ChunkProjector, tuple[torch.Tensor, ...]]:
        return projection_output.prepare_chunk_projector(
            seq_rows,
            chunk_rows,
            operands.weight_qdata,
            operands.weight_scale,
            operands.bias,
            operands.group_size,
            backend=backend,
            output_input_scale=operands.output_input_scale,
            residual_add=residual,
        )

    def run_fallback() -> torch.Tensor:
        project_chunk, projector_tensors = prepare_projector(
            sequence_length, operands.query_chunk_rows
        )
        return output_common.run_chunked_projected_query_attention_output(
            prepared,
            output_features,
            operands.query_chunk_rows,
            project_query_chunk,
            project_chunk,
            projector_tensors,
            project_coarse_gate_chunk=(
                None if gate_projection is None else gate_projection.project
            ),
            output_dtype=operands.output_dtype,
            reuse_output_for_attention=True,
        )

    def start_aligned(
        comm_chunk_rows: int,
        output: torch.Tensor,
        apply_received: peer_exchange.ApplyReceived,
        reuse_direct_receive: bool,
        own_base: int,
    ) -> tuple[peer_exchange.PeerExchangeHandle, Callable[[], torch.Tensor | None]]:
        peer_projector, _peer_retained = prepare_projector(local_sequence_length, comm_chunk_rows)
        last_exchange = _run_peer_attention_exchange(
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
        del peer_projector, _peer_retained

        def run_own() -> torch.Tensor | None:
            if gate_projection is None:
                own_gate_project = None
            else:
                active_gate = gate_projection
                own_gate_project = lambda buf, start, rows: active_gate.project(  # noqa: E731
                    buf, own_base + start, rows
                )
            own_projector, own_retained = prepare_projector(
                local_sequence_length, comm_chunk_rows, residual=residual_add
            )
            own_output = output_common.run_chunked_projected_query_attention_output(
                prepared,
                output_features,
                comm_chunk_rows,
                project_query_chunk,
                own_projector,
                own_retained,
                project_coarse_gate_chunk=own_gate_project,
                output_dtype=operands.output_dtype,
                out=None if residual_add is None else residual_add.base,
                reuse_output_for_attention=residual_add is None,
                global_row_offset=own_base,
                sequence_length=local_sequence_length,
            )
            return None if residual_add is not None else own_output

        return last_exchange, run_own

    return peer_exchange.run_overlapped_distributed_output(
        process_group,
        output_shape=(batch, local_sequence_length, output_features),
        output_dtype=operands.output_dtype,
        device=operands.key.device,
        query_chunk_rows=operands.query_chunk_rows,
        tile_rows=TILE_ROWS,
        residual_add=residual_add,
        run_fallback=run_fallback,
        start_aligned=start_aligned,
    )


def validate_distributed_residual_add(
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
    sequence_length = output_common.validate_output_metadata(
        key,
        logical_sequence_length,
        block_lengths,
        output_dtype=residual_add.base.dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    local_sequence_length = peer_exchange.validate_distributed_sequence(
        sequence_length,
        comm_group_size,
        comm_group_name,
        name="distributed sparse Piper",
    )
    peer_exchange.validate_distributed_residual_add(
        residual_add,
        batch=key.shape[0],
        local_sequence_length=local_sequence_length,
        output_features=weight_qdata.shape[0],
        storage_device=key.device,
        name="distributed sparse Piper",
        storage_label="attention",
    )


__all__: list[str] = []
