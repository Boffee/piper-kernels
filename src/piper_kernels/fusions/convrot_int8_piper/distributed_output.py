"""Bounded peer-exchange reduce-scatter for distributed fused dense Piper attention.

Overlaps bounded peer attention + output projection chunks and collective exchange
behind own-rank local sequence computation. Unlike eager reduce-scatter followed by
``residual_add_``, overlapped peer-chunk residual updates add peer and local rank
contributions into ``base`` in two separately rounded steps to keep peer scratch
memory bounded.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from piper_kernels._distributed import _peer_exchange as peer_exchange
from piper_kernels._residual import ResidualAdd
from piper_kernels.linear.convrot.int8 import _backend as linear_backend

from . import output as output_ops

_DENSE_ALIGN_ROWS = 128


def validate_distributed_dense_output(
    comm_group_size: int,
    comm_group_name: str,
    operands: output_ops.DenseOutputOperands,
    *,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool] | None = None,
) -> tuple[tuple[int, int, int, int], int, ResidualAdd | None]:
    """Validate all distributed dense attention and optional residual metadata upfront."""
    shape = operands.validate()
    batch, _heads, sequence, _head_dim = shape
    local_sequence_length = peer_exchange.validate_distributed_sequence(
        sequence,
        comm_group_size,
        comm_group_name,
        name="distributed dense Piper",
    )
    if residual is None:
        return shape, local_sequence_length, None

    base, gate, gate_indices, python_indexing = residual
    residual_add = ResidualAdd(
        base=base,
        gate=gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    peer_exchange.validate_distributed_residual_add(
        residual_add,
        batch=batch,
        local_sequence_length=local_sequence_length,
        output_features=operands.weight_qdata.shape[0],
        storage_device=operands.input_qdata.device,
        name="distributed dense Piper",
        storage_label="input",
    )
    return shape, local_sequence_length, residual_add


def run_distributed_dense_output(
    comm_group_size: int,
    comm_group_name: str,
    operands: output_ops.DenseOutputOperands,
    *,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool] | None = None,
) -> torch.Tensor:
    """Overlap peer chunk reduce-scatter behind own-slice dense attention + residual epilogue."""
    shape, local_sequence_length, residual_add = validate_distributed_dense_output(
        comm_group_size,
        comm_group_name,
        operands,
        residual=residual,
    )
    batch, heads, _sequence, resolved_head_dim = shape
    output_features = operands.weight_qdata.shape[0]
    process_group = peer_exchange.resolve_validated_process_group(
        comm_group_name,
        comm_group_size,
        name="distributed dense Piper",
    )
    if batch == 0:
        return (
            residual_add.base
            if residual_add is not None
            else operands.input_qdata.new_empty(
                (0, local_sequence_length, output_features), dtype=operands.output_dtype
            )
        )

    output_backend = linear_backend.require_linear_backend(operands.input_qdata)

    def run_fallback() -> torch.Tensor:
        return operands.run(shape, output_backend=output_backend)

    def start_aligned(
        comm_chunk_rows: int,
        output: torch.Tensor,
        apply_received: peer_exchange.ApplyReceived,
        reuse_direct_receive: bool,
        own_base: int,
    ) -> tuple[peer_exchange.PeerExchangeHandle, Callable[[], torch.Tensor | None]]:
        context, query_buffers, run_attention = operands.prepare_window_runner(
            shape, min(comm_chunk_rows, local_sequence_length)
        )
        peer_projector, _peer_retained = operands.prepare_chunk_projector(
            local_sequence_length,
            comm_chunk_rows,
            backend=output_backend,
        )
        last_exchange = peer_exchange.run_peer_attention_exchange(
            process_group,
            output,
            heads=heads,
            head_dim=resolved_head_dim,
            comm_chunk_rows=comm_chunk_rows,
            run_attention=run_attention,
            peer_projector=peer_projector,
            apply_received=apply_received,
            reuse_direct_receive=reuse_direct_receive,
        )

        def run_own() -> torch.Tensor | None:
            own_output = operands.run(
                shape,
                output_backend=output_backend,
                residual_add=residual_add,
                out=None if residual_add is None else residual_add.base,
                global_row_offset=own_base,
                sequence_length=local_sequence_length,
                query_chunk_rows=comm_chunk_rows,
                context=context,
                query_buffers=query_buffers,
            )
            return None if residual_add is not None else own_output

        return last_exchange, run_own

    return peer_exchange.run_overlapped_distributed_output(
        process_group,
        output_shape=(batch, local_sequence_length, output_features),
        output_dtype=operands.output_dtype,
        device=operands.input_qdata.device,
        query_chunk_rows=operands.query_chunk_rows,
        tile_rows=_DENSE_ALIGN_ROWS,
        residual_add=residual_add,
        run_fallback=run_fallback,
        start_aligned=start_aligned,
    )


__all__: list[str] = []
