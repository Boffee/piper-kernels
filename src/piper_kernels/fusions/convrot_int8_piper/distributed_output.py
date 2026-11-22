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
_DEFAULT_QUERY_CHUNK_ROWS = output_ops.DEFAULT_QUERY_CHUNK_ROWS


def _validate_distributed_dense_output(
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


def _run_distributed_dense_output(
    comm_group_size: int,
    comm_group_name: str,
    operands: output_ops.DenseOutputOperands,
    *,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool] | None = None,
) -> torch.Tensor:
    """Overlap peer chunk reduce-scatter behind own-slice dense attention + residual epilogue."""
    shape, local_sequence_length, residual_add = _validate_distributed_dense_output(
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


@torch.library.custom_op(
    "piper_kernels::convrot_int8_piper_projected_query_attention_reduce_scatter",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _projected_query_attention_reduce_scatter_op(  # noqa: PLR0913, PLR0917
    comm_group_size: int,
    comm_group_name: str,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    query_weight: torch.Tensor,
    query_weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    query_bias: torch.Tensor | None,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    key_length: int,
    is_causal: bool,
    output_dtype: torch.dtype,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    output_input_scale: torch.Tensor | None = None,
    *,
    head_dim: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
) -> torch.Tensor:
    """Compute distributed dense attention output + reduce-scatter into this rank's rows."""
    return _run_distributed_dense_output(
        comm_group_size,
        comm_group_name,
        output_ops.DenseOutputOperands(
            input_qdata,
            input_scale,
            query_weight,
            query_weight_scale,
            norm_weight,
            cos,
            sin,
            norm_epsilon,
            softmax_scale,
            query_bias,
            key,
            key_scale,
            key_center,
            value,
            multiplier,
            log_scale,
            value_mean,
            key_length,
            is_causal,
            output_dtype,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            output_input_scale,
            head_dim,
            query_chunk_rows,
        ),
    )


@_projected_query_attention_reduce_scatter_op.register_fake
def _projected_query_attention_reduce_scatter_op_fake(  # noqa: PLR0913, PLR0917
    comm_group_size: int,
    comm_group_name: str,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    query_weight: torch.Tensor,
    query_weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    query_bias: torch.Tensor | None,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    key_length: int,
    is_causal: bool,
    output_dtype: torch.dtype,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    output_input_scale: torch.Tensor | None = None,
    *,
    head_dim: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
) -> torch.Tensor:
    (batch, _heads, _sequence, _head_dim), local_sequence_length, _residual_add = (
        _validate_distributed_dense_output(
            comm_group_size,
            comm_group_name,
            output_ops.DenseOutputOperands(
                input_qdata,
                input_scale,
                query_weight,
                query_weight_scale,
                norm_weight,
                cos,
                sin,
                norm_epsilon,
                softmax_scale,
                query_bias,
                key,
                key_scale,
                key_center,
                value,
                multiplier,
                log_scale,
                value_mean,
                key_length,
                is_causal,
                output_dtype,
                weight_qdata,
                weight_scale,
                bias,
                group_size,
                output_input_scale,
                head_dim,
                query_chunk_rows,
            ),
        )
    )
    return input_qdata.new_empty(
        (batch, local_sequence_length, weight_qdata.shape[0]), dtype=output_dtype
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_piper_projected_query_attention_reduce_scatter_residual_add_",
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
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    query_weight: torch.Tensor,
    query_weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    query_bias: torch.Tensor | None,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    key_length: int,
    is_causal: bool,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    output_input_scale: torch.Tensor | None = None,
    *,
    head_dim: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
) -> None:
    """Compute distributed dense attention output + reduce-scatter into ``base``.

    To bound peer exchange memory on 128-token rank-aligned sequences, peer and local
    rank contributions are added to ``base`` in two separately rounded steps.
    """
    _run_distributed_dense_output(
        comm_group_size,
        comm_group_name,
        output_ops.DenseOutputOperands(
            input_qdata,
            input_scale,
            query_weight,
            query_weight_scale,
            norm_weight,
            cos,
            sin,
            norm_epsilon,
            softmax_scale,
            query_bias,
            key,
            key_scale,
            key_center,
            value,
            multiplier,
            log_scale,
            value_mean,
            key_length,
            is_causal,
            base.dtype,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            output_input_scale,
            head_dim,
            query_chunk_rows,
        ),
        residual=(base, gate, gate_indices, python_indexing),
    )


@_projected_query_attention_reduce_scatter_residual_add_op.register_fake
def _projected_query_attention_reduce_scatter_residual_add_op_fake(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    comm_group_size: int,
    comm_group_name: str,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    query_weight: torch.Tensor,
    query_weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    query_bias: torch.Tensor | None,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    key_length: int,
    is_causal: bool,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    output_input_scale: torch.Tensor | None = None,
    *,
    head_dim: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
) -> None:
    _validate_distributed_dense_output(
        comm_group_size,
        comm_group_name,
        output_ops.DenseOutputOperands(
            input_qdata,
            input_scale,
            query_weight,
            query_weight_scale,
            norm_weight,
            cos,
            sin,
            norm_epsilon,
            softmax_scale,
            query_bias,
            key,
            key_scale,
            key_center,
            value,
            multiplier,
            log_scale,
            value_mean,
            key_length,
            is_causal,
            base.dtype,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            output_input_scale,
            head_dim,
            query_chunk_rows,
        ),
        residual=(base, gate, gate_indices, python_indexing),
    )


__all__: list[str] = []
