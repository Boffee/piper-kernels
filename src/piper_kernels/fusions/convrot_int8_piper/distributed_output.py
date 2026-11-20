"""Bounded peer-exchange reduce-scatter for distributed fused dense Piper attention.

Overlaps bounded peer attention + output projection chunks and collective exchange
behind own-rank local sequence computation. Unlike eager reduce-scatter followed by
``residual_add_``, overlapped peer-chunk residual updates add peer and local rank
contributions into ``base`` in two separately rounded steps to keep peer scratch
memory bounded.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from piper_kernels._distributed import _comm as distributed_comm
from piper_kernels._distributed import _peer_exchange as peer_exchange
from piper_kernels._residual import ResidualAdd, validate_residual_add
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    count_query_scales,
    resolve_qk_quantization,
)
from piper_kernels.attention.piper_attention import _quantized_dispatch as attention
from piper_kernels.fusions.attention import _output as pipeline
from piper_kernels.fusions.convrot_int8_projection import output as projection_output
from piper_kernels.linear.convrot.int8 import _backend as linear_backend

from . import _backend
from . import output as output_ops

_DENSE_ALIGN_ROWS = 128
_DEFAULT_QUERY_CHUNK_ROWS = output_ops.DEFAULT_QUERY_CHUNK_ROWS


def _validate_distributed_dense_output(  # noqa: PLR0913, PLR0917
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
    output_input_scale: torch.Tensor | None,
    *,
    head_dim: int,
    query_chunk_rows: int,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool] | None = None,
) -> tuple[tuple[int, int, int, int], int, ResidualAdd | None]:
    """Validate all distributed dense attention and optional residual metadata upfront."""
    resolved_dtype = residual[0].dtype if residual is not None else output_dtype
    shape = output_ops.validate_output_inputs(
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
        resolved_dtype,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        output_input_scale,
        head_dim=head_dim,
        query_chunk_rows=query_chunk_rows,
    )
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
    if base.ndim != 3 or base.shape[:2] != (batch, local_sequence_length):
        raise ValueError(
            "distributed dense Piper residual base must have one local rank share of rows"
        )
    if base.device != input_qdata.device:
        raise ValueError(
            "distributed dense Piper residual base must share the input storage device"
        )
    residual_add = ResidualAdd(
        base=base,
        gate=gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    validate_residual_add(base, residual_add, weight_qdata.shape[0])
    return shape, local_sequence_length, residual_add


def _run_distributed_dense_output(  # noqa: PLR0913, PLR0917
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
    output_input_scale: torch.Tensor | None,
    *,
    head_dim: int,
    query_chunk_rows: int,
    residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool] | None = None,
) -> torch.Tensor:
    """Overlap peer chunk reduce-scatter behind own-slice dense attention + residual epilogue."""
    shape, local_sequence_length, residual_add = _validate_distributed_dense_output(
        comm_group_size,
        comm_group_name,
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
        head_dim=head_dim,
        query_chunk_rows=query_chunk_rows,
        residual=residual,
    )
    batch, heads, sequence, resolved_head_dim = shape
    resolved_output_dtype = residual_add.base.dtype if residual_add is not None else output_dtype
    output_features = weight_qdata.shape[0]
    process_group = distributed_comm.resolve_process_group(comm_group_name)
    rank = process_group.rank()
    world_size = process_group.size()
    if world_size != comm_group_size:
        raise ValueError("distributed dense Piper comm_group_size must match resolved group")
    if batch == 0:
        return (
            residual_add.base
            if residual_add is not None
            else input_qdata.new_empty(
                (0, local_sequence_length, output_features), dtype=resolved_output_dtype
            )
        )

    output_backend = linear_backend.require_linear_backend(input_qdata)
    context_operands = (key, key_scale, key_center, value, multiplier, log_scale, value_mean)

    if local_sequence_length < _DENSE_ALIGN_ROWS or local_sequence_length % _DENSE_ALIGN_ROWS:

        def build_full_projector(
            chunk_rows: int,
        ) -> tuple[pipeline.ChunkProjector, tuple[torch.Tensor, ...]]:
            return projection_output.prepare_chunk_projector(
                sequence,
                chunk_rows,
                weight_qdata,
                weight_scale,
                bias,
                group_size,
                backend=output_backend,
                output_input_scale=output_input_scale,
            )

        full_output = output_ops.run_projected_query_attention(
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
            context_operands,
            key_length,
            is_causal,
            shape,
            output_features,
            resolved_output_dtype,
            query_chunk_rows,
            build_full_projector,
        )
        return peer_exchange.run_distributed_fallback(
            process_group,
            comm_group_size,
            local_sequence_length,
            output_features,
            resolved_output_dtype,
            full_output,
            residual_add,
        )

    projection_backend = _backend.require_projection_backend(
        input_qdata, head_dim=resolved_head_dim
    )
    comm_chunk_rows = peer_exchange.choose_distributed_chunk_rows(
        local_sequence_length,
        query_chunk_rows,
        tile_rows=_DENSE_ALIGN_ROWS,
    )
    context = attention.prepare_quantized_context(
        *context_operands,
        query_length=sequence,
        key_length=key_length,
        is_causal=is_causal,
    )
    capacity = min(comm_chunk_rows, local_sequence_length)
    granularity = resolve_qk_quantization(key_scale, key.shape[2])
    query_buffers = (
        input_qdata.new_empty((batch, heads, capacity, resolved_head_dim)),
        input_qdata.new_empty(
            (batch, heads, count_query_scales(capacity, granularity)), dtype=torch.float32
        ),
    )
    output = (
        residual_add.base
        if residual_add is not None
        else input_qdata.new_empty(
            (batch, local_sequence_length, output_features), dtype=resolved_output_dtype
        )
    )
    reuse_direct_receive = residual_add is None and batch == 1

    def apply_received(receive_tensor: torch.Tensor, local_offset: int, rows: int) -> None:
        peer_exchange.apply_received_chunk(
            receive_tensor,
            output,
            local_offset,
            rows,
            residual_add=residual_add,
            reuse_direct_receive=reuse_direct_receive,
        )

    def launch_peer_exchange() -> tuple[
        dist.Work, int, int, torch.Tensor, tuple[torch.Tensor, ...]
    ]:
        attention_scratch: torch.Tensor | None = (
            None
            if output_features >= heads * resolved_head_dim
            else input_qdata.new_empty(
                (batch, capacity, heads, resolved_head_dim),
                device=input_qdata.device,
                dtype=resolved_output_dtype,
            )
        )
        peer_projector, _peer_retained = projection_output.prepare_chunk_projector(
            local_sequence_length,
            comm_chunk_rows,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            backend=output_backend,
            output_input_scale=output_input_scale,
            residual_add=None,
        )

        def run_peer_window(global_start: int, rows: int, output_buffer: torch.Tensor) -> None:
            attention_chunk = (
                output_buffer.view(batch, rows * output_features)[
                    :, : rows * heads * resolved_head_dim
                ].view(batch, rows, heads, resolved_head_dim)
                if attention_scratch is None
                else attention_scratch[:, :rows]
            )
            projection_backend.project_query(
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
                chunk_start=global_start,
                chunk_rows=rows,
                out=query_buffers,
            )
            attention.launch_quantized_attention_into(
                context,
                *query_buffers,
                attention_chunk.transpose(1, 2),
                global_row_offset=global_start,
            )
            peer_projector(attention_chunk, output_buffer, 0, rows)

        return peer_exchange.run_peer_chunk_exchange(
            process_group,
            output,
            comm_chunk_rows,
            run_peer_window,
            apply_received,
            reuse_direct_receive=reuse_direct_receive,
        )

    last_exchange = launch_peer_exchange()

    def run_own() -> torch.Tensor | None:
        own_base = rank * local_sequence_length

        def build_own_projector(
            chunk_rows: int,
        ) -> tuple[pipeline.ChunkProjector, tuple[torch.Tensor, ...]]:
            return projection_output.prepare_chunk_projector(
                local_sequence_length,
                chunk_rows,
                weight_qdata,
                weight_scale,
                bias,
                group_size,
                backend=output_backend,
                output_input_scale=output_input_scale,
                residual_add=residual_add,
            )

        own_output = output_ops.run_projected_query_attention(
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
            context_operands,
            key_length,
            is_causal,
            shape,
            output_features,
            resolved_output_dtype,
            comm_chunk_rows,
            build_own_projector,
            out=None if residual_add is None else residual_add.base,
            global_row_offset=own_base,
            sequence_length=local_sequence_length,
            context=context,
            query_buffers=query_buffers,
        )
        return None if residual_add is not None else own_output

    try:
        return peer_exchange.finish_overlapped_exchange(
            last_exchange,
            run_own=run_own,
            apply_received=apply_received,
            output=output,
            residual_add=residual_add,
        )
    finally:
        del last_exchange


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
        head_dim=head_dim,
        query_chunk_rows=query_chunk_rows,
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
            head_dim=head_dim,
            query_chunk_rows=query_chunk_rows,
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
        head_dim=head_dim,
        query_chunk_rows=query_chunk_rows,
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
        head_dim=head_dim,
        query_chunk_rows=query_chunk_rows,
        residual=(base, gate, gate_indices, python_indexing),
    )


__all__: list[str] = []
