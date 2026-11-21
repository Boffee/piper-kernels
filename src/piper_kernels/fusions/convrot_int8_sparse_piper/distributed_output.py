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
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch
from piper_kernels.attention.sparse_piper_attention._contracts import _SparsePiperRatioProfile
from piper_kernels.fusions.convrot_int8_projection import output as projection_output
from piper_kernels.fusions.sparse_piper import _output as output_common
from piper_kernels.weights.convrot.int8._quantization import validate_activation_scale

from . import _backend as fusion_backend
from . import output as output_ops
from ._layout import TILE_ROWS

_DEFAULT_QUERY_CHUNK_ROWS = output_ops.DEFAULT_QUERY_CHUNK_ROWS


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
    batch, heads, _storage_seq, head_dim = key.shape

    def run_fallback() -> torch.Tensor:
        return output_ops.run_projected_query_attention_output(
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

    def start_aligned(
        comm_chunk_rows: int,
        output: torch.Tensor,
        apply_received: peer_exchange.ApplyReceived,
        reuse_direct_receive: bool,
        own_base: int,
    ) -> tuple[peer_exchange.PeerExchangeHandle, Callable[[], torch.Tensor | None]]:
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
            head_dim=head_dim,
            input_group_size=input_group_size,
            query_bias=query_bias,
        )
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
            own_output = output_common.run_chunked_projected_query_attention_output(
                prepared,
                output_features,
                comm_chunk_rows,
                project_query_chunk,
                own_projector,
                own_retained,
                project_coarse_gate_chunk=own_gate_project,
                output_dtype=output_dtype,
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
        output_dtype=output_dtype,
        device=key.device,
        query_chunk_rows=query_chunk_rows,
        tile_rows=TILE_ROWS,
        residual_add=residual_add,
        run_fallback=run_fallback,
        start_aligned=start_aligned,
    )


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
    local_sequence_length = peer_exchange.validate_distributed_sequence(
        sequence_length,
        comm_group_size,
        comm_group_name,
        name="distributed sparse Piper",
    )
    return output_common.new_projected_output(
        key, local_sequence_length, weight_qdata.shape[0], output_dtype=output_dtype
    )


__all__: list[str] = []
