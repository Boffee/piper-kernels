"""Bounded-workspace sparse Piper attention followed by a ConvRot INT8 output projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.fusions.sparse_piper import _output as output_common
from piper_kernels.linear import _bias
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8._interfaces import LinearBackend
from piper_kernels.weights.convrot.int8._quantization import (
    validate_activation_scale,
    validate_storage,
)

from . import _backend as fusion_backend
from . import query as query_projection
from ._layout import TILE_ROWS

_DEFAULT_QUERY_CHUNK_ROWS = output_common.DEFAULT_QUERY_CHUNK_ROWS


@dataclass(frozen=True, slots=True)
class _GateProjection:
    """One ConvRot INT8 gate with prepared or chunk-prepared hidden states."""

    input_data: torch.Tensor
    input_scale: torch.Tensor | None
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None
    backend: LinearBackend
    input_group_size: int | None = None

    def project(self, output: torch.Tensor, start: int, rows: int) -> None:
        """Project one sequence window into caller-owned token-major gate storage."""
        output_features = self.weight_qdata.shape[0]
        for batch_index in range(self.input_data.shape[0]):
            chunk_input = self.input_data[batch_index, start : start + rows]
            if self.input_group_size is None:
                assert self.input_scale is not None
                chunk_scale = self.input_scale[batch_index, start : start + rows]
            else:
                chunk_input, chunk_scale = self.backend.prepare_input(
                    chunk_input, self.input_group_size, input_scale=self.input_scale
                )
            self.backend.linear_prepared(
                chunk_input,
                chunk_scale,
                self.weight_qdata,
                self.weight_scale,
                self.bias,
                output.dtype,
                out=output[batch_index, :rows].reshape(rows, output_features),
            )


def _prepare_gate_projection(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    input_data: torch.Tensor,
    input_scale: torch.Tensor | None,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    input_group_size: int | None = None,
) -> _GateProjection:
    """Validate the input and one D64/D128-per-head gate weight."""
    sequence_length = output_common.output_sequence_length(
        attention_storage,
        logical_sequence_length,
        block_lengths,
    )
    batch, heads, _storage_sequence_length, head_dim = attention_storage.shape
    if (
        input_data.ndim != 3
        or input_data.shape[:2] != (batch, sequence_length)
        or input_data.device != attention_storage.device
        or input_data.layout is not torch.strided
        or (input_group_size is None and not input_data.is_contiguous())
    ):
        raise ValueError("fused ConvRot INT8 gate input must be compatible batch/sequence storage")
    if input_group_size is None:
        if input_data.dtype is not torch.int8:
            raise ValueError("fused ConvRot INT8 prepared gate input must have INT8 dtype")
        if (
            input_scale is None
            or input_scale.shape != (batch, sequence_length)
            or input_scale.dtype is not torch.float32
            or input_scale.device != attention_storage.device
            or not input_scale.is_contiguous()
        ):
            raise ValueError("fused ConvRot INT8 gate input scale must match its prepared rows")
    else:
        validate_storage(weight_qdata, weight_scale, input_group_size, input_data.dtype)
        validate_activation_scale(input_scale, input_data.device)
    output_features = heads * head_dim
    if (
        weight_qdata.shape != (output_features, input_data.shape[-1])
        or weight_qdata.dtype is not torch.int8
        or weight_qdata.device != attention_storage.device
        or not weight_qdata.is_contiguous()
    ):
        raise ValueError("fused ConvRot INT8 gate weight must produce one D64/D128 vector per head")
    if (
        weight_scale.shape != (output_features, 1)
        or weight_scale.dtype is not torch.float32
        or weight_scale.device != attention_storage.device
        or not weight_scale.is_contiguous()
    ):
        raise ValueError("fused ConvRot INT8 gate weight scale must contain one FP32 value per row")
    if bias is not None and (
        bias.shape != (output_features,)
        or bias.device != attention_storage.device
        or not bias.is_contiguous()
    ):
        raise ValueError("fused ConvRot INT8 gate bias must be contiguous per output feature")
    if bias is not None:
        _bias.validate_dtype(bias, "fused ConvRot INT8 gate")
    if torch.is_grad_enabled() and (
        input_data.requires_grad
        or (input_scale is not None and input_scale.requires_grad)
        or weight_scale.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError("fused ConvRot INT8 gate projection is inference-only")
    return _GateProjection(
        input_data,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        linear_backend.require_linear_backend(input_data),
        input_group_size,
    )


def _prepare_optional_gate_projection(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    input_data: torch.Tensor | None,
    input_scale: torch.Tensor | None,
    weight_qdata: torch.Tensor | None,
    weight_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    input_group_size: int | None = None,
) -> _GateProjection | None:
    """Resolve an absent or complete ConvRot INT8 gate operand set."""
    required = input_data, weight_qdata, weight_scale
    if not any(operand is not None for operand in (*required, input_scale, bias)):
        return None
    if any(operand is None for operand in required):
        raise ValueError("projected ConvRot INT8 gate requires input and weight storage")
    assert input_data is not None
    assert weight_qdata is not None
    assert weight_scale is not None
    return _prepare_gate_projection(
        attention_storage,
        logical_sequence_length,
        block_lengths,
        input_data,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        input_group_size,
    )


def _validate_output_projection(
    attention_storage: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    logical_sequence_length: int,
    query_chunk_rows: int,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[int, int]:
    """Validate the projection boundary and return input and output widths."""
    input_features = output_common.validate_attention_output(
        attention_storage,
        logical_sequence_length,
        query_chunk_rows,
    )

    validate_storage(
        weight_qdata,
        weight_scale,
        group_size,
        output_dtype,
    )
    output_features = weight_qdata.shape[0]
    if weight_qdata.shape[1] != input_features or output_features < 1:
        raise ValueError(
            "fused sparse Piper output projection weight must consume all attention heads"
        )
    if weight_qdata.device != attention_storage.device:
        raise ValueError("fused sparse Piper attention and projection must share a device")
    if bias is not None and (
        bias.shape != (output_features,)
        or bias.device != attention_storage.device
        or bias.layout is not torch.strided
        or not bias.is_contiguous()
    ):
        raise ValueError(
            "fused sparse Piper output bias must be contiguous with one value per output"
        )
    if bias is not None:
        _bias.validate_dtype(bias, "fused sparse Piper output")
    if torch.is_grad_enabled() and (
        weight_scale.requires_grad or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError(
            "fused sparse Piper output is inference-only and does not support autograd"
        )
    return input_features, output_features


def _project_attention_chunk(  # noqa: PLR0913, PLR0917
    attention_chunk: torch.Tensor,
    output: torch.Tensor,
    start: int,
    rows: int,
    prepared_input: torch.Tensor,
    prepared_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    backend: LinearBackend,
    output_input_scale: torch.Tensor | None = None,
) -> None:
    """Project one ready attention chunk into its final output rows."""
    batch = attention_chunk.shape[0]
    input_features = weight_qdata.shape[1]
    prepared_input = prepared_input[:rows]
    prepared_scale = prepared_scale[:rows]
    for batch_index in range(batch):
        chunk_input = attention_chunk[batch_index, :rows].reshape(rows, input_features)
        backend.prepare_input(
            chunk_input,
            group_size,
            activation_fn=None,
            input_scale=output_input_scale,
            out=(prepared_input, prepared_scale),
        )
        backend.linear_prepared(
            prepared_input,
            prepared_scale,
            weight_qdata,
            weight_scale,
            bias,
            output.dtype,
            out=output[batch_index, start : start + rows],
        )


def _prepare_output_chunk_projector(  # noqa: PLR0913
    attention_storage: torch.Tensor,
    sequence_length: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    logical_sequence_length: int,
    query_chunk_rows: int,
    *,
    backend: LinearBackend,
    output_dtype: torch.dtype = torch.bfloat16,
    output_input_scale: torch.Tensor | None = None,
) -> tuple[int, output_common.ChunkProjector, tuple[torch.Tensor, torch.Tensor]]:
    """Prepare output-chunk buffers using the fusion's already-selected backend."""
    validate_activation_scale(output_input_scale, attention_storage.device)
    input_features, output_features = _validate_output_projection(
        attention_storage,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
        output_dtype=output_dtype,
    )
    capacity = min(sequence_length, query_chunk_rows)
    prepared_input = torch.empty(
        (capacity, input_features),
        device=attention_storage.device,
        dtype=torch.int8,
    )
    prepared_scale = torch.empty(
        capacity,
        device=attention_storage.device,
        dtype=torch.float32,
    )

    def project_chunk(
        attention_chunk: torch.Tensor,
        output: torch.Tensor,
        start: int,
        rows: int,
    ) -> None:
        _project_attention_chunk(
            attention_chunk,
            output,
            start,
            rows,
            prepared_input,
            prepared_scale,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            backend,
            output_input_scale,
        )

    return output_features, project_chunk, (prepared_input, prepared_scale)


def _run_attention_output(  # noqa: PLR0913, PLR0917
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
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_projection: _GateProjection | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    output_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pipeline bounded attention chunks into the final ConvRot INT8 output."""
    backend = fusion_backend.require_output_backend(query)
    prepared = output_common.prepare_attention(
        query,
        query_scale,
        query_summary,
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
        has_projected_coarse_gate=gate_projection is not None,
    )
    output_features, project_chunk, projector_tensors = _prepare_output_chunk_projector(
        query,
        prepared.sequence_length,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
        backend=backend,
        output_dtype=output_dtype,
        output_input_scale=output_input_scale,
    )
    return output_common.run_chunked_attention_output(
        prepared,
        output_features,
        query_chunk_rows,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=(None if gate_projection is None else gate_projection.project),
        output_dtype=output_dtype,
    )


def _run_projected_query_attention_output(  # noqa: PLR0913, PLR0917
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
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_projection: _GateProjection | None = None,
    *,
    input_group_size: int | None = None,
    out: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    query_bias: torch.Tensor | None = None,
    output_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lifetime-chunk Q through routing, attention, and ConvRot INT8 output.

    With ``input_group_size``, Q/gate inputs are floating-point sources and their
    scales are optional calibrated scalars. Otherwise they are prepared INT8
    tensors with FP32 row scales. Preparation workspaces belong to each stream.
    """
    projection_backend = fusion_backend.require_projection_backend(
        query_input, head_dim=key.shape[-1]
    )
    backend = fusion_backend.require_output_backend(key)
    prepared = output_common.prepare_attention_context(
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
        has_projected_coarse_gate=gate_projection is not None,
    )
    if (
        query_input.ndim != 3
        or query_input.shape[:2] != (key.shape[0], prepared.sequence_length)
        or query_input.shape[-1] != query_weight_qdata.shape[-1]
        or query_input.device != key.device
    ):
        raise ValueError(
            "fused ConvRot INT8 Q input must match the global attention rows and width"
        )
    if input_group_size is not None:
        validate_storage(
            query_weight_qdata, query_weight_scale, input_group_size, query_input.dtype
        )
        validate_activation_scale(query_input_scale, query_input.device)
    elif query_input_scale is None:
        raise ValueError("fused ConvRot INT8 prepared Q input requires row scales")
    output_features, project_chunk, projector_tensors = _prepare_output_chunk_projector(
        key,
        prepared.sequence_length,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
        backend=backend,
        output_dtype=output_dtype,
        output_input_scale=output_input_scale,
    )

    def project_query_chunk(
        start: int,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_group_size is None:
            assert query_input_scale is not None
            chunk_input, chunk_scale = query_input, query_input_scale
            chunk_cos, chunk_sin = cos, sin
            chunk_lengths = block_lengths
            chunk_start = start
        else:
            chunk_input, chunk_scale = backend.prepare_input(
                query_input[:, start : start + rows],
                input_group_size,
                input_scale=query_input_scale,
            )
            chunk_cos, chunk_sin = cos[start : start + rows], sin[start : start + rows]
            chunk_lengths = (
                None
                if block_lengths is None
                else block_lengths[start // TILE_ROWS : (start + rows + TILE_ROWS - 1) // TILE_ROWS]
            )
            chunk_start = 0
        return query_projection._launch_query_projection_range(
            chunk_input,
            chunk_scale,
            query_weight_qdata,
            query_weight_scale,
            query_norm_weight,
            chunk_cos,
            chunk_sin,
            query_norm_epsilon,
            softmax_scale,
            routing_mode,
            chunk_lengths,
            chunk_start=chunk_start,
            chunk_rows=rows,
            backend=projection_backend,
            head_dim=key.shape[-1],
            bias=query_bias,
        )

    return output_common.run_chunked_projected_query_attention_output(
        prepared,
        output_features,
        query_chunk_rows,
        project_query_chunk,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=(None if gate_projection is None else gate_projection.project),
        output_dtype=output_dtype,
        out=out,
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_projected_query_attention_output",
    mutates_args=(),
)
def _projected_query_attention_output_op(  # noqa: PLR0913, PLR0917
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
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
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
    *,
    input_group_size: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    gate_projection = _prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        gate_input,
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        input_group_size,
    )
    return _run_projected_query_attention_output(
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
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
        routing_mode,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        query_chunk_rows,
        block_lengths,
        block_mean,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
        output_dtype=output_dtype,
        output_input_scale=output_input_scale,
        query_bias=query_bias,
        input_group_size=input_group_size,
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_projected_query_attention_output_",
    mutates_args=("query_input", "gate_input"),
)
def _projected_query_attention_output_inplace_op(  # noqa: PLR0913, PLR0917
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
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
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
    *,
    input_group_size: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Overwrite an exclusive floating-point input after each window is consumed.

    Compiler ownership checks exclude caller inputs, aliases, and escaping values.
    Q and gate finish reading a window before its output projection can overwrite it.
    """
    if input_group_size is None:
        raise ValueError("input reuse requires chunked floating-point input preparation")
    gate_projection = _prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        gate_input,
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        input_group_size,
    )
    _run_projected_query_attention_output(
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
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
        routing_mode,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        query_chunk_rows,
        block_lengths,
        block_mean,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
        output_dtype=output_dtype,
        output_input_scale=output_input_scale,
        query_bias=query_bias,
        input_group_size=input_group_size,
        out=query_input,
    )


@_projected_query_attention_output_inplace_op.register_fake
def _projected_query_attention_output_inplace_op_fake(*_args, **_kwargs) -> None:
    return None


@_projected_query_attention_output_op.register_fake
def _projected_query_attention_output_op_fake(
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
    _key_summary: torch.Tensor,
    _key_aux: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _routing_mode: int,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    _query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    _block_mean: torch.Tensor | None = None,
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
    *,
    input_group_size: int | None = None,  # noqa: ARG001 - custom-op keyword
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return output_common.new_projected_output(
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
        output_dtype=output_dtype,
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_attention_output",
    mutates_args=(),
)
def _attention_output_op(  # noqa: PLR0913, PLR0917
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
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    output_input_scale: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    gate_projection = _prepare_optional_gate_projection(
        query,
        logical_sequence_length,
        block_lengths,
        gate_input,
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
    )
    return _run_attention_output(
        query,
        query_scale,
        query_summary,
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
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        query_chunk_rows,
        block_lengths,
        block_mean,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
        output_dtype=output_dtype,
        output_input_scale=output_input_scale,
    )


@_attention_output_op.register_fake
def _attention_output_op_fake(
    query: torch.Tensor,
    _query_scale: torch.Tensor,
    _query_summary: torch.Tensor,
    _key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_summary: torch.Tensor,
    _key_aux: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _routing_mode: int,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    _query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    _block_mean: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
    _output_input_scale: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return output_common.new_projected_output(
        query,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
        output_dtype=output_dtype,
    )


__all__: list[str] = []
