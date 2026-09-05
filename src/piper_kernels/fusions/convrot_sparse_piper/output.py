"""Bounded-workspace sparse Piper attention followed by a ConvRot output projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.sparse_piper import _output as output_common
from piper_kernels.linear import _bias
from piper_kernels.linear.convrot.int8 import _policy, reference
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

from . import query as query_projection

_DEFAULT_QUERY_CHUNK_ROWS = output_common.DEFAULT_QUERY_CHUNK_ROWS


@dataclass(frozen=True, slots=True)
class _PreparedGateProjection:
    """One ConvRot INT8 gate projected from shared prepared hidden states."""

    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None
    execution_plan: _policy.LinearExecutionPlan

    def project(self, output: torch.Tensor, start: int, rows: int) -> None:
        """Project one sequence window into caller-owned token-major gate storage."""
        output_features = self.weight_qdata.shape[0]
        for batch_index in range(self.input_qdata.shape[0]):
            convrot_backend._execute_prepared_linear(
                self.input_qdata[batch_index, start : start + rows],
                self.input_scale[batch_index, start : start + rows],
                self.weight_qdata,
                self.weight_scale,
                self.bias,
                torch.bfloat16,
                self.execution_plan,
                out=output[batch_index, :rows].reshape(rows, output_features),
            )


def _prepare_gate_projection(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> _PreparedGateProjection:
    """Validate shared prepared input and one D128-per-head gate weight."""
    sequence_length = output_common.output_sequence_length(
        attention_storage,
        logical_sequence_length,
        block_lengths,
    )
    batch, heads, _storage_sequence_length, head_dim = attention_storage.shape
    if (
        input_qdata.ndim != 3
        or input_qdata.shape[:2] != (batch, sequence_length)
        or input_qdata.dtype is not torch.int8
        or input_qdata.device != attention_storage.device
        or not input_qdata.is_contiguous()
    ):
        raise ValueError("fused ConvRot gate input must be contiguous batch/sequence INT8 storage")
    if (
        input_scale.shape != (batch, sequence_length)
        or input_scale.dtype is not torch.float32
        or input_scale.device != attention_storage.device
        or not input_scale.is_contiguous()
    ):
        raise ValueError("fused ConvRot gate input scale must match its prepared rows")
    output_features = heads * head_dim
    if (
        weight_qdata.shape != (output_features, input_qdata.shape[-1])
        or weight_qdata.dtype is not torch.int8
        or weight_qdata.device != attention_storage.device
        or not weight_qdata.is_contiguous()
    ):
        raise ValueError("fused ConvRot gate weight must produce one D128 vector per head")
    if (
        weight_scale.shape != (output_features, 1)
        or weight_scale.dtype is not torch.float32
        or weight_scale.device != attention_storage.device
        or not weight_scale.is_contiguous()
    ):
        raise ValueError("fused ConvRot gate weight scale must contain one FP32 value per row")
    if bias is not None and (
        bias.shape != (output_features,)
        or bias.device != attention_storage.device
        or not bias.is_contiguous()
    ):
        raise ValueError("fused ConvRot gate bias must be contiguous per output feature")
    if bias is not None:
        _bias.validate_dtype(bias, "fused ConvRot gate")
    if torch.is_grad_enabled() and (
        input_scale.requires_grad
        or weight_scale.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError("fused ConvRot gate projection is inference-only")
    return _PreparedGateProjection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        convrot_backend.default_execution_plan(weight_qdata),
    )


def _prepare_optional_gate_projection(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    input_qdata: torch.Tensor | None,
    input_scale: torch.Tensor | None,
    weight_qdata: torch.Tensor | None,
    weight_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> _PreparedGateProjection | None:
    """Resolve an absent or complete prepared ConvRot gate operand set."""
    required = input_qdata, input_scale, weight_qdata, weight_scale
    if not any(operand is not None for operand in (*required, bias)):
        return None
    if any(operand is None for operand in required):
        raise ValueError("projected ConvRot gate requires every prepared storage operand")
    assert input_qdata is not None
    assert input_scale is not None
    assert weight_qdata is not None
    assert weight_scale is not None
    return _prepare_gate_projection(
        attention_storage,
        logical_sequence_length,
        block_lengths,
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
    )


def _validate_output_projection(
    attention_storage: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    logical_sequence_length: int,
    query_chunk_rows: int,
) -> tuple[int, int]:
    """Validate the projection boundary and return input and output widths."""
    input_features = output_common.validate_attention_output(
        attention_storage,
        logical_sequence_length,
        query_chunk_rows,
    )

    reference.validate_storage(
        weight_qdata,
        weight_scale,
        group_size,
        torch.bfloat16,
    )
    output_features = weight_qdata.shape[0]
    if weight_qdata.shape[1] != input_features or output_features < 1:
        raise ValueError(
            "fused sparse Piper output projection weight must consume all attention heads"
        )
    if weight_qdata.device != attention_storage.device:
        raise ValueError("fused sparse Piper attention and projection must share a CUDA device")
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
    execution_plan: _policy.LinearExecutionPlan,
    target: AcceleratorTarget,
) -> None:
    """Project one ready attention chunk into its final output rows."""
    batch = attention_chunk.shape[0]
    input_features = weight_qdata.shape[1]
    for batch_index in range(batch):
        chunk_input = attention_chunk[batch_index, :rows].reshape(rows, input_features)
        convrot_backend._prepare_input(
            chunk_input,
            input_features,
            group_size,
            activation_fn=None,
            execution_plan=execution_plan,
            target=target,
            out=(prepared_input[:rows], prepared_scale[:rows]),
        )
        convrot_backend._execute_prepared_linear(
            prepared_input[:rows],
            prepared_scale[:rows],
            weight_qdata,
            weight_scale,
            bias,
            torch.bfloat16,
            execution_plan,
            out=output[batch_index, start : start + rows],
        )


def _prepare_output_chunk_projector(
    attention_storage: torch.Tensor,
    sequence_length: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    logical_sequence_length: int,
    query_chunk_rows: int,
) -> tuple[int, output_common.ChunkProjector, tuple[torch.Tensor, torch.Tensor]]:
    """Prepare one reusable ConvRot output-projection chunk boundary."""
    input_features, output_features = _validate_output_projection(
        attention_storage,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
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
    target = AcceleratorTarget.from_device(attention_storage.device)
    execution_plan = convrot_backend.default_execution_plan(weight_qdata)

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
            execution_plan,
            target,
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
    gate_projection: _PreparedGateProjection | None = None,
) -> torch.Tensor:
    """Pipeline bounded attention chunks into the final ConvRot output."""
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
    )
    return output_common.run_chunked_attention_output(
        prepared,
        output_features,
        query_chunk_rows,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=(None if gate_projection is None else gate_projection.project),
    )


def _run_projected_query_attention_output(  # noqa: PLR0913, PLR0917
    query_input_qdata: torch.Tensor,
    query_input_scale: torch.Tensor,
    query_weight_qdata: torch.Tensor,
    query_weight_scale: torch.Tensor,
    query_norm_weight: torch.Tensor,
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
    gate_projection: _PreparedGateProjection | None = None,
) -> torch.Tensor:
    """Lifetime-chunk Q through routing, attention, and ConvRot output."""
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
    if query_input_qdata.shape[:2] != (key.shape[0], prepared.sequence_length):
        raise ValueError("fused ConvRot Q input must match the global attention rows")
    output_features, project_chunk, projector_tensors = _prepare_output_chunk_projector(
        key,
        prepared.sequence_length,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
    )

    def project_query_chunk(
        start: int,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return query_projection._launch_query_projection_range(
            query_input_qdata,
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
            chunk_start=start,
            chunk_rows=rows,
        )

    return output_common.run_chunked_projected_query_attention_output(
        prepared,
        output_features,
        query_chunk_rows,
        project_query_chunk,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=(None if gate_projection is None else gate_projection.project),
    )


@torch.library.custom_op(
    "piper_kernels::convrot_sparse_piper_projected_query_attention_output",
    mutates_args=(),
)
def _projected_query_attention_output_op(  # noqa: PLR0913, PLR0917
    query_input_qdata: torch.Tensor,
    query_input_scale: torch.Tensor,
    query_weight_qdata: torch.Tensor,
    query_weight_scale: torch.Tensor,
    query_norm_weight: torch.Tensor,
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
    gate_input_qdata: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    gate_projection = _prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        gate_input_qdata,
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
    )
    return _run_projected_query_attention_output(
        query_input_qdata,
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
    )


@_projected_query_attention_output_op.register_fake
def _projected_query_attention_output_op_fake(
    _query_input_qdata: torch.Tensor,
    _query_input_scale: torch.Tensor,
    _query_weight_qdata: torch.Tensor,
    _query_weight_scale: torch.Tensor,
    _query_norm_weight: torch.Tensor,
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
    _gate_input_qdata: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return output_common.new_projected_output(
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
    )


@torch.library.custom_op(
    "piper_kernels::convrot_sparse_piper_attention_output",
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
    gate_input_qdata: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    gate_projection = _prepare_optional_gate_projection(
        query,
        logical_sequence_length,
        block_lengths,
        gate_input_qdata,
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
    _gate_input_qdata: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return output_common.new_projected_output(
        query,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
    )


__all__: list[str] = []
