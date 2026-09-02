"""Bounded-workspace sparse Piper attention followed by a ConvRot output projection."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.sparse_piper import _output as output_common
from piper_kernels.linear.convrot.int8 import _policy, reference
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

_DEFAULT_QUERY_CHUNK_ROWS = output_common.DEFAULT_QUERY_CHUNK_ROWS


def _validate_output_projection(
    query: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    logical_sequence_length: int,
    query_chunk_rows: int,
) -> tuple[int, int]:
    """Validate the projection boundary and return input and output widths."""
    input_features = output_common.validate_attention_output(
        query,
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
    if weight_qdata.device != query.device:
        raise ValueError("fused sparse Piper attention and projection must share a CUDA device")
    if bias is not None and (
        bias.shape != (output_features,)
        or bias.dtype is not torch.bfloat16
        or bias.device != query.device
        or bias.layout is not torch.strided
        or not bias.is_contiguous()
    ):
        raise ValueError(
            "fused sparse Piper output bias must be contiguous BF16 with one value per output"
        )
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
) -> torch.Tensor:
    """Pipeline bounded attention chunks into the final ConvRot output."""
    input_features, output_features = _validate_output_projection(
        query,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
    )
    prepared_attention = output_common.prepare_attention(
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
    )
    capacity = min(logical_sequence_length, query_chunk_rows)
    prepared_input = torch.empty(
        (capacity, input_features),
        device=query.device,
        dtype=torch.int8,
    )
    prepared_scale = torch.empty(capacity, device=query.device, dtype=torch.float32)
    target = AcceleratorTarget.from_device(query.device)
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

    return output_common.run_chunked_attention_output(
        prepared_attention,
        query,
        logical_sequence_length,
        output_features,
        query_chunk_rows,
        project_chunk,
        (prepared_input, prepared_scale),
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
) -> torch.Tensor:
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
) -> torch.Tensor:
    return query.new_empty(
        (query.shape[0], logical_sequence_length, weight_qdata.shape[0]),
        dtype=torch.bfloat16,
    )


__all__: list[str] = []
