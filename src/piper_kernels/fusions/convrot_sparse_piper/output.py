"""Bounded-workspace sparse Piper attention followed by a ConvRot output projection."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch
from piper_kernels.linear.convrot.int8 import _policy, reference
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

from . import _layout

_DEFAULT_QUERY_CHUNK_ROWS = 4_096


def _validate_output_projection(
    query: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    logical_sequence_length: int,
    query_chunk_rows: int,
) -> tuple[int, int, int]:
    """Validate the projection boundary and return batch, input, and output widths."""
    if query.ndim != 4:
        raise ValueError("fused sparse Piper output requires four-dimensional quantized queries")
    batch, heads, _storage_sequence_length, head_dim = query.shape
    if batch < 1 or head_dim != _layout.HEAD_DIM:
        raise ValueError("fused sparse Piper output requires nonempty batches with D128 heads")
    if isinstance(logical_sequence_length, bool) or not isinstance(logical_sequence_length, int):
        raise TypeError("fused sparse Piper logical sequence length must be an integer")
    if (
        isinstance(query_chunk_rows, bool)
        or not isinstance(query_chunk_rows, int)
        or query_chunk_rows < _layout.TILE_ROWS
        or query_chunk_rows % _layout.TILE_ROWS
    ):
        raise ValueError("fused sparse Piper query chunk rows must be a positive multiple of 64")
    if query.device.type != "cuda":
        raise ValueError("fused sparse Piper output currently requires CUDA")
    target = AcceleratorTarget.from_device(query.device)
    if not target.is_cuda_capability(12, 0):
        raise ValueError("fused sparse Piper output requires exact NVIDIA SM120")

    reference.validate_storage(
        weight_qdata,
        weight_scale,
        group_size,
        torch.bfloat16,
    )
    input_features = heads * head_dim
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
    return batch, input_features, output_features


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
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int,
) -> torch.Tensor:
    """Pipeline bounded attention chunks into the final ConvRot output."""
    batch, input_features, output_features = _validate_output_projection(
        query,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
    )
    prepared_attention = _quantized_dispatch._prepare_quantized_sparse_piper_attention(
        query,
        query_scale,
        query_summary,
        key,
        key_scale,
        key_max,
        key_min,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
    )

    chunk_blocks = query_chunk_rows // _layout.TILE_ROWS
    total_blocks = (logical_sequence_length + _layout.TILE_ROWS - 1) // _layout.TILE_ROWS
    chunk_count = (total_blocks + chunk_blocks - 1) // chunk_blocks
    capacity = min(logical_sequence_length, query_chunk_rows)
    heads, head_dim = query.shape[1], query.shape[3]
    attention_buffers = torch.empty(
        (min(2, chunk_count), batch, capacity, heads, head_dim),
        device=query.device,
        dtype=torch.bfloat16,
    )
    prepared_input = torch.empty(
        (capacity, input_features),
        device=query.device,
        dtype=torch.int8,
    )
    prepared_scale = torch.empty(capacity, device=query.device, dtype=torch.float32)
    output = torch.empty(
        (batch, logical_sequence_length, output_features),
        device=query.device,
        dtype=torch.bfloat16,
    )
    target = AcceleratorTarget.from_device(query.device)
    execution_plan = convrot_backend.default_execution_plan(weight_qdata)

    if chunk_count == 1:
        attention_chunk = attention_buffers[0]
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            prepared_attention,
            attention_chunk.transpose(1, 2),
        )
        _project_attention_chunk(
            attention_chunk,
            output,
            0,
            logical_sequence_length,
            prepared_input,
            prepared_scale,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            execution_plan,
            target,
        )
        return output

    with torch.cuda.device(query.device):
        producer = torch.cuda.current_stream(query.device)
        consumer = torch.cuda.Stream(device=query.device)
        ready = (torch.cuda.Event(), torch.cuda.Event())
        reusable = (torch.cuda.Event(), torch.cuda.Event())
        last_slot = 0
        for chunk_index, block_start in enumerate(range(0, total_blocks, chunk_blocks)):
            slot = chunk_index % 2
            last_slot = slot
            if chunk_index >= 2:
                producer.wait_event(reusable[slot])
            block_count = min(chunk_blocks, total_blocks - block_start)
            start = block_start * _layout.TILE_ROWS
            rows = min(block_count * _layout.TILE_ROWS, logical_sequence_length - start)
            attention_chunk = attention_buffers[slot, :, :rows]
            _quantized_dispatch._launch_quantized_sparse_piper_attention(
                prepared_attention,
                attention_chunk.transpose(1, 2),
                query_block_offset=block_start,
                query_block_count=block_count,
            )
            ready[slot].record(producer)
            with torch.cuda.stream(consumer):
                consumer.wait_event(ready[slot])
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
                reusable[slot].record(consumer)
        producer.wait_event(reusable[last_slot])
        attention_buffers.record_stream(consumer)
        prepared_input.record_stream(consumer)
        prepared_scale.record_stream(consumer)
        output.record_stream(consumer)
    return output


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
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
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
        key_max,
        key_min,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
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
    _key_max: torch.Tensor,
    _key_min: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
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
