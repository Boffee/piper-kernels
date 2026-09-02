"""Shared bounded execution for sparse attention followed by an NVFP4 projection."""

from __future__ import annotations

from typing import Protocol, cast

import torch

from piper_kernels.fusions.sparse_piper import _output as output_common
from piper_kernels.linear.nvfp4 import _layout, _projection, _validation
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend

DEFAULT_QUERY_CHUNK_ROWS = 8_192


class PreparationBackend(Protocol):
    """Format-specific preparation for one materialized attention chunk."""

    def prepare_static_out(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare an attention chunk into reusable standard NVFP4 storage."""
        ...


def _validate_output_projection(
    query: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    logical_sequence_length: int,
    query_chunk_rows: int,
) -> tuple[int, int]:
    """Validate the static projection boundary and return its logical dimensions."""
    if (
        isinstance(query_chunk_rows, bool)
        or not isinstance(query_chunk_rows, int)
        or query_chunk_rows < 128
        or query_chunk_rows % 128
    ):
        raise ValueError("fused NVFP4 output chunk rows must be a positive multiple of 128")
    input_features = output_common.validate_attention_output(
        query,
        logical_sequence_length,
        query_chunk_rows,
    )
    _validation.validate_activation_scale(
        activation_per_tensor_scale,
        False,
        query.device,
        "fused sparse Piper NVFP4 output",
    )
    output_features = _validation.validate_weight(
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        input_features=input_features,
        logical_dtype=torch.bfloat16,
        device=query.device,
        name="fused sparse Piper NVFP4 output",
    )
    differentiable_tensors = (
        activation_per_tensor_scale,
        weight_scale,
        *(tensor for tensor in (weight_per_tensor_scale, bias) if tensor is not None),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in differentiable_tensors):
        raise RuntimeError(
            "fused sparse Piper NVFP4 output is inference-only and does not support autograd"
        )
    return input_features, cast(int, output_features)


def _project_attention_chunk(  # noqa: PLR0913, PLR0917
    attention_chunk: torch.Tensor,
    output: torch.Tensor,
    start: int,
    rows: int,
    prepared_input: torch.Tensor,
    prepared_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    preparation: PreparationBackend,
) -> None:
    """Prepare and project one ready attention chunk into its final rows."""
    batch = attention_chunk.shape[0]
    input_features = 2 * weight_qdata.shape[1]
    for batch_index in range(batch):
        chunk_input = attention_chunk[batch_index, :rows].reshape(rows, input_features)
        input_qdata, input_scale = preparation.prepare_static_out(
            chunk_input,
            activation_per_tensor_scale,
            (prepared_input, prepared_scale),
        )
        output_chunk = output[batch_index, start : start + rows]
        if weight_per_tensor_scale is not None:
            _projection.matmul_prepared_chunk_affine_out(
                input_qdata,
                input_scale,
                activation_per_tensor_scale,
                weight_qdata,
                weight_scale,
                weight_per_tensor_scale,
                bias,
                0,
                rows,
                output_chunk,
            )
        else:
            projected = _projection.matmul_prepared_chunk_out(
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                0,
                rows,
                output_chunk,
            )
            nvfp4_backend.apply_projection_epilogue(
                projected,
                activation_per_tensor_scale,
                bias,
                projected,
            )


def run_attention_output(  # noqa: PLR0913, PLR0917
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
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    query_chunk_rows: int,
    preparation: PreparationBackend,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    compression_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
) -> torch.Tensor:
    """Pipeline bounded attention chunks into a static NVFP4 output."""
    input_features, output_features = _validate_output_projection(
        query,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        logical_sequence_length,
        query_chunk_rows,
    )
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
        compression_gate,
        coarse_scale,
        coarse_key_blocks,
    )
    sequence_length = prepared.sequence_length
    capacity = min(sequence_length, query_chunk_rows)
    prepared_input = torch.empty(
        _layout.qdata_shape(capacity, input_features),
        device=query.device,
        dtype=torch.uint8,
    )
    prepared_scale = torch.empty(
        _layout.scale_shape(capacity, input_features),
        device=query.device,
        dtype=torch.float8_e4m3fn,
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
            weight_per_tensor_scale,
            activation_per_tensor_scale,
            bias,
            preparation,
        )

    return output_common.run_chunked_attention_output(
        prepared,
        output_features,
        query_chunk_rows,
        project_chunk,
        (prepared_input, prepared_scale),
    )


__all__ = ["DEFAULT_QUERY_CHUNK_ROWS", "PreparationBackend", "run_attention_output"]
