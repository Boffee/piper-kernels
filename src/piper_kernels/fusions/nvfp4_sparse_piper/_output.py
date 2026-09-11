"""Shared bounded execution for sparse attention followed by an NVFP4 projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch

from piper_kernels.fusions.sparse_piper import _output as output_common
from piper_kernels.linear.nvfp4 import _projection, _validation
from piper_kernels.weights.nvfp4 import _layout

from . import query as query_projection

DEFAULT_QUERY_CHUNK_ROWS = 8_192
DYNAMIC_QUERY_CHUNK_ROWS = 32_768


def default_query_chunk_rows(dynamic_activation_scale: bool) -> int:
    """Use larger attention windows when global scaling prevents overlap."""
    return DYNAMIC_QUERY_CHUNK_ROWS if dynamic_activation_scale else DEFAULT_QUERY_CHUNK_ROWS


class PreparationBackend(Protocol):
    """Format-specific global scaling and attention-chunk packing."""

    def dynamic_scale(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Compute one scale across all attention rows in the projection basis."""
        ...

    def prepare_static_out(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare an attention chunk into reusable standard NVFP4 storage."""
        ...


@dataclass(frozen=True, slots=True)
class PreparedGateProjection:
    """One NVFP4 gate projected from shared prepared hidden states."""

    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    global_scale: torch.Tensor
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def project(self, output: torch.Tensor, start: int, rows: int) -> None:
        """Project one sequence window into caller-owned token-major gate storage."""
        output_chunk = output[0, :rows].reshape(rows, self.weight_qdata.shape[0])
        _projection.matmul_prepared_chunk_affine_out(
            self.input_qdata,
            self.input_scale,
            self.global_scale,
            self.weight_qdata,
            self.weight_scale,
            None,
            self.bias,
            start,
            start + rows,
            output_chunk,
        )


def prepare_gate_projection(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> PreparedGateProjection:
    """Validate a prepared NVFP4 gate that produces one D64/D128 vector per head."""
    if attention_storage.shape[0] != 1:
        raise ValueError("fused NVFP4 gate projection currently requires batch size one")
    shape = _validation.validate_prepared_linear(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        torch.bfloat16,
        "fused sparse Piper NVFP4 gate",
    )
    sequence_length = output_common.output_sequence_length(
        attention_storage,
        logical_sequence_length,
        block_lengths,
    )
    if shape.rows != sequence_length:
        raise ValueError("fused NVFP4 gate input must match the attention output rows")
    if shape.output_features != attention_storage.shape[1] * attention_storage.shape[3]:
        raise ValueError("fused NVFP4 gate must produce one D64/D128 vector per head")
    differentiable_tensors = (
        input_scale,
        input_per_tensor_scale,
        weight_scale,
        *(tensor for tensor in (weight_per_tensor_scale, bias) if tensor is not None),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in differentiable_tensors):
        raise RuntimeError("fused NVFP4 gate projection is inference-only")
    return PreparedGateProjection(
        input_qdata,
        input_scale,
        (
            input_per_tensor_scale
            if weight_per_tensor_scale is None
            else input_per_tensor_scale * weight_per_tensor_scale
        ),
        weight_qdata,
        weight_scale,
        bias,
    )


def prepare_optional_gate_projection(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    input_qdata: torch.Tensor | None,
    input_scale: torch.Tensor | None,
    input_per_tensor_scale: torch.Tensor | None,
    weight_qdata: torch.Tensor | None,
    weight_scale: torch.Tensor | None,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> PreparedGateProjection | None:
    """Resolve an absent or complete prepared NVFP4 gate operand set."""
    required = input_qdata, input_scale, input_per_tensor_scale, weight_qdata, weight_scale
    if not any(value is not None for value in (*required, weight_per_tensor_scale, bias)):
        return None
    if any(value is None for value in required):
        raise ValueError("projected NVFP4 gate requires every prepared storage operand")
    assert input_qdata is not None
    assert input_scale is not None
    assert input_per_tensor_scale is not None
    assert weight_qdata is not None
    assert weight_scale is not None
    return prepare_gate_projection(
        attention_storage,
        logical_sequence_length,
        block_lengths,
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
    )


def _validate_output_projection(
    attention_storage: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    logical_sequence_length: int,
    query_chunk_rows: int,
    dynamic_activation_scale: bool,
) -> tuple[int, int]:
    """Validate the projection boundary and return its logical dimensions."""
    # The shared pipeline validates shapes, not NVFP4's accelerator requirements.
    _validation._validate_device(attention_storage.device, "fused sparse Piper NVFP4 output")
    if (
        isinstance(query_chunk_rows, bool)
        or not isinstance(query_chunk_rows, int)
        or query_chunk_rows < 128
        or query_chunk_rows % 128
    ):
        raise ValueError("fused NVFP4 output chunk rows must be a positive multiple of 128")
    input_features = output_common.validate_attention_output(
        attention_storage,
        logical_sequence_length,
        query_chunk_rows,
    )
    _validation.validate_activation_scale(
        activation_per_tensor_scale,
        dynamic_activation_scale,
        attention_storage.device,
        "fused sparse Piper NVFP4 output",
    )
    output_features = _validation.validate_weight(
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        input_features=input_features,
        device=attention_storage.device,
        name="fused sparse Piper NVFP4 output",
    )
    differentiable_tensors = (
        weight_scale,
        *(
            tensor
            for tensor in (activation_per_tensor_scale, weight_per_tensor_scale, bias)
            if tensor is not None
        ),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in differentiable_tensors):
        raise RuntimeError(
            "fused sparse Piper NVFP4 output is inference-only and does not support autograd"
        )
    return input_features, cast(int, output_features)


def _prepare_output_projector(  # noqa: PLR0913, PLR0917
    attention_storage: torch.Tensor,
    sequence_length: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    logical_sequence_length: int,
    query_chunk_rows: int,
    preparation: PreparationBackend,
    dynamic_activation_scale: bool,
) -> tuple[
    int,
    output_common.ChunkProjector | None,
    output_common.AttentionProjector | None,
    tuple[torch.Tensor, ...],
]:
    """Build static chunk or global dynamic projection with reusable packing storage."""
    input_features, output_features = _validate_output_projection(
        attention_storage,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        logical_sequence_length,
        query_chunk_rows,
        dynamic_activation_scale,
    )
    capacity = min(sequence_length, query_chunk_rows)
    device = attention_storage.device

    def allocate_storage() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(
                _layout.qdata_shape(capacity, input_features), device=device, dtype=torch.uint8
            ),
            torch.empty(
                _layout.scale_shape(capacity, input_features),
                device=device,
                dtype=torch.float8_e4m3fn,
            ),
        )

    def project_rows(
        attention_rows: torch.Tensor,
        output_rows: torch.Tensor,
        scale: torch.Tensor,
        storage: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        input_qdata, input_scale = preparation.prepare_static_out(attention_rows, scale, storage)
        _projection.matmul_prepared_chunk_affine_out(
            input_qdata,
            input_scale,
            scale,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            bias,
            0,
            attention_rows.shape[0],
            output_rows,
        )

    if dynamic_activation_scale:

        def project_attention(attention: torch.Tensor) -> torch.Tensor:
            batch, rows = attention.shape[:2]
            flat_attention = attention.view(batch * rows, input_features)
            scale = preparation.dynamic_scale(flat_attention)
            # The global path needs packing storage only after attention finishes.
            storage = allocate_storage()
            if output_features <= input_features:
                # Pack each input chunk before overwriting it. Flattening batches
                # preserves this ordering even when the projection narrows rows.
                output = attention.view(-1)[: batch * rows * output_features].view(
                    batch * rows, output_features
                )
            else:
                output = attention.new_empty((batch * rows, output_features))
            for start in range(0, batch * rows, capacity):
                stop = min(start + capacity, batch * rows)
                project_rows(flat_attention[start:stop], output[start:stop], scale, storage)
            return output.view(batch, rows, output_features)

        return output_features, None, project_attention, ()

    assert activation_per_tensor_scale is not None
    storage = allocate_storage()

    def project_chunk(
        attention_chunk: torch.Tensor,
        output: torch.Tensor,
        start: int,
        rows: int,
    ) -> None:
        for batch_index in range(attention_chunk.shape[0]):
            project_rows(
                attention_chunk[batch_index, :rows].reshape(rows, input_features),
                output[batch_index, start : start + rows],
                activation_per_tensor_scale,
                storage,
            )

    return output_features, project_chunk, None, storage


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
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    query_chunk_rows: int | None,
    preparation: PreparationBackend,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_projection: PreparedGateProjection | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    dynamic_activation_scale: bool = False,
) -> torch.Tensor:
    """Project bounded attention with a calibrated or global dynamic scale."""
    if query_chunk_rows is None:
        query_chunk_rows = default_query_chunk_rows(dynamic_activation_scale)
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
    output_features, project_chunk, project_attention, projector_tensors = (
        _prepare_output_projector(
            query,
            prepared.sequence_length,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            activation_per_tensor_scale,
            bias,
            logical_sequence_length,
            query_chunk_rows,
            preparation,
            dynamic_activation_scale,
        )
    )
    return output_common.run_chunked_attention_output(
        prepared,
        output_features,
        query_chunk_rows,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=(None if gate_projection is None else gate_projection.project),
        output_dtype=output_dtype,
        project_attention=project_attention,
    )


def run_projected_query_attention_output(  # noqa: PLR0913, PLR0917
    query_input_qdata: torch.Tensor,
    query_input_scale: torch.Tensor,
    query_input_per_tensor_scale: torch.Tensor,
    query_weight_qdata: torch.Tensor,
    query_weight_scale: torch.Tensor,
    query_weight_per_tensor_scale: torch.Tensor | None,
    query_bias: torch.Tensor | None,
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
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    query_chunk_rows: int | None,
    preparation: PreparationBackend,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_projection: PreparedGateProjection | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    dynamic_activation_scale: bool = False,
) -> torch.Tensor:
    """Lifetime-chunk NVFP4 Q through routing, attention, and output."""
    if query_chunk_rows is None:
        query_chunk_rows = default_query_chunk_rows(dynamic_activation_scale)
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
    if query_input_qdata.shape[0] != prepared.sequence_length:
        raise ValueError("fused NVFP4 Q input must match the global attention rows")
    output_features, project_chunk, project_attention, projector_tensors = (
        _prepare_output_projector(
            key,
            prepared.sequence_length,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            activation_per_tensor_scale,
            bias,
            logical_sequence_length,
            query_chunk_rows,
            preparation,
            dynamic_activation_scale,
        )
    )

    def project_query_chunk(
        start: int,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return query_projection._launch_query_range(
            query_input_qdata,
            query_input_scale,
            query_input_per_tensor_scale,
            query_weight_qdata,
            query_weight_scale,
            query_weight_per_tensor_scale,
            query_bias,
            query_norm_weight,
            cos,
            sin,
            query_norm_epsilon,
            softmax_scale,
            query_projection.DEFAULT_CHUNK_ROWS,
            routing_mode,
            block_lengths,
            chunk_start=start,
            chunk_rows=rows,
            head_dim=key.shape[-1],
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
        project_attention=project_attention,
    )


__all__ = [
    "DEFAULT_QUERY_CHUNK_ROWS",
    "PreparationBackend",
    "PreparedGateProjection",
    "prepare_gate_projection",
    "prepare_optional_gate_projection",
    "run_attention_output",
    "run_projected_query_attention_output",
]
