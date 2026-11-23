"""Bounded sparse Piper attention followed by an NVFP4 or ConvRot NVFP4 projection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from piper_kernels._residual import ResidualAdd, validate_residual_add
from piper_kernels.attention.sparse_piper_attention._contracts import _SparsePiperRatioProfile
from piper_kernels.fusions.attention import _sparse_output as output_common
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4_backend
from piper_kernels.linear.nvfp4 import _projection, _validation
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.nvfp4 import _layout
from piper_kernels.weights.nvfp4.triton import dynamic_scale as nvfp4_dynamic_scale

from . import query as query_projection

_DEFAULT_QUERY_CHUNK_ROWS = 8_192
_DYNAMIC_QUERY_CHUNK_ROWS = 32_768


def _default_query_chunk_rows(dynamic_activation_scale: bool) -> int:
    """Use larger attention windows when global scaling prevents overlap."""
    return _DYNAMIC_QUERY_CHUNK_ROWS if dynamic_activation_scale else _DEFAULT_QUERY_CHUNK_ROWS


def _validate_optional_group_size(
    group_size: int | None,
    input_features: int | torch.SymInt,
) -> None:
    """Validate optional ConvRot NVFP4 group size and feature divisibility."""
    if group_size is None:
        return
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError(f"ConvRot group size must be an integer, got {group_size!r}")
    validate_group_size(group_size)
    if isinstance(input_features, int) and input_features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 projection input features {input_features} must be divisible "
            f"by group size {group_size}"
        )


@dataclass(frozen=True, slots=True)
class _PreparedGateProjection:
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


def _prepare_gate_projection(
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
) -> _PreparedGateProjection:
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
    return _PreparedGateProjection(
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


def _prepare_optional_gate_projection(
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
) -> _PreparedGateProjection | None:
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
    return _prepare_gate_projection(
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
    group_size: int | None = None,
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
    _validate_optional_group_size(group_size, input_features)
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


def _prepare_output_projector(  # noqa: PLR0913
    attention_storage: torch.Tensor,
    sequence_length: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    logical_sequence_length: int,
    query_chunk_rows: int,
    *,
    high_first: bool,
    group_size: int | None,
    dynamic_activation_scale: bool,
    residual_add: ResidualAdd | None = None,
) -> tuple[
    int,
    output_common.ChunkProjector | None,
    output_common.AttentionProjector | None,
    tuple[torch.Tensor, ...],
]:
    """Build static chunk or global dynamic projection with reusable packing storage.

    Static None materializes one unit scalar for all chunks. Returned tensors
    retain both packing storage and the static scale on the consumer stream.
    """
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
        group_size,
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
        if group_size is None:
            input_qdata, input_scale = nvfp4_backend.prepare_static_out(
                attention_rows,
                scale,
                storage,
                high_first=high_first,
            )
        else:
            input_qdata, input_scale = convrot_nvfp4_backend.prepare_static_out(
                attention_rows,
                scale,
                group_size,
                storage,
                high_first=high_first,
            )
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

    apply_residual: Callable[[torch.Tensor, ResidualAdd, int], torch.Tensor] | None = None
    if residual_add is not None:
        from piper_kernels._residual.triton import apply_residual_add  # noqa: PLC0415

        apply_residual = apply_residual_add

    def project_residual_rows(
        attention_rows: torch.Tensor,
        scale: torch.Tensor,
        storage: tuple[torch.Tensor, torch.Tensor],
        projected_chunk: torch.Tensor | None,
        row_offset: int,
    ) -> None:
        assert residual_add is not None
        assert apply_residual is not None
        rows = attention_rows.shape[0]
        # Safe to overwrite attention_rows after prepare_static_out packs them into storage;
        # the consumer stream owns attention_chunk until attention_slots.release returns.
        target = (
            attention_rows.view(-1)[: rows * output_features].view(rows, output_features)
            if projected_chunk is None
            else projected_chunk[:rows]
        )
        project_rows(attention_rows, target, scale, storage)
        apply_residual(target, residual_add, row_offset)

    if dynamic_activation_scale:

        def project_attention(attention: torch.Tensor) -> torch.Tensor:
            batch, rows = attention.shape[:2]
            flat_attention = attention.view(batch * rows, input_features)
            scale = (
                nvfp4_dynamic_scale(flat_attention)
                if group_size is None
                else convrot_nvfp4_backend.dynamic_scale(flat_attention, group_size)
            )
            # The global path needs packing storage only after attention finishes.
            storage = allocate_storage()
            if residual_add is not None:
                projected_chunk = (
                    None
                    if output_features <= input_features
                    else attention.new_empty((capacity, output_features))
                )
                for start in range(0, batch * rows, capacity):
                    stop = min(start + capacity, batch * rows)
                    project_residual_rows(
                        flat_attention[start:stop], scale, storage, projected_chunk, start
                    )
                return residual_add.base
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

    # Resolve unit metadata once; every chunk uses the same device scalar.
    static_scale = (
        attention_storage.new_ones((), dtype=torch.float32)
        if activation_per_tensor_scale is None
        else activation_per_tensor_scale
    )
    storage = allocate_storage()
    if residual_add is None:

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
                    static_scale,
                    storage,
                )

        return output_features, project_chunk, None, (*storage, static_scale)

    projected_chunk = (
        None
        if output_features <= input_features
        else torch.empty((capacity, output_features), device=device, dtype=residual_add.base.dtype)
    )

    def project_residual_chunk(
        attention_chunk: torch.Tensor,
        _output: torch.Tensor,
        start: int,
        rows: int,
    ) -> None:
        for batch_index in range(attention_chunk.shape[0]):
            project_residual_rows(
                attention_chunk[batch_index, :rows].reshape(rows, input_features),
                static_scale,
                storage,
                projected_chunk,
                batch_index * sequence_length + start,
            )

    projector_tensors = (
        *storage,
        static_scale,
        residual_add.gate,
        residual_add.gate_indices,
        *(() if projected_chunk is None else (projected_chunk,)),
    )
    return output_features, project_residual_chunk, None, projector_tensors


def _run_attention_output(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
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
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    query_chunk_rows: int | None,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_projection: _PreparedGateProjection | None = None,
    *,
    high_first: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
    dynamic_activation_scale: bool = False,
    group_size: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project bounded attention with plain or ConvRot NVFP4 output preparation."""
    if query_chunk_rows is None:
        query_chunk_rows = _default_query_chunk_rows(dynamic_activation_scale)
    prepared = output_common.prepare_attention(
        query,
        query_scale,
        query_summary,
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
            high_first=high_first,
            group_size=group_size,
            dynamic_activation_scale=dynamic_activation_scale,
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


def _run_projected_query_attention_output(  # noqa: PLR0913, PLR0917
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
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    query_chunk_rows: int | None,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_projection: _PreparedGateProjection | None = None,
    *,
    high_first: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
    dynamic_activation_scale: bool = False,
    group_size: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    residual_add: ResidualAdd | None = None,
) -> torch.Tensor:
    """Lifetime-chunk NVFP4 Q through routing, attention, and output."""
    if query_chunk_rows is None:
        query_chunk_rows = _default_query_chunk_rows(dynamic_activation_scale)
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
    validated_query = query_projection._validate_query_inputs(
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
        head_dim=key.shape[-1],
    )
    if validated_query[0] != prepared.sequence_length:
        raise ValueError("fused NVFP4 Q input must match the global attention rows")
    if validated_query[1] != ratio_profile.num_heads:
        raise ValueError("fused NVFP4 Q heads must match sparse attention heads")
    resolved_output_dtype = output_dtype if residual_add is None else residual_add.base.dtype
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
            high_first=high_first,
            group_size=group_size,
            dynamic_activation_scale=dynamic_activation_scale,
            residual_add=residual_add,
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
            validated=validated_query,
        )

    return output_common.run_chunked_projected_query_attention_output(
        prepared,
        output_features,
        query_chunk_rows,
        project_query_chunk,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=(None if gate_projection is None else gate_projection.project),
        output_dtype=resolved_output_dtype,
        project_attention=project_attention,
        out=None if residual_add is None else residual_add.base,
    )


@torch.library.custom_op(
    "piper_kernels::nvfp4_sparse_piper_projected_query_attention_output",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _projected_query_attention_output_op(  # noqa: PLR0913, PLR0917
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
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    query_chunk_rows: int | None = None,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input_qdata: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_input_per_tensor_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    high_first: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
    dynamic_activation_scale: bool = False,
    group_size: int | None = None,
) -> torch.Tensor:
    gate_projection = _prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        gate_input_qdata,
        gate_input_scale,
        gate_input_per_tensor_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_bias,
    )
    return _run_projected_query_attention_output(
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
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        query_chunk_rows,
        block_lengths,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
        high_first=high_first,
        output_dtype=output_dtype,
        dynamic_activation_scale=dynamic_activation_scale,
        group_size=group_size,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )


@_projected_query_attention_output_op.register_fake
def _projected_query_attention_output_op_fake(
    _query_input_qdata: torch.Tensor,
    _query_input_scale: torch.Tensor,
    _query_input_per_tensor_scale: torch.Tensor,
    _query_weight_qdata: torch.Tensor,
    _query_weight_scale: torch.Tensor,
    _query_weight_per_tensor_scale: torch.Tensor | None,
    _query_bias: torch.Tensor | None,
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
    _weight_per_tensor_scale: torch.Tensor | None,
    _activation_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _query_chunk_rows: int | None = None,
    block_lengths: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input_qdata: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_input_per_tensor_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_weight_per_tensor_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    high_first: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
    dynamic_activation_scale: bool = False,
    group_size: int | None = None,
) -> torch.Tensor:
    del high_first, dynamic_activation_scale
    _validate_optional_group_size(group_size, 2 * weight_qdata.shape[1])
    sequence_length = output_common.validate_output_metadata(
        key,
        logical_sequence_length,
        block_lengths,
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    return output_common.new_projected_output(
        key, sequence_length, weight_qdata.shape[0], output_dtype=output_dtype
    )


def _validate_residual_add(
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    key: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    weight_qdata: torch.Tensor,
    force_tile_ids: torch.Tensor | None,
    force_tile_counts: torch.Tensor | None,
) -> ResidualAdd:
    """Check from metadata that ``base`` has the projected output's rows and dtype."""
    sequence_length = output_common.validate_output_metadata(
        key,
        logical_sequence_length,
        block_lengths,
        output_dtype=base.dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    if base.ndim != 3 or base.shape[:2] != (key.shape[0], sequence_length):
        raise ValueError("sparse Piper residual base must have one row per output row")
    if base.device != key.device:
        raise ValueError("sparse Piper residual base must share the attention storage device")
    residual_add = ResidualAdd(
        base=base,
        gate=gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    validate_residual_add(base, residual_add, weight_qdata.shape[0])
    return residual_add


@torch.library.custom_op(
    "piper_kernels::nvfp4_sparse_piper_projected_query_attention_residual_add_",
    mutates_args=("base",),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _projected_query_attention_residual_add_op(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
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
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    query_chunk_rows: int | None = None,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input_qdata: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_input_per_tensor_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    high_first: bool = False,
    dynamic_activation_scale: bool = False,
    group_size: int | None = None,
) -> None:
    """Add ``gate[gate_indices] * attention_output`` into ``base`` without a full output.

    Matches ``nvfp4_sparse_piper_projected_query_attention_output``, using ``base``'s
    dtype as the output dtype and folding the gated residual update per projection window.
    """
    residual_add = _validate_residual_add(
        base,
        gate,
        gate_indices,
        python_indexing,
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata,
        force_tile_ids,
        force_tile_counts,
    )
    gate_projection = _prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        gate_input_qdata,
        gate_input_scale,
        gate_input_per_tensor_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_bias,
    )
    _run_projected_query_attention_output(
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
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        query_chunk_rows,
        block_lengths,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
        high_first=high_first,
        output_dtype=base.dtype,
        dynamic_activation_scale=dynamic_activation_scale,
        group_size=group_size,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
        residual_add=residual_add,
    )


@_projected_query_attention_residual_add_op.register_fake
def _projected_query_attention_residual_add_op_fake(  # noqa: PLR0913
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    _query_input_qdata: torch.Tensor,
    _query_input_scale: torch.Tensor,
    _query_input_per_tensor_scale: torch.Tensor,
    _query_weight_qdata: torch.Tensor,
    _query_weight_scale: torch.Tensor,
    _query_weight_per_tensor_scale: torch.Tensor | None,
    _query_bias: torch.Tensor | None,
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
    _weight_per_tensor_scale: torch.Tensor | None,
    _activation_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _query_chunk_rows: int | None = None,
    block_lengths: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input_qdata: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_input_per_tensor_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_weight_per_tensor_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    high_first: bool = False,
    dynamic_activation_scale: bool = False,
    group_size: int | None = None,
) -> None:
    del high_first, dynamic_activation_scale
    _validate_optional_group_size(group_size, 2 * weight_qdata.shape[1])
    _validate_residual_add(
        base,
        gate,
        gate_indices,
        python_indexing,
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata,
        force_tile_ids,
        force_tile_counts,
    )


@torch.library.custom_op(
    "piper_kernels::nvfp4_sparse_piper_attention_output",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _attention_output_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
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
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    query_chunk_rows: int | None = None,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input_qdata: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_input_per_tensor_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    high_first: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
    dynamic_activation_scale: bool = False,
    group_size: int | None = None,
) -> torch.Tensor:
    gate_projection = _prepare_optional_gate_projection(
        query,
        logical_sequence_length,
        block_lengths,
        gate_input_qdata,
        gate_input_scale,
        gate_input_per_tensor_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_bias,
    )
    return _run_attention_output(
        query,
        query_scale,
        query_summary,
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
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        query_chunk_rows,
        block_lengths,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
        high_first=high_first,
        output_dtype=output_dtype,
        dynamic_activation_scale=dynamic_activation_scale,
        group_size=group_size,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )


@_attention_output_op.register_fake
def _attention_output_op_fake(
    query: torch.Tensor,
    _query_scale: torch.Tensor,
    _query_summary: torch.Tensor,
    _key: torch.Tensor,
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
    _weight_per_tensor_scale: torch.Tensor | None,
    _activation_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _query_chunk_rows: int | None = None,
    block_lengths: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input_qdata: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_input_per_tensor_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_weight_per_tensor_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    high_first: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
    dynamic_activation_scale: bool = False,
    group_size: int | None = None,
) -> torch.Tensor:
    del high_first, dynamic_activation_scale
    _validate_optional_group_size(group_size, 2 * weight_qdata.shape[1])
    sequence_length = output_common.validate_output_metadata(
        query,
        logical_sequence_length,
        block_lengths,
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    return output_common.new_projected_output(
        query, sequence_length, weight_qdata.shape[0], output_dtype=output_dtype
    )


__all__: list[str] = []
