"""Bounded-workspace sparse Piper attention followed by a ConvRot INT8 output projection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from piper_kernels._residual import ResidualAdd
from piper_kernels._validation import validate_positive_finite_float
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS
from piper_kernels.attention.sparse_piper_attention._contracts import _SparsePiperRatioProfile
from piper_kernels.fusions.attention import _sparse_output as output_common
from piper_kernels.fusions.attention._validation import validate_qk_transform_inputs
from piper_kernels.fusions.convrot_int8_projection import output as projection_output
from piper_kernels.linear import _bias
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8._backend import LinearBackend
from piper_kernels.weights.convrot.int8._quantization import (
    validate_activation_scale,
    validate_storage,
)

from . import _backend as fusion_backend
from . import query as query_projection
from ._interfaces import ProjectionBackend

DEFAULT_QUERY_CHUNK_ROWS = output_common.DEFAULT_QUERY_CHUNK_ROWS
_DEFAULT_QUERY_CHUNK_ROWS = DEFAULT_QUERY_CHUNK_ROWS


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
    if not output_common.has_optional_operands(
        required,
        (input_scale, bias),
        "projected ConvRot INT8 gate requires input and weight storage",
    ):
        return None
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

    output_features = projection_output.validate_output_projection(
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_features=input_features,
        device=attention_storage.device,
        output_dtype=output_dtype,
        name="fused sparse Piper output",
    )
    return input_features, output_features


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
    residual_add: ResidualAdd | None = None,
) -> tuple[int, output_common.ChunkProjector, tuple[torch.Tensor, ...]]:
    """Prepare output-chunk buffers using the fusion's already-selected backend.

    With a validated ``residual_add``, chunks add their gated projection into its base
    rows instead of filling output rows that attention may borrow.
    """
    validate_activation_scale(output_input_scale, attention_storage.device)
    _input_features, output_features = _validate_output_projection(
        attention_storage,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        logical_sequence_length,
        query_chunk_rows,
        output_dtype=output_dtype,
    )
    project_chunk, retained = projection_output.prepare_chunk_projector(
        sequence_length,
        query_chunk_rows,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        backend=backend,
        output_input_scale=output_input_scale,
        residual_add=residual_add,
    )
    return output_features, project_chunk, retained


def _validate_projected_query_input(  # noqa: PLR0913
    query_input: torch.Tensor,
    query_input_scale: torch.Tensor | None,
    query_weight_qdata: torch.Tensor,
    query_weight_scale: torch.Tensor,
    key: torch.Tensor,
    sequence_length: int,
    *,
    query_norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_norm_epsilon: float,
    softmax_scale: float,
    input_group_size: int | None = None,
    query_bias: torch.Tensor | None = None,
) -> tuple[int, int, int, int] | None:
    """Return the full projection shape when the input is already prepared."""
    if key.ndim != 4:
        raise ValueError("fused sparse Piper output requires four-dimensional quantized storage")
    batch, heads, _storage_sequence_length, head_dim = key.shape
    if (
        query_input.ndim != 3
        or query_input.shape[:2] != (batch, sequence_length)
        or query_input.shape[-1] != query_weight_qdata.shape[-1]
        or query_input.device != key.device
        or query_input.layout is not torch.strided
    ):
        raise ValueError(
            "fused ConvRot INT8 Q input must match the global attention rows and width"
        )
    if input_group_size is None:
        if query_input_scale is None:
            raise ValueError("fused ConvRot INT8 prepared Q input requires row scales")
        _, _, query_heads = query_projection._validate_inputs(
            query_input,
            query_input_scale,
            query_weight_qdata,
            query_weight_scale,
            query_norm_weight,
            cos,
            sin,
            norm_epsilon=query_norm_epsilon,
            softmax_scale=softmax_scale,
            head_dim=head_dim,
            bias=query_bias,
        )
        if query_heads != heads:
            raise ValueError("fused ConvRot INT8 Q heads must match sparse attention key heads")
        return batch, sequence_length, query_heads, head_dim

    validate_storage(query_weight_qdata, query_weight_scale, input_group_size, query_input.dtype)
    validate_activation_scale(query_input_scale, query_input.device)
    output_features = heads * head_dim
    if query_weight_qdata.device != key.device or query_weight_qdata.shape[0] != output_features:
        raise ValueError("fused ConvRot INT8 Q heads must match sparse attention key heads")
    if query_bias is not None:
        _bias.validate_dtype(query_bias, "Q projection")
        if (
            query_bias.shape != (output_features,)
            or query_bias.device != key.device
            or query_bias.layout is not torch.strided
            or not query_bias.is_contiguous()
        ):
            raise ValueError(
                "Q projection bias must be contiguous with one value per output feature"
            )
    validate_qk_transform_inputs(
        query_norm_weight,
        cos,
        sin,
        sequence_length=sequence_length,
        head_dim=head_dim,
        device=key.device,
        norm_epsilon=query_norm_epsilon,
        name="Q projection",
    )
    validate_positive_finite_float(
        softmax_scale,
        "Q projection softmax scale must be finite and positive",
    )
    return None


def prepare_projected_gate_projection(
    key: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    query_input: torch.Tensor,
    gate_input: torch.Tensor | None,
    gate_input_scale: torch.Tensor | None,
    gate_weight_qdata: torch.Tensor | None,
    gate_weight_scale: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    input_group_size: int | None,
) -> _GateProjection | None:
    """Resolve and validate an optional coarse gate for projected-query attention."""
    if input_group_size is None:
        resolved_gate_input = gate_input
    elif gate_input is not None:
        raise ValueError("chunk-prepared ConvRot INT8 Q and gate must share one input")
    else:
        resolved_gate_input = query_input if gate_weight_qdata is not None else None
    return _prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        resolved_gate_input,
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        input_group_size,
    )


@dataclass(frozen=True, slots=True)
class SparseProjectedQueryOperands:
    """Shared operands for bounded sparse projected-query attention-output ops."""

    query_input: torch.Tensor
    query_input_scale: torch.Tensor | None
    query_weight_qdata: torch.Tensor
    query_weight_scale: torch.Tensor
    query_norm_weight: torch.Tensor | None
    cos: torch.Tensor
    sin: torch.Tensor
    query_norm_epsilon: float
    softmax_scale: float
    key: torch.Tensor
    key_scale: torch.Tensor
    key_center: torch.Tensor
    key_center_scale: torch.Tensor
    key_summary: torch.Tensor
    key_aux: torch.Tensor
    value: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_mean: torch.Tensor
    ratio_profile: _SparsePiperRatioProfile
    sparse_key_blocks: int
    logical_sequence_length: int
    routing_mode: int
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None
    group_size: int
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS
    block_lengths: torch.Tensor | None = None
    coarse_gate: torch.Tensor | None = None
    coarse_scale: float | None = None
    coarse_key_blocks: int | None = None
    sparse_query_blocks: int | None = None
    input_group_size: int | None = None
    output_dtype: torch.dtype = torch.bfloat16
    query_bias: torch.Tensor | None = None
    output_input_scale: torch.Tensor | None = None
    force_tile_ids: torch.Tensor | None = None
    force_tile_counts: torch.Tensor | None = None

    def _project_query_chunk(
        self,
        start: int,
        rows: int,
        *,
        backend: LinearBackend,
        projection_backend: ProjectionBackend,
        validated_shape: tuple[int, int, int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_input, chunk_scale = self.query_input, self.query_input_scale
        cos, sin, lengths = self.cos, self.sin, self.block_lengths
        if self.input_group_size is not None:
            end = start + rows
            chunk_input, chunk_scale = backend.prepare_input(
                chunk_input[:, start:end].contiguous(),
                self.input_group_size,
                input_scale=chunk_scale,
            )
            cos, sin = cos[start:end], sin[start:end]
            if lengths is not None:
                lengths = lengths[start // TILE_ROWS : (end + TILE_ROWS - 1) // TILE_ROWS]
            start = 0
        assert chunk_scale is not None
        return query_projection._launch_query_projection_range(
            chunk_input,
            chunk_scale,
            self.query_weight_qdata,
            self.query_weight_scale,
            self.query_norm_weight,
            cos,
            sin,
            self.query_norm_epsilon,
            self.softmax_scale,
            self.routing_mode,
            lengths,
            chunk_start=start,
            chunk_rows=rows,
            backend=projection_backend,
            head_dim=self.key.shape[-1],
            bias=self.query_bias,
            validated_shape=validated_shape,
        )

    def prepare_context(
        self,
        *,
        has_projected_coarse_gate: bool = False,
        start_head: int = 0,
        end_head: int | None = None,
    ) -> tuple[
        LinearBackend,
        output_common._PreparedAttentionContext,
        output_common.QueryChunkProjector,
    ]:
        """Validate Q/K/V and build shared sparse context plus bounded query chunk projector."""
        projection_backend = fusion_backend.require_projection_backend(
            self.query_input, head_dim=self.key.shape[-1]
        )
        backend = fusion_backend.require_output_backend(self.key)
        prepared = output_common.prepare_attention_context(
            self.key,
            self.key_scale,
            self.key_center,
            self.key_center_scale,
            self.key_summary,
            self.key_aux,
            self.value,
            self.value_scale_multiplier,
            self.value_mean,
            self.ratio_profile,
            self.sparse_key_blocks,
            self.logical_sequence_length,
            self.routing_mode,
            self.block_lengths,
            self.coarse_gate,
            self.coarse_scale,
            self.coarse_key_blocks,
            self.sparse_query_blocks,
            has_projected_coarse_gate=has_projected_coarse_gate,
            force_tile_ids=self.force_tile_ids,
            force_tile_counts=self.force_tile_counts,
            start_head=start_head,
            end_head=end_head,
        )
        validated_shape = _validate_projected_query_input(
            self.query_input,
            self.query_input_scale,
            self.query_weight_qdata,
            self.query_weight_scale,
            self.key,
            prepared.sequence_length,
            query_norm_weight=self.query_norm_weight,
            cos=self.cos,
            sin=self.sin,
            query_norm_epsilon=self.query_norm_epsilon,
            softmax_scale=self.softmax_scale,
            input_group_size=self.input_group_size,
            query_bias=self.query_bias,
        )
        project_query_chunk = partial(
            self._project_query_chunk,
            backend=backend,
            projection_backend=projection_backend,
            validated_shape=validated_shape,
        )
        return backend, prepared, project_query_chunk

    def run(
        self,
        *,
        gate_projection: _GateProjection | None = None,
        out: torch.Tensor | None = None,
        residual_add: ResidualAdd | None = None,
        start_head: int = 0,
        end_head: int | None = None,
    ) -> torch.Tensor:
        """Lifetime-chunk Q through routing, attention, and ConvRot INT8 output."""
        backend, prepared, project_query_chunk = self.prepare_context(
            has_projected_coarse_gate=gate_projection is not None,
            start_head=start_head,
            end_head=end_head,
        )
        output_features, project_chunk, projector_tensors = _prepare_output_chunk_projector(
            self.key,
            prepared.sequence_length,
            self.weight_qdata,
            self.weight_scale,
            self.bias,
            self.group_size,
            self.logical_sequence_length,
            self.query_chunk_rows,
            backend=backend,
            output_dtype=self.output_dtype,
            output_input_scale=self.output_input_scale,
            residual_add=residual_add,
        )
        return output_common.run_chunked_projected_query_attention_output(
            prepared,
            output_features,
            self.query_chunk_rows,
            project_query_chunk,
            project_chunk,
            projector_tensors,
            project_coarse_gate_chunk=(
                None if gate_projection is None else gate_projection.project
            ),
            output_dtype=self.output_dtype,
            out=out if residual_add is None else residual_add.base,
            reuse_output_for_attention=residual_add is None,
        )


def _resolve_distributed_comm(
    comm_group_size: int | None,
    comm_group_name: str | None,
) -> tuple[int, str] | None:
    if comm_group_size is None and comm_group_name is None:
        return None
    if comm_group_size is None or comm_group_name is None:
        raise ValueError(
            "distributed sparse Piper requires both comm_group_size and comm_group_name"
        )
    return comm_group_size, comm_group_name


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_projected_query_attention_output",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
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
    comm_group_size: int | None = None,
    comm_group_name: str | None = None,
) -> torch.Tensor:
    comm = _resolve_distributed_comm(comm_group_size, comm_group_name)
    gate_projection = prepare_projected_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        query_input,
        gate_input,
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        input_group_size,
    )
    operands = SparseProjectedQueryOperands(
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
        input_group_size=input_group_size,
        output_dtype=output_dtype,
        query_bias=query_bias,
        output_input_scale=output_input_scale,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    if comm is not None:
        from . import distributed_output  # noqa: PLC0415

        return distributed_output.run_distributed_projected_query_attention_output(
            comm[0], comm[1], operands, gate_projection=gate_projection
        )
    return operands.run(gate_projection=gate_projection)


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_projected_query_attention_output_",
    mutates_args=("query_input",),
    tags=(torch.Tag.cudagraph_unsafe,),
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
) -> None:
    """Overwrite an exclusive floating-point input after each window is consumed.

    Compiler ownership checks exclude caller inputs, aliases, and escaping values.
    Q and gate finish reading a window before its output projection can overwrite it.
    """
    if input_group_size is None:
        raise ValueError("input reuse requires chunked floating-point input preparation")
    gate_projection = prepare_projected_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        query_input,
        gate_input,
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        input_group_size,
    )
    SparseProjectedQueryOperands(
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
        input_group_size=input_group_size,
        output_dtype=output_dtype,
        query_bias=query_bias,
        output_input_scale=output_input_scale,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    ).run(gate_projection=gate_projection, out=query_input)


@_projected_query_attention_output_inplace_op.register_fake
def _projected_query_attention_output_inplace_op_fake(
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
    _weight_qdata: torch.Tensor,
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
) -> None:
    output_common.validate_output_metadata(
        key,
        logical_sequence_length,
        block_lengths,
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )


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
    comm_group_size: int | None = None,
    comm_group_name: str | None = None,
) -> torch.Tensor:
    comm = _resolve_distributed_comm(comm_group_size, comm_group_name)
    if comm is not None:
        from piper_kernels._distributed import _peer_exchange as peer_exchange  # noqa: PLC0415

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
            comm[0],
            comm[1],
            name="distributed sparse Piper",
        )
        return output_common.new_projected_output(
            key, local_sequence_length, weight_qdata.shape[0], output_dtype=output_dtype
        )
    return output_common.fake_projected_output(
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )


def _resolve_head_slice(
    ratio_profile: _SparsePiperRatioProfile,
    query_weight_qdata: torch.Tensor,
    key: torch.Tensor,
    head_slice_index: int,
    num_head_slices: int,
) -> tuple[int, int]:
    """Return the profile heads ``[start_head, end_head)`` that this call's Q covers."""
    if not 0 <= head_slice_index < num_head_slices:
        raise ValueError("sparse Piper head slice index must lie in [0, num_head_slices)")
    query_heads = query_weight_qdata.shape[0] // key.shape[-1]
    if ratio_profile.num_heads != query_heads * num_head_slices:
        raise ValueError("sparse Piper ratio profile must cover every head slice's query heads")
    start_head = head_slice_index * query_heads
    return start_head, start_head + query_heads


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_projected_query_attention_residual_add_",
    mutates_args=("base",),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _projected_query_attention_residual_add_op(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
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
    comm_group_size: int | None = None,
    comm_group_name: str | None = None,
    head_slice_index: int = 0,
    num_head_slices: int = 1,
) -> None:
    """Add ``gate[gate_indices] * attention_output`` into ``base`` without a full-size output.

    The attention operands match
    ``convrot_int8_sparse_piper_projected_query_attention_output``, whose ``gate_*``
    operands project the coarse gate; ``gate`` is the residual gate, and the output
    dtype is ``base``'s. The linear backend adds each window's projection into its
    ``base`` rows in FP32; its native epilogue preserves projection rounding without scratch.

    When ``comm_group_size`` and ``comm_group_name`` are provided, overlaps bounded
    peer-chunk reduce-scatter into this rank's ``base`` rows (head slicing must remain
    at ``head_slice_index=0, num_head_slices=1``).
    """
    comm = _resolve_distributed_comm(comm_group_size, comm_group_name)
    if comm is not None:
        if head_slice_index != 0 or num_head_slices != 1:
            raise ValueError("distributed sparse Piper residual add does not support head slicing")
        from . import distributed_output  # noqa: PLC0415

        residual_add = ResidualAdd(
            base=base,
            gate=gate,
            gate_indices=gate_indices,
            python_indexing=python_indexing,
        )
        distributed_output.validate_distributed_residual_add(
            comm[0],
            comm[1],
            residual_add,
            key,
            logical_sequence_length,
            block_lengths,
            weight_qdata,
            force_tile_ids,
            force_tile_counts,
        )
        start_head, end_head = 0, None
    else:
        start_head, end_head = _resolve_head_slice(
            ratio_profile, query_weight_qdata, key, head_slice_index, num_head_slices
        )
        residual_add = output_common.validate_residual_add(
            base,
            gate,
            gate_indices,
            python_indexing,
            key,
            logical_sequence_length,
            block_lengths,
            weight_qdata.shape[0],
            force_tile_ids,
            force_tile_counts,
        )
    gate_projection = prepare_projected_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        query_input,
        gate_input,
        gate_input_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        input_group_size,
    )
    operands = SparseProjectedQueryOperands(
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
        input_group_size=input_group_size,
        output_dtype=base.dtype,
        query_bias=query_bias,
        output_input_scale=output_input_scale,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    if comm is not None:
        from . import distributed_output  # noqa: PLC0415

        distributed_output.run_distributed_projected_query_attention_output(
            comm[0],
            comm[1],
            operands,
            gate_projection=gate_projection,
            residual_add=residual_add,
        )
        return
    operands.run(
        gate_projection=gate_projection,
        residual_add=residual_add,
        start_head=start_head,
        end_head=end_head,
    )


@_projected_query_attention_residual_add_op.register_fake
def _projected_query_attention_residual_add_op_fake(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    _query_input: torch.Tensor,
    _query_input_scale: torch.Tensor | None,
    query_weight_qdata: torch.Tensor,
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
    ratio_profile: _SparsePiperRatioProfile,
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
    comm_group_size: int | None = None,
    comm_group_name: str | None = None,
    head_slice_index: int = 0,
    num_head_slices: int = 1,
) -> None:
    comm = _resolve_distributed_comm(comm_group_size, comm_group_name)
    if comm is not None:
        if head_slice_index != 0 or num_head_slices != 1:
            raise ValueError("distributed sparse Piper residual add does not support head slicing")
        from . import distributed_output  # noqa: PLC0415

        distributed_output.validate_distributed_residual_add(
            comm[0],
            comm[1],
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
        return
    _resolve_head_slice(ratio_profile, query_weight_qdata, key, head_slice_index, num_head_slices)
    output_common.validate_residual_add(
        base,
        gate,
        gate_indices,
        python_indexing,
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
        force_tile_ids,
        force_tile_counts,
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_attention_output",
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
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input_qdata: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    output_input_scale: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    backend = fusion_backend.require_output_backend(query)
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
        reuse_output_for_attention=True,
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
    _bias: torch.Tensor | None,
    _group_size: int,
    _query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input_qdata: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
    _output_input_scale: torch.Tensor | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return output_common.fake_projected_output(
        query,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )


GateProjection = _GateProjection
validate_output_projection = _validate_output_projection

__all__: list[str] = []
