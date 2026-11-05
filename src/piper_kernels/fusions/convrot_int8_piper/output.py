"""Bounded Q projection, dense attention, and ConvRot INT8 output projection."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from piper_kernels.attention.piper_attention import _quantized_dispatch as attention
from piper_kernels.fusions.attention import _output as pipeline
from piper_kernels.fusions.convrot_int8_projection import output as projection_output
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.weights.convrot.int8._quantization import validate_activation_scale

from . import _backend, _schedule, query

if TYPE_CHECKING:
    from piper_kernels.fusions.residual_updates.triton import (
        ResidualAdd,
        ResidualAddLayout,
    )

# Share the storage cap with sparse; dense scheduling may select smaller windows.
DEFAULT_QUERY_CHUNK_ROWS = pipeline.DEFAULT_QUERY_CHUNK_ROWS


def _validate_inputs(  # noqa: PLR0913, PLR0917
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
    query_chunk_rows: int = DEFAULT_QUERY_CHUNK_ROWS,
) -> tuple[int, int, int, int]:
    """Validate producer, native context, and output metadata without tensor work."""
    batch, sequence, heads, head_dim = query._validate_inputs(
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
        head_dim,
    )
    shape = (batch, heads, sequence, head_dim)
    attention.validate_quantized_context(
        key,
        key_scale,
        value,
        multiplier,
        log_scale,
        value_mean,
        query_shape=shape,
        query_device=input_qdata.device,
        key_length=key_length,
        is_causal=is_causal,
        output_dtype=output_dtype,
    )
    if (
        isinstance(query_chunk_rows, bool)
        or not isinstance(query_chunk_rows, int)
        or query_chunk_rows < 128
        or query_chunk_rows % 128
    ):
        raise ValueError("dense Piper query chunks must be positive multiples of 128 rows")
    validate_activation_scale(output_input_scale, input_qdata.device)
    projection_output.validate_output_projection(
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_features=heads * head_dim,
        device=input_qdata.device,
        output_dtype=output_dtype,
        name="dense Piper output",
    )
    return shape


def _run_projected_query_attention(  # noqa: PLR0913, PLR0917
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
    context_operands: tuple[torch.Tensor, ...],
    key_length: int,
    is_causal: bool,
    query_shape: tuple[int, int, int, int],
    output_features: int,
    output_dtype: torch.dtype,
    query_chunk_rows: int,
    build_projector: Callable[[int], tuple[pipeline.ChunkProjector, tuple[torch.Tensor, ...]]],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run validated Q windows through attention and a chunk projector.

    ``context_operands`` are the prepared K and V outputs. Without ``out``, attention
    windows may borrow unwritten output rows; ``out`` rows hold live data, so
    attention then uses separate bounded buffers.
    """
    batch, heads, sequence, head_dim = query_shape
    query_chunk_rows = _schedule.select_query_chunk_rows(
        query_shape,
        input_qdata.device,
        query_chunk_rows,
        is_causal=is_causal,
    )
    backend = _backend.require_projection_backend(input_qdata, head_dim=head_dim)
    context = attention.prepare_quantized_context(
        *context_operands,
        query_length=sequence,
        key_length=key_length,
        is_causal=is_causal,
    )
    capacity = (min(query_chunk_rows, sequence) + 63) // 64 * 64
    query_buffers = (
        input_qdata.new_empty((batch, heads, capacity, head_dim)),
        input_qdata.new_empty((batch, heads, capacity // 32), dtype=torch.float32),
    )
    project_chunk, retained = build_projector(query_chunk_rows)

    def launch_chunk(
        output: torch.Tensor, start: int, rows: int, _auxiliary: torch.Tensor | None
    ) -> None:
        backend.project_query(
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
            chunk_start=start,
            chunk_rows=rows,
            out=query_buffers,
        )
        attention.launch_quantized_attention_into(
            context, *query_buffers, output.transpose(1, 2), global_row_offset=start
        )

    return pipeline.run_chunked_attention_output(
        (batch, sequence, heads, head_dim),
        input_qdata.device,
        output_features,
        query_chunk_rows,
        launch_chunk,
        project_chunk,
        retained,
        output_dtype=output_dtype,
        out=out,
        reuse_output_for_attention=out is None,
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_piper_projected_query_attention_output", mutates_args=()
)
def _projected_query_attention_output_op(  # noqa: PLR0913, PLR0917
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
    query_chunk_rows: int = DEFAULT_QUERY_CHUNK_ROWS,
) -> torch.Tensor:
    """Pipeline local Q through attention and output with query_chunk_rows as a cap."""
    batch, heads, sequence, head_dim = _validate_inputs(
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
    output_features = weight_qdata.shape[0]
    if batch == 0:
        return input_qdata.new_empty((batch, sequence, output_features), dtype=output_dtype)
    output_backend = linear_backend.require_linear_backend(input_qdata)

    def build_projector(
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

    return _run_projected_query_attention(
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
        (key, key_scale, value, multiplier, log_scale, value_mean),
        key_length,
        is_causal,
        (batch, heads, sequence, head_dim),
        output_features,
        output_dtype,
        query_chunk_rows,
        build_projector,
    )


@_projected_query_attention_output_op.register_fake
def _projected_query_attention_output_op_fake(  # noqa: PLR0913, PLR0917
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
    query_chunk_rows: int = DEFAULT_QUERY_CHUNK_ROWS,
) -> torch.Tensor:
    batch, _, sequence, _ = _validate_inputs(
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
    return input_qdata.new_empty((batch, sequence, weight_qdata.shape[0]), dtype=output_dtype)


def _validate_residual_add_inputs(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
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
    output_input_scale: torch.Tensor | None,
    *,
    head_dim: int,
    query_chunk_rows: int,
) -> tuple[tuple[int, int, int, int], "ResidualAdd", "ResidualAddLayout"]:
    """Check attention-output and residual metadata without tensor work."""
    # Triton loads only when the residual add runs.
    from piper_kernels.fusions.residual_updates import triton as residual_updates  # noqa: PLC0415

    shape = _validate_inputs(
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
    )
    batch, _, sequence, _ = shape
    if base.ndim != 3 or base.shape[:2] != (batch, sequence):
        raise ValueError("dense Piper residual base must have one row per query token")
    updates = residual_updates.ResidualAdd(
        base=base,
        gate=gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    layout = residual_updates.validate_residual_add(base, updates, weight_qdata.shape[0])
    return shape, updates, layout


@torch.library.custom_op(
    "piper_kernels::convrot_int8_piper_projected_query_attention_residual_add_",
    mutates_args=("base",),
)
def _projected_query_attention_residual_add_op(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
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
    query_chunk_rows: int = DEFAULT_QUERY_CHUNK_ROWS,
) -> None:
    """Add ``gate[gate_indices] * attention_output`` into ``base`` without a full-size output.

    The attention operands match ``convrot_int8_piper_projected_query_attention_output``;
    the output dtype is ``base``'s. Each window is projected into bounded scratch and
    added into its ``base`` rows in FP32 with one rounding.
    """
    shape, updates, layout = _validate_residual_add_inputs(
        base,
        gate,
        gate_indices,
        python_indexing,
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
        value,
        multiplier,
        log_scale,
        value_mean,
        key_length,
        is_causal,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        output_input_scale,
        head_dim=head_dim,
        query_chunk_rows=query_chunk_rows,
    )
    if shape[0] == 0:
        return
    output_backend = linear_backend.require_linear_backend(input_qdata)

    def build_projector(
        chunk_rows: int,
    ) -> tuple[pipeline.ChunkProjector, tuple[torch.Tensor, ...]]:
        return projection_output.prepare_residual_add_projector(
            shape[2],
            chunk_rows,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            updates,
            layout,
            backend=output_backend,
            output_input_scale=output_input_scale,
        )

    _run_projected_query_attention(
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
        (key, key_scale, value, multiplier, log_scale, value_mean),
        key_length,
        is_causal,
        shape,
        weight_qdata.shape[0],
        base.dtype,
        query_chunk_rows,
        build_projector,
        out=base,
    )


@_projected_query_attention_residual_add_op.register_fake
def _projected_query_attention_residual_add_op_fake(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
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
    query_chunk_rows: int = DEFAULT_QUERY_CHUNK_ROWS,
) -> None:
    _validate_residual_add_inputs(
        base,
        gate,
        gate_indices,
        python_indexing,
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
        value,
        multiplier,
        log_scale,
        value_mean,
        key_length,
        is_causal,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        output_input_scale,
        head_dim=head_dim,
        query_chunk_rows=query_chunk_rows,
    )
