"""Bounded Q projection, dense attention, and ConvRot INT8 output projection."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from piper_kernels._residual import ResidualAdd, validate_residual_add
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    count_query_scales,
    resolve_qk_quantization,
)
from piper_kernels.attention.piper_attention import _quantized_dispatch as attention
from piper_kernels.fusions.attention import _output as pipeline
from piper_kernels.fusions.convrot_int8_projection import output as projection_output
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.weights.convrot.int8._quantization import validate_activation_scale

from . import _backend, _schedule, query

# Share the storage cap with sparse; dense scheduling may select smaller windows.
DEFAULT_QUERY_CHUNK_ROWS = pipeline.DEFAULT_QUERY_CHUNK_ROWS


@dataclass(frozen=True, slots=True)
class DenseOutputOperands:
    """Shared operands for bounded dense ConvRot INT8 attention-output execution."""

    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    query_weight: torch.Tensor
    query_weight_scale: torch.Tensor
    norm_weight: torch.Tensor | None
    cos: torch.Tensor
    sin: torch.Tensor
    norm_epsilon: float
    softmax_scale: float
    query_bias: torch.Tensor | None
    key: torch.Tensor
    key_scale: torch.Tensor
    key_center: torch.Tensor
    value: torch.Tensor
    multiplier: torch.Tensor
    log_scale: torch.Tensor
    value_mean: torch.Tensor
    key_length: int
    is_causal: bool
    output_dtype: torch.dtype
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None
    group_size: int
    output_input_scale: torch.Tensor | None
    head_dim: int
    query_chunk_rows: int = DEFAULT_QUERY_CHUNK_ROWS

    def validate(self) -> tuple[int, int, int, int]:
        """Validate producer, native context, and output metadata without tensor work."""
        batch, sequence, heads, head_dim = query._validate_inputs(
            self.input_qdata,
            self.input_scale,
            self.query_weight,
            self.query_weight_scale,
            self.norm_weight,
            self.cos,
            self.sin,
            self.norm_epsilon,
            self.softmax_scale,
            self.query_bias,
            self.head_dim,
        )
        shape = (batch, heads, sequence, head_dim)
        attention.validate_quantized_context(
            self.key,
            self.key_scale,
            self.key_center,
            self.value,
            self.multiplier,
            self.log_scale,
            self.value_mean,
            query_shape=shape,
            query_device=self.input_qdata.device,
            key_length=self.key_length,
            is_causal=self.is_causal,
            output_dtype=self.output_dtype,
        )
        if (
            isinstance(self.query_chunk_rows, bool)
            or not isinstance(self.query_chunk_rows, int)
            or self.query_chunk_rows < 128
            or self.query_chunk_rows % 128
        ):
            raise ValueError("dense Piper query chunks must be positive multiples of 128 rows")
        validate_activation_scale(self.output_input_scale, self.input_qdata.device)
        projection_output.validate_output_projection(
            self.weight_qdata,
            self.weight_scale,
            self.bias,
            self.group_size,
            input_features=heads * head_dim,
            device=self.input_qdata.device,
            output_dtype=self.output_dtype,
            name="dense Piper output",
        )
        return shape

    def validate_residual_add(
        self,
        residual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool],
    ) -> tuple[tuple[int, int, int, int], ResidualAdd]:
        """Check attention-output and residual metadata without tensor work."""
        shape = self.validate()
        batch, _, sequence, _ = shape
        base, gate, gate_indices, python_indexing = residual
        if base.ndim != 3 or base.shape[:2] != (batch, sequence):
            raise ValueError("dense Piper residual base must have one row per query token")
        if base.device != self.input_qdata.device:
            raise ValueError("dense Piper residual base must share the input storage device")
        residual_add = ResidualAdd(
            base=base,
            gate=gate,
            gate_indices=gate_indices,
            python_indexing=python_indexing,
        )
        validate_residual_add(base, residual_add, self.weight_qdata.shape[0])
        return shape, residual_add

    def prepare_window_runner(
        self,
        query_shape: tuple[int, int, int, int],
        capacity: int,
        *,
        context: attention.QuantizedContext | None = None,
        query_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        attention.QuantizedContext,
        tuple[torch.Tensor, torch.Tensor],
        Callable[[int, int, torch.Tensor], None],
    ]:
        """Prepare native projection/attention state and reusable Q scratch for chunked launches."""
        batch, heads, sequence, head_dim = query_shape
        backend = _backend.require_projection_backend(self.input_qdata, head_dim=head_dim)
        if context is None:
            context = attention.prepare_quantized_context(
                self.key,
                self.key_scale,
                self.key_center,
                self.value,
                self.multiplier,
                self.log_scale,
                self.value_mean,
                query_length=sequence,
                key_length=self.key_length,
                is_causal=self.is_causal,
            )
        if query_buffers is None:
            padded_capacity = (capacity + 63) // 64 * 64
            # Q scales follow the granularity of the validated K scales.
            granularity = resolve_qk_quantization(self.key_scale, self.key.shape[2])
            query_buffers = (
                self.input_qdata.new_empty((batch, heads, padded_capacity, head_dim)),
                self.input_qdata.new_empty(
                    (batch, heads, count_query_scales(padded_capacity, granularity)),
                    dtype=torch.float32,
                ),
            )
        active_buffers = query_buffers
        active_context = context

        def launch_window(
            global_start: int,
            rows: int,
            attention_chunk: torch.Tensor,
        ) -> None:
            backend.project_query(
                self.input_qdata,
                self.input_scale,
                self.query_weight,
                self.query_weight_scale,
                self.norm_weight,
                self.cos,
                self.sin,
                self.norm_epsilon,
                self.softmax_scale,
                self.query_bias,
                chunk_start=global_start,
                chunk_rows=rows,
                out=active_buffers,
            )
            attention.launch_quantized_attention_into(
                active_context,
                *active_buffers,
                attention_chunk.transpose(1, 2),
                global_row_offset=global_start,
            )

        return context, query_buffers, launch_window

    def prepare_chunk_projector(
        self,
        sequence_length: int,
        chunk_rows: int,
        *,
        backend: linear_backend.LinearBackend,
        residual_add: ResidualAdd | None = None,
    ) -> tuple[pipeline.ChunkProjector, tuple[torch.Tensor, ...]]:
        """Build chunked output-projection storage for the shared backend."""
        return projection_output.prepare_chunk_projector(
            sequence_length,
            chunk_rows,
            self.weight_qdata,
            self.weight_scale,
            self.bias,
            self.group_size,
            backend=backend,
            output_input_scale=self.output_input_scale,
            residual_add=residual_add,
        )

    def run(
        self,
        query_shape: tuple[int, int, int, int],
        *,
        output_backend: linear_backend.LinearBackend | None = None,
        residual_add: ResidualAdd | None = None,
        out: torch.Tensor | None = None,
        global_row_offset: int = 0,
        sequence_length: int | None = None,
        query_chunk_rows: int | None = None,
        context: attention.QuantizedContext | None = None,
        query_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run validated Q windows through attention and a chunk projector.

        Without ``out``, attention windows may borrow unwritten output rows; ``out`` rows
        hold live data, so attention then uses separate bounded buffers.
        """
        batch, heads, sequence, head_dim = query_shape
        target_sequence = sequence if sequence_length is None else sequence_length
        resolved_chunk_rows = _schedule.select_query_chunk_rows(
            (batch, heads, target_sequence, head_dim),
            self.input_qdata.device,
            self.query_chunk_rows if query_chunk_rows is None else query_chunk_rows,
            is_causal=self.is_causal,
        )
        resolved_output_backend = (
            linear_backend.require_linear_backend(self.input_qdata)
            if output_backend is None
            else output_backend
        )
        _context, _buffers, launch_window = self.prepare_window_runner(
            query_shape,
            min(resolved_chunk_rows, target_sequence),
            context=context,
            query_buffers=query_buffers,
        )
        project_chunk, retained = self.prepare_chunk_projector(
            target_sequence,
            resolved_chunk_rows,
            backend=resolved_output_backend,
            residual_add=residual_add,
        )

        def launch_chunk(
            output: torch.Tensor, start: int, rows: int, _auxiliary: torch.Tensor | None
        ) -> None:
            launch_window(global_row_offset + start, rows, output)

        return pipeline.run_chunked_attention_output(
            (batch, target_sequence, heads, head_dim),
            self.input_qdata.device,
            self.weight_qdata.shape[0],
            resolved_chunk_rows,
            launch_chunk,
            project_chunk,
            retained,
            output_dtype=self.output_dtype,
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
    key_center: torch.Tensor,
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
    operands = DenseOutputOperands(
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
        key_center,
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
        head_dim,
        query_chunk_rows,
    )
    shape = operands.validate()
    if shape[0] == 0:
        return input_qdata.new_empty((0, shape[2], weight_qdata.shape[0]), dtype=output_dtype)
    return operands.run(shape)


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
    key_center: torch.Tensor,
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
    batch, _, sequence, _ = DenseOutputOperands(
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
        key_center,
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
        head_dim,
        query_chunk_rows,
    ).validate()
    return input_qdata.new_empty((batch, sequence, weight_qdata.shape[0]), dtype=output_dtype)


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
    key_center: torch.Tensor,
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
    the output dtype is ``base``'s. The linear backend adds each window's projection
    into its ``base`` rows in FP32. The native epilogue preserves projection rounding
    without scratch.
    """
    operands = DenseOutputOperands(
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
        key_center,
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
        head_dim,
        query_chunk_rows,
    )
    shape, residual_add = operands.validate_residual_add(
        (base, gate, gate_indices, python_indexing)
    )
    if shape[0] == 0:
        return
    operands.run(shape, residual_add=residual_add, out=base)


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
    key_center: torch.Tensor,
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
    DenseOutputOperands(
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
        key_center,
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
        head_dim,
        query_chunk_rows,
    ).validate_residual_add((base, gate, gate_indices, python_indexing))


__all__: list[str] = []
