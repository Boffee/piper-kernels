"""Shared sparse-attention output pipeline adapters and Q/K/V storage helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from piper_kernels._residual import ResidualAdd
from piper_kernels._residual import validate_residual_add as _validate_residual_add_operands
from piper_kernels.attention.kernels.sparse_piper.layout import (
    QUERY_SCALE_ROWS,
    SUPPORTED_HEAD_DIMS,
    TILE_ROWS,
    ValueFormat,
    padded_sequence_length,
)
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    _SparsePiperRatioProfile,
    validate_block_lengths,
    validate_output_dtype,
)

from . import _output as output_pipeline

if TYPE_CHECKING:
    from piper_kernels.attention.sparse_piper_attention._prepared import (
        _PreparedSparsePiperAttention,
    )

DEFAULT_QUERY_CHUNK_ROWS = output_pipeline.DEFAULT_QUERY_CHUNK_ROWS
_MIN_PROJECTED_GATE_PIPELINE_CHUNKS = 8

AttentionProjector = output_pipeline.AttentionProjector
ChunkProjector = output_pipeline.ChunkProjector
CoarseGateChunkProjector = output_pipeline.AuxiliaryChunkProjector
type QueryChunkProjector = Callable[[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

output_sequence_length = _quantized_dispatch._quantized_attention_output_sequence_length
validate_output_metadata = _quantized_dispatch.validate_quantized_attention_output


def has_optional_operands(
    required: Sequence[object],
    optional: Sequence[object],
    message: str,
) -> bool:
    """Return whether an optional operand group is present, rejecting partial sets."""
    if not any(value is not None for value in (*required, *optional)):
        return False
    if any(value is None for value in required):
        raise ValueError(message)
    return True


def validate_residual_add(
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    key: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    output_features: int,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> ResidualAdd:
    """Check from metadata that ``base`` has the projected output's rows and dtype."""
    sequence_length = validate_output_metadata(
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
    _validate_residual_add_operands(base, residual_add, output_features)
    return residual_add


def fake_projected_output(
    storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    output_features: int,
    *,
    output_dtype: torch.dtype,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate sparse attention-output metadata and allocate its fake output."""
    sequence_length = validate_output_metadata(
        storage,
        logical_sequence_length,
        block_lengths,
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    return new_projected_output(
        storage,
        sequence_length,
        output_features,
        output_dtype=output_dtype,
    )


def validate_optional_block_lengths(
    block_lengths: torch.Tensor | None,
    sequence_length: int,
    device: torch.device,
    context: str,
) -> None:
    """Validate optional trusted valid-prefix metadata for aligned K64 storage."""
    if block_lengths is not None:
        validate_block_lengths(
            block_lengths,
            sequence_length=sequence_length,
            device=device,
            context=context,
            require_contiguous=True,
            check_values=False,
        )


def validate_query_range(chunk_start: int, chunk_rows: int, sequence_length: int) -> None:
    """Validate an aligned ``[chunk_start, chunk_start + chunk_rows)`` query window."""
    if (
        isinstance(chunk_start, bool)
        or not isinstance(chunk_start, int)
        or isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_start < 0
        or chunk_rows < 1
        or chunk_start % TILE_ROWS
        or chunk_start + chunk_rows > sequence_length
    ):
        raise ValueError("Q projection range must be a nonempty aligned sequence window")


def allocate_query_storage(
    template: torch.Tensor,
    batch: int,
    heads: int,
    rows: int,
    head_dim: int,
    *,
    summary_template: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate tile-padded Q storage, Q scales, and summary buffers."""
    storage_sequence_length = padded_sequence_length(rows)
    resolved_summary = template if summary_template is None else summary_template
    return (
        template.new_empty((batch, heads, storage_sequence_length, head_dim), dtype=torch.int8),
        template.new_empty(
            (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
            dtype=torch.float32,
        ),
        resolved_summary.new_empty(
            (batch, heads, storage_sequence_length // TILE_ROWS, head_dim),
            dtype=torch.float32,
        ),
    )


def allocate_key_storage(
    template: torch.Tensor,
    batch: int,
    heads: int,
    rows: int,
    head_dim: int,
    routing_mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate tile-padded sparse-Piper key storage and per-tile metadata."""
    storage_sequence_length = padded_sequence_length(rows)
    storage_tiles = storage_sequence_length // TILE_ROWS
    summary_shape = (batch, heads, storage_tiles, head_dim)
    return (
        template.new_empty((batch, heads, storage_sequence_length, head_dim), dtype=torch.int8),
        template.new_empty((batch, heads, storage_tiles), dtype=torch.float32),
        template.new_empty((batch, heads, storage_tiles, 2, head_dim), dtype=torch.int8),
        template.new_empty((batch, heads, storage_tiles, 2), dtype=torch.float32),
        template.new_empty(summary_shape, dtype=torch.float32),
        template.new_empty(
            (batch, heads, 0 if routing_mode == _MEAN_ROUTING else storage_tiles, head_dim),
            dtype=torch.float32,
        ),
    )


def allocate_value_storage(
    template: torch.Tensor,
    batch: int,
    heads: int,
    rows: int,
    head_dim: int,
    storage_format: ValueFormat = ValueFormat.TRANSPOSED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate tile-padded sparse-Piper value storage and per-tile metadata."""
    storage_sequence_length = padded_sequence_length(rows)
    storage_tiles = storage_sequence_length // TILE_ROWS
    return (
        template.new_empty(
            storage_format.shape(batch, heads, storage_sequence_length, head_dim),
            dtype=torch.int8,
        ),
        template.new_empty((batch, heads, storage_tiles, 1), dtype=torch.float32),
        template.new_empty((batch, heads, storage_tiles, head_dim), dtype=torch.bfloat16),
    )


@dataclass(frozen=True, slots=True)
class _PreparedAttentionOutput:
    """Shared launch state for one bounded attention-to-projection pipeline."""

    attention: _PreparedSparsePiperAttention
    sequence_length: int
    coarse_output: torch.Tensor | None
    coarse_gate: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _PreparedAttentionContext:
    """Global launch state for query chunks that have not been projected yet."""

    quantized_context: _quantized_dispatch._PreparedQuantizedSparsePiperContext
    sequence_length: int
    coarse_gate: torch.Tensor | None


def new_projected_output(
    attention_storage: torch.Tensor,
    sequence_length: int,
    output_features: int,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate the projected output after its boundary validates metadata."""
    return attention_storage.new_empty(
        (attention_storage.shape[0], sequence_length, output_features),
        dtype=output_dtype,
    )


def validate_attention_output(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    query_chunk_rows: int,
) -> int:
    """Validate the common boundary and return the flattened head width."""
    if attention_storage.ndim != 4:
        raise ValueError("fused sparse Piper output requires four-dimensional quantized storage")
    batch, heads, _storage_sequence_length, head_dim = attention_storage.shape
    if batch < 1 or head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError("fused sparse Piper output requires nonempty batches with D64/D128 heads")
    if isinstance(logical_sequence_length, bool) or not isinstance(logical_sequence_length, int):
        raise TypeError("fused sparse Piper logical sequence length must be an integer")
    if (
        isinstance(query_chunk_rows, bool)
        or not isinstance(query_chunk_rows, int)
        or query_chunk_rows < TILE_ROWS
        or query_chunk_rows % TILE_ROWS
    ):
        raise ValueError("fused sparse Piper query chunk rows must be a positive multiple of 64")
    return heads * head_dim


def prepare_attention_context(  # noqa: PLR0913, PLR0917
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
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    *,
    has_projected_coarse_gate: bool = False,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    start_head: int = 0,
    end_head: int | None = None,
) -> _PreparedAttentionContext:
    """Prepare global K/V state for bounded, independently projected Q chunks.

    The operands hold ``ratio_profile``'s query heads ``[start_head, end_head)``.
    """
    if coarse_gate is not None and has_projected_coarse_gate:
        raise ValueError("coarse gate cannot be both materialized and projected")
    has_coarse_gate = coarse_gate is not None or has_projected_coarse_gate
    if (coarse_scale is not None) != has_coarse_gate or (
        coarse_key_blocks is not None
    ) != has_coarse_gate:
        raise ValueError("coarse output requires a coarse gate, scale, and key block scope")
    quantized_context = _quantized_dispatch._prepare_quantized_sparse_piper_context(
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
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
        start_head=start_head,
        end_head=end_head,
    )
    return _PreparedAttentionContext(
        quantized_context=quantized_context,
        sequence_length=output_sequence_length(
            key,
            logical_sequence_length,
            block_lengths,
        ),
        coarse_gate=coarse_gate,
    )


def prepare_attention(  # noqa: PLR0913, PLR0917
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
    block_lengths: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    *,
    has_projected_coarse_gate: bool = False,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> _PreparedAttentionOutput:
    """Prepare fine routing and, when requested, one shared coarse result."""
    context = prepare_attention_context(
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
        has_projected_coarse_gate=has_projected_coarse_gate,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    prepared, coarse_output = _quantized_dispatch._prepare_quantized_sparse_piper_query(
        context.quantized_context,
        query,
        query_scale,
        query_summary,
        global_block_offset=0,
    )
    return _PreparedAttentionOutput(
        attention=prepared,
        sequence_length=context.sequence_length,
        coarse_output=coarse_output,
        coarse_gate=coarse_gate,
    )


def validate_coarse_gate(
    coarse_gate: torch.Tensor,
    *,
    batch: int,
    sequence_length: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate optional materialized coarse gate storage before launching chunks."""
    if (
        coarse_gate.shape != (batch, sequence_length, heads, head_dim)
        or coarse_gate.dtype is not dtype
        or coarse_gate.device != device
        or coarse_gate.layout is not torch.strided
        or coarse_gate.stride(-1) != 1
    ):
        raise ValueError("sparse Piper coarse gate must match the attention output")


def _run_chunked_attention_pipeline(  # noqa: PLR0913
    attention_storage: torch.Tensor,
    sequence_length: int,
    has_coarse_residual: bool,
    coarse_gate: torch.Tensor | None,
    output_features: int,
    query_chunk_rows: int,
    launch_chunk: output_pipeline.AttentionChunkLauncher,
    project_chunk: ChunkProjector | None,
    projector_tensors: Sequence[torch.Tensor],
    *,
    project_coarse_gate_chunk: CoarseGateChunkProjector | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    project_attention: AttentionProjector | None = None,
    reuse_output_for_attention: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate sparse coarse gates and supply the shared pipeline's output shape."""
    validate_output_dtype(output_dtype)
    if has_coarse_residual and (coarse_gate is None) == (project_coarse_gate_chunk is None):
        raise ValueError("coarse attention requires exactly one coarse gate source")
    if not has_coarse_residual and (
        coarse_gate is not None or project_coarse_gate_chunk is not None
    ):
        raise ValueError("coarse gate source requires coarse attention")
    batch, heads, _storage_sequence_length, head_dim = attention_storage.shape
    if coarse_gate is not None:
        validate_coarse_gate(
            coarse_gate,
            batch=batch,
            sequence_length=sequence_length,
            heads=heads,
            head_dim=head_dim,
            dtype=output_dtype,
            device=attention_storage.device,
        )

    return output_pipeline.run_chunked_attention_output(
        (batch, sequence_length, heads, head_dim),
        attention_storage.device,
        output_features,
        query_chunk_rows,
        launch_chunk,
        project_chunk,
        projector_tensors,
        auxiliary_input=coarse_gate,
        project_auxiliary_chunk=project_coarse_gate_chunk,
        auxiliary_pipeline_min_chunks=_MIN_PROJECTED_GATE_PIPELINE_CHUNKS,
        output_dtype=output_dtype,
        project_attention=project_attention,
        reuse_output_for_attention=reuse_output_for_attention,
        out=out,
    )


def run_chunked_attention_output(
    prepared: _PreparedAttentionOutput,
    output_features: int,
    query_chunk_rows: int,
    project_chunk: ChunkProjector | None,
    projector_tensors: Sequence[torch.Tensor],
    *,
    project_coarse_gate_chunk: CoarseGateChunkProjector | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    project_attention: AttentionProjector | None = None,
    reuse_output_for_attention: bool = False,
) -> torch.Tensor:
    """Pipeline a materialized Q boundary through bounded attention output."""
    prepared_attention = prepared.attention

    def launch_chunk(
        attention_chunk: torch.Tensor,
        start: int,
        rows: int,
        gate_chunk: torch.Tensor | None,
    ) -> None:
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            prepared_attention,
            attention_chunk.transpose(1, 2),
            query_block_offset=start // TILE_ROWS,
            query_block_count=(rows + TILE_ROWS - 1) // TILE_ROWS,
            coarse_output=prepared.coarse_output,
            coarse_gate=gate_chunk,
        )

    return _run_chunked_attention_pipeline(
        prepared_attention.query.data,
        prepared.sequence_length,
        prepared.coarse_output is not None,
        prepared.coarse_gate,
        output_features,
        query_chunk_rows,
        launch_chunk,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=project_coarse_gate_chunk,
        output_dtype=output_dtype,
        project_attention=project_attention,
        reuse_output_for_attention=reuse_output_for_attention,
    )


def launch_projected_query_chunk(
    prepared: _PreparedAttentionContext,
    project_query_chunk: QueryChunkProjector,
    global_start: int,
    rows: int,
    attention_chunk: torch.Tensor,
    *,
    coarse_gate: torch.Tensor | None = None,
) -> None:
    """Project, route, and launch sparse attention for one sequence window."""
    query, query_scale, query_summary = project_query_chunk(global_start, rows)
    local_attention, coarse_output = _quantized_dispatch._prepare_quantized_sparse_piper_query(
        prepared.quantized_context,
        query,
        query_scale,
        query_summary,
        global_block_offset=global_start // TILE_ROWS,
    )
    _quantized_dispatch._launch_quantized_sparse_piper_attention(
        local_attention,
        attention_chunk.transpose(1, 2),
        coarse_output=coarse_output,
        coarse_gate=coarse_gate,
    )


def run_chunked_projected_query_attention_output(  # noqa: PLR0913
    prepared: _PreparedAttentionContext,
    output_features: int,
    query_chunk_rows: int,
    project_query_chunk: QueryChunkProjector,
    project_chunk: ChunkProjector | None,
    projector_tensors: Sequence[torch.Tensor],
    *,
    project_coarse_gate_chunk: CoarseGateChunkProjector | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    project_attention: AttentionProjector | None = None,
    reuse_output_for_attention: bool = False,
    out: torch.Tensor | None = None,
    global_row_offset: int = 0,
    sequence_length: int | None = None,
) -> torch.Tensor:
    """Project, route, attend, and consume one bounded Q window at a time."""
    target_sequence_length = (
        prepared.sequence_length if sequence_length is None else sequence_length
    )

    def launch_chunk(
        attention_chunk: torch.Tensor,
        start: int,
        rows: int,
        gate_chunk: torch.Tensor | None,
    ) -> None:
        launch_projected_query_chunk(
            prepared,
            project_query_chunk,
            global_row_offset + start,
            rows,
            attention_chunk,
            coarse_gate=gate_chunk,
        )

    coarse_gate = (
        prepared.coarse_gate
        if prepared.coarse_gate is None
        or (global_row_offset == 0 and target_sequence_length == prepared.sequence_length)
        else prepared.coarse_gate[:, global_row_offset : global_row_offset + target_sequence_length]
    )
    return _run_chunked_attention_pipeline(
        prepared.quantized_context.kernel_context.key,
        target_sequence_length,
        prepared.quantized_context.pooled_value is not None,
        coarse_gate,
        output_features,
        query_chunk_rows,
        launch_chunk,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=project_coarse_gate_chunk,
        output_dtype=output_dtype,
        project_attention=project_attention,
        reuse_output_for_attention=reuse_output_for_attention,
        out=out,
    )


__all__ = [
    "DEFAULT_QUERY_CHUNK_ROWS",
    "allocate_key_storage",
    "allocate_query_storage",
    "allocate_value_storage",
    "fake_projected_output",
    "has_optional_operands",
    "launch_projected_query_chunk",
    "new_projected_output",
    "output_sequence_length",
    "prepare_attention",
    "prepare_attention_context",
    "run_chunked_attention_output",
    "run_chunked_projected_query_attention_output",
    "validate_attention_output",
    "validate_coarse_gate",
    "validate_optional_block_lengths",
    "validate_output_metadata",
    "validate_query_range",
    "validate_residual_add",
]
