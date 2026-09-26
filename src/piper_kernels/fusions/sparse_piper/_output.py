"""Sparse-attention adapters for the shared output pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS, TILE_ROWS
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch
from piper_kernels.attention.sparse_piper_attention._dtype import validate_output_dtype
from piper_kernels.fusions.attention import _output as output_pipeline

if TYPE_CHECKING:
    from piper_kernels.attention.sparse_piper_attention._prepared import (
        _PreparedSparsePiperAttention,
    )

DEFAULT_QUERY_CHUNK_ROWS = 4_096
_MIN_PROJECTED_GATE_PIPELINE_CHUNKS = 8

AttentionProjector = output_pipeline.AttentionProjector
ChunkProjector = output_pipeline.ChunkProjector
CoarseGateChunkProjector = output_pipeline.AuxiliaryChunkProjector
type QueryChunkProjector = Callable[[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

output_sequence_length = _quantized_dispatch._quantized_attention_output_sequence_length


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
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    output_features: int,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Allocate the projected compact or valid-front padded output shape."""
    validate_output_dtype(output_dtype)
    return attention_storage.new_empty(
        (
            attention_storage.shape[0],
            output_sequence_length(
                attention_storage,
                logical_sequence_length,
                block_lengths,
            ),
            output_features,
        ),
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
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    *,
    has_projected_coarse_gate: bool = False,
) -> _PreparedAttentionContext:
    """Prepare global K/V state for bounded, independently projected Q chunks."""
    if coarse_gate is not None and has_projected_coarse_gate:
        raise ValueError("coarse gate cannot be both materialized and projected")
    has_coarse_gate = coarse_gate is not None or has_projected_coarse_gate
    if (block_mean is not None) != has_coarse_gate:
        raise ValueError("block means and coarse gate must be supplied together")
    quantized_context = _quantized_dispatch._prepare_quantized_sparse_piper_context(
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
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
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
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    *,
    has_projected_coarse_gate: bool = False,
) -> _PreparedAttentionOutput:
    """Prepare fine routing and, when requested, one shared coarse result."""
    context = prepare_attention_context(
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
        has_projected_coarse_gate=has_projected_coarse_gate,
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


def source_files() -> tuple[str, ...]:
    """Include the shared pipeline in compiler-pass cache invalidation."""
    return tuple(path for path in (__file__, output_pipeline.__file__) if path is not None)


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

    return output_pipeline.run_chunked_attention_output(
        (
            attention_storage.shape[0],
            sequence_length,
            attention_storage.shape[1],
            attention_storage.shape[3],
        ),
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
    )


def run_chunked_projected_query_attention_output(
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project, route, attend, and consume one bounded Q window at a time."""

    def launch_chunk(
        attention_chunk: torch.Tensor,
        start: int,
        rows: int,
        gate_chunk: torch.Tensor | None,
    ) -> None:
        query, query_scale, query_summary = project_query_chunk(start, rows)
        local_attention, coarse_output = _quantized_dispatch._prepare_quantized_sparse_piper_query(
            prepared.quantized_context,
            query,
            query_scale,
            query_summary,
            global_block_offset=start // TILE_ROWS,
        )
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            local_attention,
            attention_chunk.transpose(1, 2),
            coarse_output=coarse_output,
            coarse_gate=gate_chunk,
        )

    return _run_chunked_attention_pipeline(
        prepared.quantized_context.kernel_context.key,
        prepared.sequence_length,
        prepared.quantized_context.pooled_value is not None,
        prepared.coarse_gate,
        output_features,
        query_chunk_rows,
        launch_chunk,
        project_chunk,
        projector_tensors,
        project_coarse_gate_chunk=project_coarse_gate_chunk,
        output_dtype=output_dtype,
        project_attention=project_attention,
        out=out,
    )


__all__ = [
    "DEFAULT_QUERY_CHUNK_ROWS",
    "new_projected_output",
    "output_sequence_length",
    "prepare_attention",
    "prepare_attention_context",
    "run_chunked_attention_output",
    "run_chunked_projected_query_attention_output",
    "source_files",
    "validate_attention_output",
]
