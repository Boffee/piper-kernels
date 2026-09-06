"""Internal sparse Piper orchestration for already-quantized operands."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _backend
from ._budget import _resolve_route_layout, _ResolvedRouteLayout
from ._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
    _PreparedSparsePiperContext,
)
from ._routing import (
    packed_routes_and_coarse_from_summaries,
    packed_routes_from_summaries,
)
from ._routing_modes import (
    validate_routing_mode,
)
from .coarse import _resolve_coarse_key_blocks


@dataclass(frozen=True, slots=True)
class _PreparedQuantizedSparsePiperContext:
    """Global sparse-attention state used by independently produced Q chunks."""

    kernel_context: _PreparedSparsePiperContext
    route_layout: _ResolvedRouteLayout
    key_summary: torch.Tensor
    key_aux: torch.Tensor
    routing_mode: int
    pooled_value: torch.Tensor | None = None
    coarse_scale: float | None = None


def _prepare_quantized_sparse_piper_context(  # noqa: PLR0913, PLR0917
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
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
) -> _PreparedQuantizedSparsePiperContext:
    """Prepare global K/V and routing state without requiring materialized Q."""
    _backend.require_attention_backend(key)
    layout = _resolve_route_layout(
        tuple(head_keep_ratio_units),
        sparse_key_blocks,
        key.device,
    )
    validate_routing_mode(routing_mode)
    pooled_value = None
    if block_mean is None:
        if coarse_scale is not None or coarse_key_blocks is not None:
            raise ValueError("coarse output metadata requires block means")
        route_key_blocks = sparse_key_blocks
    else:
        if coarse_scale is None:
            raise ValueError("coarse output requires a scale")
        route_key_blocks = _resolve_coarse_key_blocks(
            sparse_key_blocks,
            coarse_key_blocks,
            available_coarse_key_blocks=block_mean.shape[2],
        )
        pooled_value = block_mean[:, :, :route_key_blocks]
    kernel_context = _prepare_sparse_piper_context_from_quantized(
        key,
        key_scale,
        value,
        value_scale_multiplier,
        value_mean,
        layout.head_keep_blocks,
        layout.route_head_offsets,
        sparse_key_blocks=sparse_key_blocks,
        routes_per_query=layout.routes_per_query,
        logical_sequence_length=logical_sequence_length,
        block_lengths=block_lengths,
        sparse_query_blocks=sparse_query_blocks,
    )
    return _PreparedQuantizedSparsePiperContext(
        kernel_context=kernel_context,
        route_layout=layout,
        key_summary=key_summary[:, :, :route_key_blocks],
        key_aux=key_aux[:, :, :route_key_blocks],
        routing_mode=routing_mode,
        pooled_value=pooled_value,
        coarse_scale=coarse_scale,
    )


def _prepare_quantized_sparse_piper_query(
    context: _PreparedQuantizedSparsePiperContext,
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    *,
    global_block_offset: int,
) -> tuple[_PreparedSparsePiperAttention, torch.Tensor | None]:
    """Route and prepare one compact Q chunk against global K/V state."""
    if context.pooled_value is None:
        routed = packed_routes_from_summaries(
            query_summary,
            context.key_summary,
            context.key_aux,
            context.route_layout,
            context.routing_mode,
        )
        coarse_output = None
    else:
        assert context.coarse_scale is not None
        routed_with_coarse = packed_routes_and_coarse_from_summaries(
            query_summary,
            context.key_summary,
            context.key_aux,
            context.pooled_value,
            context.route_layout,
            sparse_key_blocks=context.kernel_context.sparse_key_blocks,
            coarse_scale=context.coarse_scale,
            routing_mode=context.routing_mode,
        )
        routed = routed_with_coarse.routes
        coarse_output = routed_with_coarse.coarse_output
    return (
        _PreparedSparsePiperAttention(
            context=context.kernel_context,
            query=_prepare_sparse_piper_query_from_quantized(
                query,
                query_scale,
                routed.indices,
                context.kernel_context,
                global_block_offset=global_block_offset,
            ),
        ),
        coarse_output,
    )


def _prepare_quantized_sparse_piper_attention(  # noqa: PLR0913, PLR0917
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
    sparse_query_blocks: int | None = None,
) -> _PreparedSparsePiperAttention:
    """Build validated shared launch state for quantized sparse Piper."""
    context = _prepare_quantized_sparse_piper_context(
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
        sparse_query_blocks=sparse_query_blocks,
    )
    return _prepare_quantized_sparse_piper_query(
        context,
        query,
        query_scale,
        query_summary,
        global_block_offset=0,
    )[0]


def _prepare_quantized_sparse_piper_attention_with_coarse(  # noqa: PLR0913, PLR0917
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
    block_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
) -> tuple[_PreparedSparsePiperAttention, torch.Tensor]:
    """Prepare fine routes and a potentially wider coarse K/V prefix."""
    context = _prepare_quantized_sparse_piper_context(
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
    prepared, coarse_output = _prepare_quantized_sparse_piper_query(
        context,
        query,
        query_scale,
        query_summary,
        global_block_offset=0,
    )
    assert coarse_output is not None
    return prepared, coarse_output


def _quantized_attention_output_sequence_length(
    query: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
) -> int:
    """Resolve compact logical rows or valid-front padded physical rows."""
    return query.shape[2] if block_lengths is not None else logical_sequence_length


def _new_quantized_attention_output(
    query: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
) -> torch.Tensor:
    return query.new_empty(
        (
            query.shape[0],
            _quantized_attention_output_sequence_length(
                query,
                logical_sequence_length,
                block_lengths,
            ),
            query.shape[1],
            query.shape[3],
        ),
        dtype=torch.bfloat16,
    )


def _launch_quantized_sparse_piper_attention(
    prepared: _PreparedSparsePiperAttention,
    output: torch.Tensor,
    *,
    query_block_offset: int = 0,
    query_block_count: int | None = None,
    coarse_output: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
) -> None:
    """Launch a prepared quantized sparse-Piper query range."""
    backend = _backend.require_attention_backend(prepared.query.data)
    backend.launch(
        prepared,
        output,
        query_block_offset=query_block_offset,
        query_block_count=query_block_count,
        coarse_output=coarse_output,
        coarse_gate=coarse_gate,
    )


@torch.library.custom_op(
    "piper_kernels::sparse_piper_attention_from_quantized",
    mutates_args=(),
)
def _sparse_piper_attention_from_quantized_op(  # noqa: PLR0913, PLR0917
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
    sparse_query_blocks: int | None = None,
) -> torch.Tensor:
    """Run quantized sparse Piper with compact or internally padded K64 storage.

    ``block_lengths`` is trusted layout metadata with one INT32 value per
    physical K64 block. Each value must be in ``[1, 64]``, valid tokens must
    occupy the front of the block, and the metadata becomes the source of
    token validity instead of ``logical_sequence_length``. Supplying it returns
    the full padded query storage; outputs for padded query rows are unspecified
    and must be removed by the caller's layout gather. ``sparse_query_blocks``
    optionally limits route use to the leading query blocks; the suffix is dense.
    """
    output = _new_quantized_attention_output(
        query,
        logical_sequence_length,
        block_lengths,
    )
    prepared = _prepare_quantized_sparse_piper_attention(
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
        sparse_query_blocks,
    )
    _launch_quantized_sparse_piper_attention(prepared, output.transpose(1, 2))
    return output


@torch.library.custom_op(
    "piper_kernels::sparse_piper_attention_with_coarse_residual_from_quantized",
    mutates_args=(),
)
def _sparse_piper_attention_with_coarse_residual_from_quantized_op(  # noqa: PLR0913, PLR0917
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
    block_mean: torch.Tensor,
    coarse_gate: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
) -> torch.Tensor:
    """Run quantized sparse attention plus a gated, independently scoped residual."""
    output = _new_quantized_attention_output(
        query,
        logical_sequence_length,
        block_lengths,
    )
    prepared, coarse_output = _prepare_quantized_sparse_piper_attention_with_coarse(
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
        block_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
        routing_mode,
        coarse_scale,
        block_lengths,
        coarse_key_blocks,
        sparse_query_blocks,
    )
    _launch_quantized_sparse_piper_attention(
        prepared,
        output.transpose(1, 2),
        coarse_output=coarse_output,
        coarse_gate=coarse_gate,
    )
    return output


@_sparse_piper_attention_from_quantized_op.register_fake
def _sparse_piper_attention_from_quantized_op_fake(
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
    _block_lengths: torch.Tensor | None = None,
    _sparse_query_blocks: int | None = None,
) -> torch.Tensor:
    return _new_quantized_attention_output(
        query,
        logical_sequence_length,
        _block_lengths,
    )


@_sparse_piper_attention_with_coarse_residual_from_quantized_op.register_fake
def _sparse_piper_attention_with_coarse_residual_from_quantized_op_fake(
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
    _block_mean: torch.Tensor,
    _coarse_gate: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _routing_mode: int,
    _coarse_scale: float,
    _block_lengths: torch.Tensor | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
) -> torch.Tensor:
    return _new_quantized_attention_output(
        query,
        logical_sequence_length,
        _block_lengths,
    )
