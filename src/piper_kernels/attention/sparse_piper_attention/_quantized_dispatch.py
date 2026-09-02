"""Internal sparse Piper orchestration for already-quantized operands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._budget import _resolve_route_layout
from ._routes import _MEAN_POOL_ROUTING, PackedRoutes, validate_routing_mode
from .coarse import apply_coarse_attention_residual
from .dsa import (
    packed_dsa_routes_and_coarse_from_summaries,
    packed_dsa_routes_from_summaries,
)
from .mean_pool import (
    packed_mean_pool_routes_and_coarse_from_summaries,
    packed_mean_pool_routes_from_summaries,
)

if TYPE_CHECKING:
    from .triton import _PreparedSparsePiperAttention

try:
    from .gluon import (
        _launch_sparse_piper_attention as _launch_sm120_attention,
    )
    from .triton import (
        _prepare_sparse_piper_attention_from_quantized as _prepare_sm120_quantized_attention,
    )
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("triton"):
        raise
    _launch_sm120_attention = None
    _prepare_sm120_quantized_attention = None


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
) -> _PreparedSparsePiperAttention:
    """Build the validated SM120 launch state for quantized sparse Piper."""
    layout = _resolve_route_layout(
        tuple(head_keep_ratio_units),
        sparse_key_blocks,
        query.device,
    )
    validate_routing_mode(routing_mode)
    if routing_mode == _MEAN_POOL_ROUTING:
        routes = packed_mean_pool_routes_from_summaries(
            query_summary,
            key_summary[:, :, :sparse_key_blocks],
            layout,
        )
    else:
        routes = packed_dsa_routes_from_summaries(
            query_summary,
            key_summary[:, :, :sparse_key_blocks],
            key_aux[:, :, :sparse_key_blocks],
            layout,
        )
    return _prepare_quantized_sparse_piper_attention_from_routes(
        query,
        query_scale,
        key,
        key_scale,
        value,
        value_scale_multiplier,
        value_mean,
        routes,
        sparse_key_blocks,
        logical_sequence_length,
        block_lengths,
    )


def _prepare_quantized_sparse_piper_attention_from_routes(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    routes: PackedRoutes,
    sparse_key_blocks: int,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
) -> _PreparedSparsePiperAttention:
    """Build validated SM120 launch state from an already-resolved route policy."""
    if _prepare_sm120_quantized_attention is None:
        raise RuntimeError("quantized-input sparse Piper SM120 implementation is unavailable")
    return _prepare_sm120_quantized_attention(
        query,
        query_scale,
        key,
        key_scale,
        value,
        value_scale_multiplier,
        value_mean,
        routes.indices,
        routes.keep_blocks,
        routes.head_offsets,
        sparse_key_blocks=sparse_key_blocks,
        routes_per_query=routes.indices.shape[2],
        logical_sequence_length=logical_sequence_length,
        block_lengths=block_lengths,
    )


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
) -> tuple[_PreparedSparsePiperAttention, torch.Tensor]:
    """Prepare fine attention and coarse output from the same routing scores."""
    layout = _resolve_route_layout(
        tuple(head_keep_ratio_units),
        sparse_key_blocks,
        query.device,
    )
    validate_routing_mode(routing_mode)
    sparse_block_mean = block_mean[:, :, :sparse_key_blocks]
    if routing_mode == _MEAN_POOL_ROUTING:
        routed = packed_mean_pool_routes_and_coarse_from_summaries(
            query_summary,
            key_summary[:, :, :sparse_key_blocks],
            sparse_block_mean,
            layout,
            coarse_scale=coarse_scale,
        )
    else:
        routed = packed_dsa_routes_and_coarse_from_summaries(
            query_summary,
            key_summary[:, :, :sparse_key_blocks],
            key_aux[:, :, :sparse_key_blocks],
            sparse_block_mean,
            layout,
            coarse_scale=coarse_scale,
        )
    prepared = _prepare_quantized_sparse_piper_attention_from_routes(
        query,
        query_scale,
        key,
        key_scale,
        value,
        value_scale_multiplier,
        value_mean,
        routed.routes,
        sparse_key_blocks,
        logical_sequence_length,
        block_lengths,
    )
    return prepared, routed.coarse_output


def _new_quantized_attention_output(
    query: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
) -> torch.Tensor:
    output_sequence_length = (
        query.shape[2] if block_lengths is not None else logical_sequence_length
    )
    return query.new_empty(
        (query.shape[0], output_sequence_length, query.shape[1], query.shape[3]),
        dtype=torch.bfloat16,
    )


def _launch_quantized_sparse_piper_attention(
    prepared: _PreparedSparsePiperAttention,
    output: torch.Tensor,
    *,
    query_block_offset: int = 0,
    query_block_count: int | None = None,
) -> None:
    """Launch a prepared quantized sparse-Piper query range."""
    if _launch_sm120_attention is None:
        raise RuntimeError("quantized-input sparse Piper SM120 implementation is unavailable")
    _launch_sm120_attention(
        prepared,
        output,
        query_block_offset=query_block_offset,
        query_block_count=query_block_count,
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
) -> torch.Tensor:
    """Run quantized sparse Piper with compact or internally padded K64 storage.

    ``block_lengths`` is trusted layout metadata with one INT32 value per
    physical K64 block. Each value must be in ``[1, 64]``, valid tokens must
    occupy the front of the block, and the values must sum to
    ``logical_sequence_length``. Supplying it returns the full padded query
    storage; outputs for padded query rows are unspecified and must be removed
    by the caller's layout gather.
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
    compression_gate: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run quantized sparse attention plus a gated block-level residual."""
    fine_output = _new_quantized_attention_output(
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
    )
    _launch_quantized_sparse_piper_attention(prepared, fine_output.transpose(1, 2))
    return apply_coarse_attention_residual(
        fine_output,
        coarse_output,
        compression_gate,
    )


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
    _compression_gate: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _routing_mode: int,
    _coarse_scale: float,
    _block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    return _new_quantized_attention_output(
        query,
        logical_sequence_length,
        _block_lengths,
    )
