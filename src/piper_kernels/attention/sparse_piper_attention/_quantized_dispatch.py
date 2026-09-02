"""Internal sparse Piper orchestration for already-quantized operands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._budget import _resolve_route_layout
from .dsa import packed_dsa_routes_from_summaries

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
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
) -> _PreparedSparsePiperAttention:
    """Build the validated SM120 launch state for quantized sparse Piper."""
    if _prepare_sm120_quantized_attention is None:
        raise RuntimeError("quantized-input sparse Piper SM120 implementation is unavailable")
    layout = _resolve_route_layout(
        tuple(head_keep_ratio_units),
        sparse_key_blocks,
        query.device,
    )
    routes = packed_dsa_routes_from_summaries(
        query_summary,
        key_max[:, :, :sparse_key_blocks],
        key_min[:, :, :sparse_key_blocks],
        layout,
    )
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
        routes_per_query=layout.routes_per_query,
        logical_sequence_length=logical_sequence_length,
        block_lengths=block_lengths,
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
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
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
    batch, heads, _, head_dim = query.shape
    output_sequence_length = (
        query.shape[2] if block_lengths is not None else logical_sequence_length
    )
    output = torch.empty(
        (batch, output_sequence_length, heads, head_dim),
        device=query.device,
        dtype=torch.bfloat16,
    )
    prepared = _prepare_quantized_sparse_piper_attention(
        query,
        query_scale,
        query_summary,
        key,
        key_scale,
        key_max,
        key_min,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
        block_lengths,
    )
    _launch_quantized_sparse_piper_attention(prepared, output.transpose(1, 2))
    return output


@_sparse_piper_attention_from_quantized_op.register_fake
def _sparse_piper_attention_from_quantized_op_fake(
    query: torch.Tensor,
    _query_scale: torch.Tensor,
    _query_summary: torch.Tensor,
    _key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_max: torch.Tensor,
    _key_min: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    output_sequence_length = (
        query.shape[2] if _block_lengths is not None else logical_sequence_length
    )
    return query.new_empty(
        (query.shape[0], output_sequence_length, query.shape[1], query.shape[3]),
        dtype=torch.bfloat16,
    )
