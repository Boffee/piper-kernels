"""Internal sparse Piper orchestration for already-quantized operands."""

from __future__ import annotations

import torch

from ._budget import _resolve_route_layout
from .dsa import packed_dsa_routes_from_summaries

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
) -> torch.Tensor:
    if _prepare_sm120_quantized_attention is None or _launch_sm120_attention is None:
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
    batch, heads, _, head_dim = query.shape
    output = torch.empty(
        (batch, logical_sequence_length, heads, head_dim),
        device=query.device,
        dtype=torch.bfloat16,
    )
    prepared = _prepare_sm120_quantized_attention(
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
    )
    _launch_sm120_attention(prepared, output.transpose(1, 2))
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
) -> torch.Tensor:
    return query.new_empty(
        (query.shape[0], logical_sequence_length, query.shape[1], query.shape[3]),
        dtype=torch.bfloat16,
    )
