"""Internal sparse Piper orchestration for already-quantized operands."""

from __future__ import annotations

import torch

from .dsa import SparsePiperAttentionPlan, packed_dsa_routes_from_summaries

try:
    from .gluon import (
        _launch_gluon_paired_routed_piper_attention as _launch_sm120_attention,
    )
    from .triton import (
        _prepare_quantized_routed_piper_attention as _prepare_sm120_quantized_attention,
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
def _sm120_sparse_piper_attention_from_quantized(  # noqa: PLR0913, PLR0917
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
    keep_blocks: torch.Tensor,
    head_offsets: torch.Tensor,
    suffix_start: int,
    valid_sequence_length: int,
    routes_per_query: int,
    query_chunk_blocks: int,
) -> torch.Tensor:
    if _prepare_sm120_quantized_attention is None or _launch_sm120_attention is None:
        raise RuntimeError("quantized-input sparse Piper SM120 implementation is unavailable")
    plan = SparsePiperAttentionPlan(
        keep_blocks=keep_blocks,
        head_offsets=head_offsets,
        key_block_count=suffix_start // 64,
        routes_per_query=routes_per_query,
        query_chunk_blocks=query_chunk_blocks,
    )
    routes = packed_dsa_routes_from_summaries(
        query_summary,
        key_max[:, :, : plan.key_block_count],
        key_min[:, :, : plan.key_block_count],
        plan,
    )
    batch, heads, sequence_length, head_dim = query.shape
    output = torch.empty(
        (batch, sequence_length, heads, head_dim),
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
        suffix_start=suffix_start,
        valid_sequence_length=valid_sequence_length,
        routes_per_query=routes_per_query,
        attention_output=output.transpose(1, 2),
    )
    _launch_sm120_attention(prepared)
    return output


@_sm120_sparse_piper_attention_from_quantized.register_fake
def _sm120_sparse_piper_attention_from_quantized_fake(
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
    _keep_blocks: torch.Tensor,
    _head_offsets: torch.Tensor,
    _suffix_start: int,
    _valid_sequence_length: int,
    _routes_per_query: int,
    _query_chunk_blocks: int,
) -> torch.Tensor:
    return query.new_empty(
        (query.shape[0], query.shape[2], query.shape[1], query.shape[3]),
        dtype=torch.bfloat16,
    )
