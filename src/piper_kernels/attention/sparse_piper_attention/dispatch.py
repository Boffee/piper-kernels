"""Public orchestration for sparse Piper Attention."""

from __future__ import annotations

import math

import torch

from .dsa import (
    SparsePiperAttentionPlan,
    packed_dsa_routes_from_plan,
    prepare_dsa_route_plan,
)
from .reference import reference_sparse_piper_attention


def prepare_sparse_piper_attention_plan(
    keep_blocks: torch.Tensor,
    sparse_key_blocks: int,
    *,
    query_chunk_blocks: int = 384,
) -> SparsePiperAttentionPlan:
    """Prepare reusable per-head route-budget metadata.

    ``keep_blocks[h]`` is the number of sparse-prefix K64 tiles retained for
    head ``h``. The dense suffix is separate and is always included.
    """
    return prepare_dsa_route_plan(
        keep_blocks,
        key_block_count=sparse_key_blocks,
        query_chunk_blocks=query_chunk_blocks,
    )


def _validate_inputs(  # noqa: PLR0912
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SparsePiperAttentionPlan,
    *,
    suffix_start: int,
    valid_sequence_length: int | None,
    scale: float | None,
) -> tuple[int, float]:
    tensors = (query, key, value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must use [batch,sequence,heads,features]")
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("sparse Piper requires equal Q/K/V shapes")
    if query.dtype is not torch.bfloat16 or any(
        tensor.dtype is not query.dtype for tensor in tensors
    ):
        raise ValueError("sparse Piper Q/K/V must use bfloat16")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must share a device")
    if any(tensor.layout is not torch.strided or tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must have contiguous feature dimensions")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("sparse Piper is inference-only and does not support autograd")

    _batch, sequence, heads, head_dim = query.shape
    if head_dim != 128:
        raise ValueError("sparse Piper requires head_dim=128")
    if sequence < 64 or sequence % 64:
        raise ValueError("sparse Piper requires a K64-aligned physical sequence")
    if not 64 <= suffix_start <= sequence or suffix_start % 64:
        raise ValueError("suffix_start must leave a nonempty K64-aligned sparse prefix")
    valid_sequence = sequence if valid_sequence_length is None else valid_sequence_length
    if not suffix_start <= valid_sequence <= sequence:
        raise ValueError("valid_sequence_length must cover the sparse prefix and fit Q/K/V")
    if valid_sequence < sequence and valid_sequence <= sequence - 64:
        raise ValueError("sparse Piper supports padding only in the final K64 tile")
    if plan.keep_blocks.shape != (heads,):
        raise ValueError("the sparse Piper plan must contain one keep count per head")
    if plan.keep_blocks.device != query.device or plan.head_offsets.device != query.device:
        raise ValueError("the sparse Piper plan and Q/K/V must share a device")
    if plan.key_block_count != suffix_start // 64:
        raise ValueError("the sparse Piper plan must cover exactly the sparse-prefix K64 tiles")
    converted_scale = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(converted_scale) or converted_scale <= 0:
        raise ValueError("sparse Piper scale must be finite and positive")
    return valid_sequence, converted_scale


def sparse_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SparsePiperAttentionPlan,
    *,
    suffix_start: int,
    valid_sequence_length: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Route every query over a sparse K/V prefix and an always-dense suffix.

    Q/K/V are pre-tiled sequence-major ``[B,S,H,128]`` tensors. Rows before
    ``suffix_start`` form routeable K64 tiles. Rows from ``suffix_start`` to
    ``valid_sequence_length`` are included for every query. All selected rows
    participate in one softmax. The returned tensor retains the physical
    aligned sequence length; callers discard any padded tail.
    """
    valid_sequence, converted_scale = _validate_inputs(
        query,
        key,
        value,
        plan,
        suffix_start=suffix_start,
        valid_sequence_length=valid_sequence_length,
        scale=scale,
    )
    target_is_sm120 = query.device.type == "cuda" and (
        torch.cuda.get_device_capability(query.device) == (12, 0)
    )
    if target_is_sm120:
        return _sm120_sparse_piper_attention(
            query,
            key,
            value,
            plan.keep_blocks,
            plan.head_offsets,
            suffix_start,
            valid_sequence,
            converted_scale,
            plan.routes_per_query,
            plan.query_chunk_blocks,
        )
    return _run_sparse_piper_attention(
        query,
        key,
        value,
        plan,
        suffix_start=suffix_start,
        valid_sequence_length=valid_sequence,
        scale=converted_scale,
        target_is_sm120=False,
    )


def _run_sparse_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SparsePiperAttentionPlan,
    *,
    suffix_start: int,
    valid_sequence_length: int,
    scale: float,
    target_is_sm120: bool,
) -> torch.Tensor:
    """Execute validated sparse routing outside Dynamo tracing."""
    _batch, sequence, _heads, _head_dim = query.shape
    query_block_count = sequence // 64
    sparse_key_block_count = suffix_start // 64
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)
    query_blocks = query_head_major.unflatten(2, (query_block_count, 64))
    key_blocks = key_head_major[:, :, :suffix_start].unflatten(
        2,
        (sparse_key_block_count, 64),
    )
    query_valid_counts = None
    if valid_sequence_length < sequence:
        query_valid_counts = torch.full(
            (query_block_count,),
            64,
            device=query.device,
            dtype=torch.int32,
        )
        query_valid_counts[-1] = valid_sequence_length - (query_block_count - 1) * 64
    routes = packed_dsa_routes_from_plan(
        query_blocks,
        key_blocks,
        plan,
        query_valid_counts=query_valid_counts,
    )

    if not target_is_sm120:
        return reference_sparse_piper_attention(
            query,
            key,
            value,
            routes,
            suffix_start=suffix_start,
            valid_sequence_length=valid_sequence_length,
            scale=scale,
        )

    from .gluon import _launch_gluon_paired_routed_piper_attention  # noqa: PLC0415
    from .triton import (  # noqa: PLC0415
        _prepare_folded_tile_scaled_routed_piper_attention,
    )

    output = torch.empty_like(query)
    prepared = _prepare_folded_tile_scaled_routed_piper_attention(
        query_blocks,
        key_blocks,
        routes.indices,
        routes.keep_blocks,
        scale,
        valid_query_count=(None if valid_sequence_length == sequence else valid_sequence_length),
        route_head_offsets=routes.head_offsets,
        combined_key=key_head_major[:, :, :valid_sequence_length],
        combined_value=value_head_major[:, :, :valid_sequence_length],
        attention_output=output.transpose(1, 2),
    )
    _launch_gluon_paired_routed_piper_attention(prepared)
    return output


@torch.library.custom_op("piper_kernels::sparse_piper_attention", mutates_args=())
def _sm120_sparse_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keep_blocks: torch.Tensor,
    head_offsets: torch.Tensor,
    suffix_start: int,
    valid_sequence_length: int,
    scale: float,
    routes_per_query: int,
    query_chunk_blocks: int,
) -> torch.Tensor:
    plan = SparsePiperAttentionPlan(
        keep_blocks=keep_blocks,
        head_offsets=head_offsets,
        key_block_count=suffix_start // 64,
        routes_per_query=routes_per_query,
        query_chunk_blocks=query_chunk_blocks,
    )
    return _run_sparse_piper_attention(
        query,
        key,
        value,
        plan,
        suffix_start=suffix_start,
        valid_sequence_length=valid_sequence_length,
        scale=scale,
        target_is_sm120=True,
    )


@_sm120_sparse_piper_attention.register_fake
def _sm120_sparse_piper_attention_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _keep_blocks: torch.Tensor,
    _head_offsets: torch.Tensor,
    _suffix_start: int,
    _valid_sequence_length: int,
    _scale: float,
    _routes_per_query: int,
    _query_chunk_blocks: int,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
