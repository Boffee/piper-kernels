"""Public orchestration for sparse Piper Attention."""

from __future__ import annotations

import math

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from .dsa import (
    SparsePiperAttentionPlan,
    packed_dsa_routes_from_plan,
    prepare_dsa_route_plan,
)
from .reference import reference_sparse_piper_attention

try:
    from .gluon import (
        _launch_gluon_paired_routed_piper_attention as _launch_sm120_attention,
    )
    from .triton import (
        _prepare_folded_tile_scaled_routed_piper_attention as _prepare_sm120_attention,
    )
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("triton"):
        raise
    _launch_sm120_attention = None
    _prepare_sm120_attention = None


def _supports_sm120(target: AcceleratorTarget) -> bool:
    return _launch_sm120_attention is not None and target.is_cuda_capability(12, 0)


def prepare_sparse_piper_attention_plan(
    keep_blocks: torch.Tensor,
    *,
    query_chunk_blocks: int = 384,
) -> SparsePiperAttentionPlan:
    """Prepare reusable per-head route-budget metadata.

    ``keep_blocks[h]`` is the number of sparse-prefix K64 tiles retained for
    head ``h``. The dense suffix is separate and is always included.
    """
    return prepare_dsa_route_plan(
        keep_blocks,
        query_chunk_blocks=query_chunk_blocks,
    )


def _validate_sparse_key_blocks(
    sparse_key_blocks: int,
    *,
    sequence_blocks: int,
    minimum_blocks: int,
) -> None:
    if isinstance(sparse_key_blocks, bool):
        raise TypeError("sparse_key_blocks must be an integer")
    if torch.compiler.is_compiling():
        torch._check(
            sparse_key_blocks >= minimum_blocks,
            lambda: "sparse_key_blocks cannot be smaller than a per-head route budget",
        )
        torch._check(
            sparse_key_blocks <= sequence_blocks,
            lambda: "sparse_key_blocks cannot exceed the sequence block count",
        )
        return
    if not isinstance(sparse_key_blocks, int):
        raise TypeError("sparse_key_blocks must be an integer")
    if not minimum_blocks <= sparse_key_blocks <= sequence_blocks:
        raise ValueError(
            "sparse_key_blocks must cover every per-head route budget and fit the sequence"
        )


def _validate_plan(
    plan: SparsePiperAttentionPlan,
    *,
    heads: int,
    device: torch.device,
) -> None:
    if plan.keep_blocks.shape != (heads,):
        raise ValueError("the sparse Piper plan must contain one keep count per head")
    if plan.head_offsets.shape != (heads + 1,):
        raise ValueError("the sparse Piper plan must contain one route offset per head boundary")
    if plan.keep_blocks.dtype is not torch.int32 or plan.head_offsets.dtype is not torch.int32:
        raise ValueError("the sparse Piper plan must use INT32 keep counts and route offsets")
    if plan.keep_blocks.device != device or plan.head_offsets.device != device:
        raise ValueError("the sparse Piper plan and Q/K/V must share a device")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SparsePiperAttentionPlan,
    *,
    sparse_key_blocks: int,
    scale: float | None,
) -> float:
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
        raise ValueError("sparse Piper requires a K64-aligned sequence")
    _validate_plan(plan, heads=heads, device=query.device)
    _validate_sparse_key_blocks(
        sparse_key_blocks,
        sequence_blocks=sequence // 64,
        minimum_blocks=plan.max_keep_blocks,
    )
    converted_scale = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(converted_scale) or converted_scale <= 0:
        raise ValueError("sparse Piper scale must be finite and positive")
    return converted_scale


def sparse_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SparsePiperAttentionPlan,
    *,
    sparse_key_blocks: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Route every query over a sparse K/V prefix and an always-dense suffix.

    Q/K/V are pre-tiled sequence-major ``[B,S,H,128]`` tensors. The first
    ``sparse_key_blocks`` K64 tiles form the routeable prefix. Every remaining
    K/V row is included for every query. All selected rows participate in one
    softmax.
    """
    converted_scale = _validate_inputs(
        query,
        key,
        value,
        plan,
        sparse_key_blocks=sparse_key_blocks,
        scale=scale,
    )
    target = AcceleratorTarget.from_device(query.device)
    if _supports_sm120(target):
        return _sm120_sparse_piper_attention(
            query,
            key,
            value,
            plan.keep_blocks,
            plan.head_offsets,
            sparse_key_blocks,
            converted_scale,
            plan.routes_per_query,
            plan.max_keep_blocks,
            plan.query_chunk_blocks,
        )
    return _run_sparse_piper_attention(
        query,
        key,
        value,
        plan,
        sparse_key_blocks=sparse_key_blocks,
        scale=converted_scale,
        target_is_sm120=False,
    )


def _run_sparse_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SparsePiperAttentionPlan,
    *,
    sparse_key_blocks: int,
    scale: float,
    target_is_sm120: bool,
) -> torch.Tensor:
    """Execute validated sparse routing outside Dynamo tracing."""
    _batch, sequence, _heads, _head_dim = query.shape
    query_block_count = sequence // 64
    sparse_key_rows = sparse_key_blocks * 64
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)
    query_blocks = query_head_major.unflatten(2, (query_block_count, 64))
    key_blocks = key_head_major[:, :, :sparse_key_rows].unflatten(
        2,
        (sparse_key_blocks, 64),
    )
    routes = packed_dsa_routes_from_plan(
        query_blocks,
        key_blocks,
        plan,
    )

    if not target_is_sm120:
        return reference_sparse_piper_attention(
            query,
            key,
            value,
            routes,
            sparse_key_blocks=sparse_key_blocks,
            scale=scale,
        )

    assert _launch_sm120_attention is not None
    assert _prepare_sm120_attention is not None
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    prepared = _prepare_sm120_attention(
        query_blocks,
        key_blocks,
        routes.indices,
        routes.keep_blocks,
        scale,
        route_head_offsets=routes.head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
        attention_output=output.transpose(1, 2),
    )
    _launch_sm120_attention(prepared)
    return output


@torch.library.custom_op("piper_kernels::sparse_piper_attention", mutates_args=())
def _sm120_sparse_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keep_blocks: torch.Tensor,
    head_offsets: torch.Tensor,
    sparse_key_blocks: int,
    scale: float,
    routes_per_query: int,
    max_keep_blocks: int,
    query_chunk_blocks: int,
) -> torch.Tensor:
    plan = SparsePiperAttentionPlan(
        keep_blocks=keep_blocks,
        head_offsets=head_offsets,
        routes_per_query=routes_per_query,
        max_keep_blocks=max_keep_blocks,
        query_chunk_blocks=query_chunk_blocks,
    )
    return _run_sparse_piper_attention(
        query,
        key,
        value,
        plan,
        sparse_key_blocks=sparse_key_blocks,
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
    _sparse_key_blocks: int,
    _scale: float,
    _routes_per_query: int,
    _max_keep_blocks: int,
    _query_chunk_blocks: int,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
