"""INT8 preparation for the selected sparse Piper SM120 kernel."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_kernels,
)
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS
from piper_kernels.attention.piper_attention import _quantization as piper_quantization

from ._block_layout import valid_block_rows, validate_sparse_query_blocks

_BLOCK_M = 64
_BLOCK_N = 64
_HEAD_DIM = 128


@triton.jit
def _quantize_value_per_tile_kernel(
    value_ptr,
    value_mean_ptr,
    value_scale_ptr,
    output_ptr,
    block_lengths_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_od,
    stride_ok,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    mask_block_lengths: tl.constexpr,
):
    """Quantize one D128 K64 storage tile while masking the logical V tail."""
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid_rows = offsets_n < key_length
    if mask_block_lengths:
        valid_rows &= offsets_n - key_block * block_n < tl.load(block_lengths_ptr + key_block)
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    ).to(tl.float32)
    value_mean = tl.load(value_mean_ptr + batch_head * head_dim + offsets_d)
    quantized, value_scale_multiplier = sparse_piper_kernels.quantize_value_tile(  # pyright: ignore[reportGeneralTypeIssues]
        tl.reshape(value, (block_n, 1, head_dim)),
        tl.reshape(value_mean, (1, head_dim)),
        valid_rows,
        1,
        head_dim,
        block_n,
        block_n,
    )
    quantized = tl.reshape(quantized, (block_n, head_dim))
    value_scale_multiplier = tl.reshape(value_scale_multiplier, ())
    tile_count = tl.cdiv(key_length, block_n)
    tl.store(
        value_scale_ptr + batch_head * tile_count + key_block,
        value_scale_multiplier,
    )
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_d[None, :] * stride_od
        + offsets_n[:, None] * stride_ok,
        quantized,
    )


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperContext:
    """Sequence-global K/V storage and sparse-attention policy metadata."""

    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_mean: torch.Tensor
    route_head_offsets: torch.Tensor
    head_keep_blocks: torch.Tensor
    routes_per_query: int
    block_lengths: torch.Tensor | None
    sparse_key_blocks: int
    sparse_query_blocks: int | None
    logical_sequence_length: int


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperQuery:
    """Query-local quantized storage and routes at one global block offset."""

    data: torch.Tensor
    scale: torch.Tensor
    routes: torch.Tensor
    global_block_offset: int = 0


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperAttention:
    """Sequence-global context paired with one full or local query range."""

    context: _PreparedSparsePiperContext
    query: _PreparedSparsePiperQuery


def _prepare_sparse_piper_attention(
    query: torch.Tensor,
    routes: torch.Tensor,
    head_keep_blocks: torch.Tensor,
    scale: float,
    *,
    sparse_key_blocks: int,
    route_head_offsets: torch.Tensor,
    combined_key: torch.Tensor,
    combined_value: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
) -> _PreparedSparsePiperAttention:
    """Prepare grouped Q/K and one folded V scale per logical K64 tile."""
    if (
        combined_key.shape != query.shape
        or combined_value.shape != combined_key.shape
        or not 1 <= sparse_key_blocks <= query.shape[2] // _BLOCK_N
    ):
        raise ValueError("combined Q/K/V and the sparse-prefix K64 count must agree")
    if combined_key.stride(-1) != 1 or combined_value.stride(-1) != 1:
        raise ValueError("combined K/V feature dimensions must be contiguous")

    batch, heads, logical_sequence_length, head_dim = query.shape
    if block_lengths is not None and (
        logical_sequence_length % _BLOCK_N
        or block_lengths.shape != (logical_sequence_length // _BLOCK_N,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != query.device
        or not block_lengths.is_contiguous()
    ):
        raise ValueError("padded sparse Piper requires one contiguous device INT32 length per K64")
    if block_lengths is not None:
        valid_rows = valid_block_rows(block_lengths).reshape(-1)
        valid_rows = valid_rows[None, None, :, None]
        query = torch.where(valid_rows, query, 0)
        combined_key = torch.where(valid_rows, combined_key, 0)
        combined_value = torch.where(valid_rows, combined_value, 0)
    tile_count = (logical_sequence_length + _BLOCK_N - 1) // _BLOCK_N
    storage_sequence_length = tile_count * _BLOCK_N
    validate_sparse_query_blocks(
        sparse_query_blocks,
        query_blocks=tile_count,
        context="sparse Piper",
    )

    key_mean, value_mean = piper_quantization.compute_kv_means(
        combined_key,
        combined_value,
        is_causal=False,
    )
    prepared_qk = qk_quantization.prepare_query_key(
        query,
        combined_key,
        key_mean,
        scale,
        grouped=True,
        storage_key_length=storage_sequence_length,
        storage_query_length=storage_sequence_length,
    )
    value_int8 = torch.empty(
        (batch, heads, head_dim, storage_sequence_length),
        device=combined_value.device,
        dtype=torch.int8,
    )
    value_scale_multiplier = torch.empty(
        (batch, heads, tile_count, 1),
        device=combined_value.device,
        dtype=torch.float32,
    )

    with device_context(query.device):
        _quantize_value_per_tile_kernel[(tile_count, batch * heads)](
            combined_value,
            value_mean,
            value_scale_multiplier,
            value_int8,
            block_lengths if block_lengths is not None else value_mean,
            logical_sequence_length,
            combined_value.stride(0),
            combined_value.stride(1),
            combined_value.stride(2),
            value_int8.stride(0),
            value_int8.stride(1),
            value_int8.stride(2),
            value_int8.stride(3),
            heads=heads,
            head_dim=head_dim,
            block_n=_BLOCK_N,
            mask_block_lengths=block_lengths is not None,
            num_warps=4,
        )

        context = _PreparedSparsePiperContext(
            key=prepared_qk.key,
            value=value_int8,
            key_scale=prepared_qk.key_scale,
            value_scale_multiplier=value_scale_multiplier,
            value_mean=value_mean,
            route_head_offsets=route_head_offsets,
            head_keep_blocks=head_keep_blocks,
            routes_per_query=routes.shape[2],
            block_lengths=block_lengths,
            sparse_key_blocks=sparse_key_blocks,
            sparse_query_blocks=sparse_query_blocks,
            logical_sequence_length=logical_sequence_length,
        )
        return _PreparedSparsePiperAttention(
            context=context,
            query=_prepare_sparse_piper_query_from_quantized(
                prepared_qk.query,
                prepared_qk.query_scale,
                routes,
                context,
            ),
        )


def _prepare_sparse_piper_context_from_quantized(  # noqa: PLR0912
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_blocks: torch.Tensor,
    route_head_offsets: torch.Tensor,
    *,
    sparse_key_blocks: int,
    routes_per_query: int,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
) -> _PreparedSparsePiperContext:
    """Validate sequence-global quantized K/V storage and routing policy.

    ``block_lengths`` opts into internally padded K64 storage. Each entry gives
    the valid prefix length in ``[1, 64]`` of one physical block and supersedes
    ``logical_sequence_length`` for masking. Without it, storage is the padded
    form of the compact logical sequence with at most one ragged tail.
    """
    if key.ndim != 4 or key.dtype is not torch.int8:
        raise ValueError(
            "quantized sparse Piper K must be [batch,heads,storage_sequence,D128] INT8"
        )
    batch, heads, storage_sequence_length, head_dim = key.shape
    if (
        head_dim != _HEAD_DIM
        or storage_sequence_length < _BLOCK_M
        or storage_sequence_length % _BLOCK_M
    ):
        raise ValueError("quantized sparse Piper requires K64-aligned D128 query storage")
    if block_lengths is None:
        if (
            logical_sequence_length < _BLOCK_M
            or logical_sequence_length > storage_sequence_length
            or (logical_sequence_length + _BLOCK_M - 1) // _BLOCK_M * _BLOCK_M
            != storage_sequence_length
        ):
            raise ValueError("quantized sparse Piper storage must be the padded logical sequence")
    elif not 1 <= logical_sequence_length <= storage_sequence_length:
        raise ValueError("quantized sparse Piper logical length must fit block-length storage")
    if (
        value.shape != (batch, heads, head_dim, storage_sequence_length)
        or value.dtype is not torch.int8
    ):
        raise ValueError(
            "quantized sparse Piper V must be transposed INT8 [B,H,D,storage_sequence]"
        )
    tile_count = storage_sequence_length // _BLOCK_N
    if key_scale.shape != (batch, heads, tile_count):
        raise ValueError("quantized sparse Piper K scales must contain one value per K64")
    if value_scale_multiplier.shape != (batch, heads, tile_count, 1):
        raise ValueError("quantized sparse Piper V scales must contain one value per K64")
    if value_mean.shape != (batch, heads, head_dim):
        raise ValueError("quantized sparse Piper V mean must be [batch,heads,D128]")
    # Layout construction owns the value-range invariants documented
    # above. Inspecting CUDA values here would add a validation kernel or a host
    # synchronization to every launch; only launch-critical tensor properties
    # are checked on this hot path.
    if block_lengths is not None and (
        block_lengths.shape != (tile_count,) or block_lengths.dtype is not torch.int32
    ):
        raise ValueError("quantized sparse Piper block lengths must be one INT32 value per K64")
    scales = key_scale, value_scale_multiplier, value_mean
    if any(scale.dtype is not torch.float32 for scale in scales):
        raise ValueError("quantized sparse Piper scales and V mean must use FP32")
    tensors = (
        key,
        value,
        *scales,
        head_keep_blocks,
        route_head_offsets,
        *((block_lengths,) if block_lengths is not None else ()),
    )
    if any(tensor.device != key.device for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must share a device")
    if any(tensor.layout is not torch.strided or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must be contiguous strided tensors")
    if not 1 <= sparse_key_blocks <= tile_count:
        raise ValueError("quantized sparse Piper prefix must fit the K64 tile count")
    total_query_blocks = (
        tile_count
        if block_lengths is not None
        else (logical_sequence_length + _BLOCK_M - 1) // _BLOCK_M
    )
    validate_sparse_query_blocks(
        sparse_query_blocks,
        query_blocks=total_query_blocks,
        context="quantized sparse Piper",
    )
    if head_keep_blocks.shape != (heads,) or head_keep_blocks.dtype is not torch.int32:
        raise ValueError("quantized sparse Piper head keep blocks must be one INT32 value per head")
    if route_head_offsets.shape != (heads + 1,) or route_head_offsets.dtype is not torch.int32:
        raise ValueError("quantized sparse Piper route offsets must be an INT32 head vector")

    return _PreparedSparsePiperContext(
        key=key,
        value=value,
        key_scale=key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_mean=value_mean,
        route_head_offsets=route_head_offsets,
        head_keep_blocks=head_keep_blocks,
        routes_per_query=routes_per_query,
        block_lengths=block_lengths,
        sparse_key_blocks=sparse_key_blocks,
        sparse_query_blocks=sparse_query_blocks,
        logical_sequence_length=logical_sequence_length,
    )


def _prepare_sparse_piper_query_from_quantized(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    routes: torch.Tensor,
    context: _PreparedSparsePiperContext,
    *,
    global_block_offset: int = 0,
) -> _PreparedSparsePiperQuery:
    """Validate query-local quantized storage and locate it globally."""
    if query.ndim != 4 or query.dtype is not torch.int8:
        raise ValueError(
            "quantized sparse Piper Q must be [batch,heads,storage_sequence,D128] INT8"
        )
    batch, heads, storage_sequence_length, head_dim = query.shape
    if (
        query.shape[:2] != context.key.shape[:2]
        or head_dim != _HEAD_DIM
        or storage_sequence_length < _BLOCK_M
        or storage_sequence_length % _BLOCK_M
    ):
        raise ValueError("quantized sparse Piper requires compatible K64-aligned D128 Q storage")
    query_block_count = storage_sequence_length // _BLOCK_M
    total_query_blocks = context.key.shape[2] // _BLOCK_N
    if (
        isinstance(global_block_offset, bool)
        or not isinstance(global_block_offset, int)
        or global_block_offset < 0
        or global_block_offset + query_block_count > total_query_blocks
    ):
        raise ValueError("quantized sparse Piper Q storage must fit the global sequence")
    if query_scale.shape != (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS):
        raise ValueError("quantized sparse Piper Q scales must contain one value per Q32")
    if routes.shape != (batch, query_block_count, context.routes_per_query):
        raise ValueError("quantized sparse Piper routes must match batch/query/packed budgets")
    if query_scale.dtype is not torch.float32:
        raise ValueError("quantized sparse Piper Q scales must use FP32")
    if routes.dtype is not torch.uint16:
        raise ValueError("quantized sparse Piper routes must use UINT16")
    tensors = query, query_scale, routes
    if any(tensor.device != context.key.device for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must share a device")
    if any(tensor.layout is not torch.strided or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must be contiguous strided tensors")
    return _PreparedSparsePiperQuery(
        data=query,
        scale=query_scale,
        routes=routes,
        global_block_offset=global_block_offset,
    )


def _prepare_sparse_piper_attention_from_quantized(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    routes: torch.Tensor,
    head_keep_blocks: torch.Tensor,
    route_head_offsets: torch.Tensor,
    *,
    sparse_key_blocks: int,
    routes_per_query: int,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
    query_block_offset: int = 0,
) -> _PreparedSparsePiperAttention:
    """Construct one full or local query state over shared quantized K/V."""
    context = _prepare_sparse_piper_context_from_quantized(
        key,
        key_scale,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_blocks,
        route_head_offsets,
        sparse_key_blocks=sparse_key_blocks,
        routes_per_query=routes_per_query,
        logical_sequence_length=logical_sequence_length,
        block_lengths=block_lengths,
        sparse_query_blocks=sparse_query_blocks,
    )
    return _PreparedSparsePiperAttention(
        context=context,
        query=_prepare_sparse_piper_query_from_quantized(
            query,
            query_scale,
            routes,
            context,
            global_block_offset=query_block_offset,
        ),
    )
