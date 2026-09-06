"""Shared Triton preparation for the quantized sparse-Piper tensor contract."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

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
from piper_kernels.attention.piper_attention import _quantization as piper_quantization

from ._block_layout import valid_block_rows, validate_sparse_query_blocks
from ._prepared import (
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
    _PreparedSparsePiperContext,
)

_BLOCK_N = 64


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
