"""INT8 preparation for the selected sparse Piper SM120 kernel."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_kernels,
)
from piper_kernels.attention.piper_attention import _quantization as piper_quantization

_BLOCK_M = 64
_BLOCK_N = 64
_HEAD_DIM = 128


@triton.jit
def _quantize_value_per_tile_kernel(
    value_ptr,
    value_mean_ptr,
    value_scale_ptr,
    output_ptr,
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
):
    """Quantize one D128 K64 storage tile while masking the logical V tail."""
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid_rows = offsets_n < key_length
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
class _PreparedSparsePiperAttention:
    """Quantized inputs and routing metadata consumed by the Gluon launch."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_mean: torch.Tensor
    routes: torch.Tensor
    route_head_offsets: torch.Tensor
    keep_blocks: torch.Tensor
    sparse_key_blocks: int
    logical_sequence_length: int


def _prepare_sparse_piper_attention(
    query: torch.Tensor,
    routes: torch.Tensor,
    keep_blocks: torch.Tensor,
    scale: float,
    *,
    sparse_key_blocks: int,
    route_head_offsets: torch.Tensor,
    combined_key: torch.Tensor,
    combined_value: torch.Tensor,
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
    tile_count = (logical_sequence_length + _BLOCK_N - 1) // _BLOCK_N
    storage_sequence_length = tile_count * _BLOCK_N

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

    _quantize_value_per_tile_kernel[(tile_count, batch * heads)](
        combined_value,
        value_mean,
        value_scale_multiplier,
        value_int8,
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
        num_warps=4,
    )

    return _PreparedSparsePiperAttention(
        query=prepared_qk.query,
        key=prepared_qk.key,
        value=value_int8,
        query_scale=prepared_qk.query_scale,
        key_scale=prepared_qk.key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_mean=value_mean,
        routes=routes,
        route_head_offsets=route_head_offsets,
        keep_blocks=keep_blocks,
        sparse_key_blocks=sparse_key_blocks,
        logical_sequence_length=logical_sequence_length,
    )


def _prepare_sparse_piper_attention_from_quantized(  # noqa: PLR0912
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    routes: torch.Tensor,
    keep_blocks: torch.Tensor,
    route_head_offsets: torch.Tensor,
    *,
    sparse_key_blocks: int,
    routes_per_query: int,
    logical_sequence_length: int,
) -> _PreparedSparsePiperAttention:
    """Construct sparse Piper launch state from already-quantized operands."""
    if query.ndim != 4 or query.dtype is not torch.int8:
        raise ValueError(
            "quantized sparse Piper Q must be [batch,heads,storage_sequence,D128] INT8"
        )
    batch, heads, storage_sequence_length, head_dim = query.shape
    if (
        head_dim != _HEAD_DIM
        or storage_sequence_length < _BLOCK_M
        or storage_sequence_length % _BLOCK_M
    ):
        raise ValueError("quantized sparse Piper requires K64-aligned D128 query storage")
    if (
        logical_sequence_length < _BLOCK_M
        or logical_sequence_length > storage_sequence_length
        or (logical_sequence_length + _BLOCK_M - 1) // _BLOCK_M * _BLOCK_M
        != storage_sequence_length
    ):
        raise ValueError("quantized sparse Piper storage must be the padded logical sequence")
    if key.shape != query.shape or key.dtype is not torch.int8:
        raise ValueError("quantized sparse Piper K must match Q and use INT8")
    if (
        value.shape != (batch, heads, head_dim, storage_sequence_length)
        or value.dtype is not torch.int8
    ):
        raise ValueError(
            "quantized sparse Piper V must be transposed INT8 [B,H,D,storage_sequence]"
        )
    tile_count = storage_sequence_length // _BLOCK_N
    if query_scale.shape != (batch, heads, storage_sequence_length // 32):
        raise ValueError("quantized sparse Piper Q scales must contain one value per Q32")
    if key_scale.shape != (batch, heads, tile_count):
        raise ValueError("quantized sparse Piper K scales must contain one value per K64")
    if value_scale_multiplier.shape != (batch, heads, tile_count, 1):
        raise ValueError("quantized sparse Piper V scales must contain one value per K64")
    if value_mean.shape != (batch, heads, head_dim):
        raise ValueError("quantized sparse Piper V mean must be [batch,heads,D128]")
    scales = query_scale, key_scale, value_scale_multiplier, value_mean
    if any(scale.dtype is not torch.float32 for scale in scales):
        raise ValueError("quantized sparse Piper scales and V mean must use FP32")
    tensors = query, key, value, *scales, routes, keep_blocks, route_head_offsets
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must share a device")
    if any(tensor.layout is not torch.strided or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must be contiguous strided tensors")
    if not 1 <= sparse_key_blocks <= tile_count:
        raise ValueError("quantized sparse Piper prefix must fit the K64 tile count")
    query_block_count = (logical_sequence_length + _BLOCK_M - 1) // _BLOCK_M
    if routes.shape != (batch, query_block_count, routes_per_query):
        raise ValueError("quantized sparse Piper routes must match batch/query/packed budgets")
    if routes.dtype is not torch.uint16:
        raise ValueError("quantized sparse Piper routes must use UINT16")
    if keep_blocks.shape != (heads,) or keep_blocks.dtype is not torch.int32:
        raise ValueError("quantized sparse Piper keep counts must be one INT32 value per head")
    if route_head_offsets.shape != (heads + 1,) or route_head_offsets.dtype is not torch.int32:
        raise ValueError("quantized sparse Piper route offsets must be an INT32 head vector")

    return _PreparedSparsePiperAttention(
        query=query,
        key=key,
        value=value,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_mean=value_mean,
        routes=routes,
        route_head_offsets=route_head_offsets,
        keep_blocks=keep_blocks,
        sparse_key_blocks=sparse_key_blocks,
        logical_sequence_length=logical_sequence_length,
    )
