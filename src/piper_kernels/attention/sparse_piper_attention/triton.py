"""INT8 preparation for the selected sparse Piper SM120 kernel."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.piper_attention import triton as piper_backend

_BLOCK_M = 64
_BLOCK_N = 64
_HEAD_DIM = 128
_P_UINT8_RANGE = tl.constexpr(255.0)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)


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
    key_block_offset,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    mask_key_length: tl.constexpr,
):
    """Quantize one complete D128 K64 tile with one folded V scale."""
    key_block = key_block_offset + tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid = offsets_n < key_length if mask_key_length else tl.full((block_n,), True, tl.int1)
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    value_mean = tl.load(value_mean_ptr + batch_head * head_dim + offsets_d)
    centered = tl.where(valid[:, None], value - value_mean[None, :], 0.0)
    maximum = tl.max(tl.max(tl.abs(centered), axis=1), axis=0)
    value_scale = maximum / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(centered / value_scale)
    tile_count = tl.cdiv(key_length, block_n)
    tl.store(
        value_scale_ptr + batch_head * tile_count + key_block,
        value_scale * _P_UINT8_RANGE,
    )
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_d[None, :] * stride_od
        + offsets_n[:, None] * stride_ok,
        quantized,
        mask=valid[:, None],
    )


@dataclass(frozen=True, slots=True)
class _PreparedRoutedPiperAttention:
    """Quantized storage and compact route metadata for one launch."""

    attention: piper_backend._PreparedPiperAttention
    routes: torch.Tensor
    route_head_offsets: torch.Tensor
    keep_blocks: torch.Tensor
    query_block_count: int
    sparse_key_block_count: int
    dense_suffix_length: int


def _prepare_folded_tile_scaled_routed_piper_attention(
    query_blocks: torch.Tensor,
    key_blocks: torch.Tensor,
    routes: torch.Tensor,
    keep_blocks: torch.Tensor,
    scale: float,
    *,
    valid_query_count: int | None = None,
    route_head_offsets: torch.Tensor | None = None,
    combined_key: torch.Tensor | None = None,
    combined_value: torch.Tensor | None = None,
    attention_output: torch.Tensor | None = None,
) -> _PreparedRoutedPiperAttention:
    """Prepare grouped Q/K and one folded V scale per logical K64 tile."""
    if route_head_offsets is None:
        raise ValueError("sparse Piper requires packed route head offsets")
    if combined_key is None or combined_value is None:
        raise ValueError("sparse Piper requires compact sequence-major K/V views")

    query = query_blocks.flatten(2, 3)
    if (
        combined_key.ndim != 4
        or combined_key.shape[:2] != query.shape[:2]
        or combined_key.shape[-1] != _HEAD_DIM
        or combined_value.shape != combined_key.shape
        or combined_key.shape[2] < key_blocks.shape[2] * _BLOCK_N
    ):
        raise ValueError("combined K/V must contain the sparse prefix plus dense suffix")
    if combined_key.stride(-1) != 1 or combined_value.stride(-1) != 1:
        raise ValueError("combined K/V feature dimensions must be contiguous")

    plan = replace(
        piper_backend._default_piper_attention_execution_plan(query, False),
        block_m=_BLOCK_M,
        use_tensor_descriptors=True,
        num_stages=2,
    )
    if not plan.grouped_qk or not plan.split_pv_head_dim:
        raise ValueError("sparse Piper requires grouped Q/K and split PV")
    with torch.cuda.device(query.device):
        install_uint8_int8_dot_hook()

    batch, heads, query_length, head_dim = query.shape
    logical_query_length = query_length if valid_query_count is None else valid_query_count
    if not 1 <= logical_query_length <= query_length:
        raise ValueError("valid query count must fit the physical query length")
    key_length = combined_key.shape[2]
    tile_count = int(triton.cdiv(key_length, _BLOCK_N))
    storage_key_length = tile_count * _BLOCK_N

    key_mean, value_mean = piper_backend._compute_kv_means(
        combined_key,
        combined_value,
        is_causal=False,
    )
    query_int8, query_scale = qk_quantization.prepare_query(
        query[:, :, :logical_query_length],
        scale,
        grouped=True,
        storage_query_length=query_length,
    )
    key_int8, key_scale = qk_quantization.prepare_key(
        combined_key,
        key_mean,
        grouped=True,
        storage_key_length=storage_key_length,
    )
    value_int8 = torch.empty(
        (batch, heads, head_dim, storage_key_length),
        device=combined_value.device,
        dtype=torch.int8,
    )
    value_scale_multiplier = torch.empty(
        (batch, heads, tile_count, 1),
        device=combined_value.device,
        dtype=torch.float32,
    )

    def quantize_tiles(
        launch_tile_count: int,
        *,
        key_block_offset: int,
        mask_key_length: bool,
    ) -> None:
        _quantize_value_per_tile_kernel[(launch_tile_count, batch * heads)](
            combined_value,
            value_mean,
            value_scale_multiplier,
            value_int8,
            key_length,
            combined_value.stride(0),
            combined_value.stride(1),
            combined_value.stride(2),
            value_int8.stride(0),
            value_int8.stride(1),
            value_int8.stride(2),
            value_int8.stride(3),
            key_block_offset,
            heads=heads,
            head_dim=head_dim,
            block_n=_BLOCK_N,
            mask_key_length=mask_key_length,
            num_warps=4,
        )

    full_tile_count = key_length // _BLOCK_N
    if full_tile_count:
        quantize_tiles(full_tile_count, key_block_offset=0, mask_key_length=False)
    if key_length % _BLOCK_N:
        quantize_tiles(1, key_block_offset=full_tile_count, mask_key_length=True)

    key_descriptor, value_descriptor = piper_backend._make_key_value_descriptors(
        key_int8,
        value_int8,
        split_pv_head_dim=True,
    )
    query_descriptor = piper_backend._make_query_descriptor(query_int8, _BLOCK_M)
    if attention_output is None:
        attention_output = torch.empty_like(query, memory_format=torch.contiguous_format)
    elif (
        attention_output.shape != query.shape
        or attention_output.dtype != query.dtype
        or attention_output.device != query.device
        or attention_output.stride(-1) != 1
    ):
        raise ValueError("sparse Piper output must match Q and have contiguous features")

    attention = piper_backend._PreparedPiperAttention(
        query=query_int8,
        query_descriptor=query_descriptor,
        key=key_descriptor,
        value=value_descriptor,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_log_scale=torch.empty((1,), device=query.device, dtype=torch.float16),
        value_mean=value_mean,
        output=attention_output,
        key_length=key_length,
        is_causal=False,
        plan=plan,
    )
    return _PreparedRoutedPiperAttention(
        attention=attention,
        routes=routes,
        route_head_offsets=route_head_offsets,
        keep_blocks=keep_blocks,
        query_block_count=query_blocks.shape[2],
        sparse_key_block_count=key_blocks.shape[2],
        dense_suffix_length=combined_key.shape[2] - key_blocks.shape[2] * _BLOCK_N,
    )
