"""Shared K/V statistics and value quantization for Piper Attention preparation."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

_P_UINT8_RANGE = tl.constexpr(255.0)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)
_MEAN_CHUNK_N = 1024
_MEAN_BLOCK_N = 64
_MEAN_BLOCK_D = 64


@triton.jit
def _kv_mean_partial_kernel(
    key_ptr,
    value_ptr,
    key_partial_ptr,
    value_partial_ptr,
    key_length,
    num_chunks,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    is_causal: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_n: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """Reduce one raw K chunk and, when non-causal, its V chunk."""
    chunk = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_d = feature_block * block_d + tl.arange(0, block_d)
    offsets_n = tl.arange(0, block_n)
    key_accumulator = tl.zeros((block_d,), dtype=tl.float32)
    if not is_causal:
        value_accumulator = tl.zeros((block_d,), dtype=tl.float32)
    chunk_start = chunk * chunk_n
    for offset in tl.range(0, chunk_n, block_n, disable_licm=True):
        current_n = chunk_start + offset + offsets_n
        mask = (current_n[:, None] < key_length) & (offsets_d[None, :] < head_dim)
        key = tl.load(
            key_ptr
            + batch * stride_kb
            + head * stride_kh
            + current_n[:, None] * stride_kn
            + offsets_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        key_accumulator += tl.sum(key, axis=0)
        if not is_causal:
            value = tl.load(
                value_ptr
                + batch * stride_vb
                + head * stride_vh
                + current_n[:, None] * stride_vn
                + offsets_d[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            value_accumulator += tl.sum(value, axis=0)
    output_offsets = (batch_head * num_chunks + chunk) * head_dim + offsets_d
    tl.store(key_partial_ptr + output_offsets, key_accumulator, mask=offsets_d < head_dim)
    if not is_causal:
        tl.store(
            value_partial_ptr + output_offsets,
            value_accumulator,
            mask=offsets_d < head_dim,
        )


@triton.jit
def _kv_mean_finalize_kernel(
    key_partial_ptr,
    value_partial_ptr,
    key_mean_ptr,
    value_mean_ptr,
    key_length,
    num_chunks,
    is_causal: tl.constexpr,
    head_dim: tl.constexpr,
    block_chunks: tl.constexpr,
    block_d: tl.constexpr,
):
    """Merge K and optional non-causal V partials into compact FP32 means."""
    batch_head = tl.program_id(0)
    feature_block = tl.program_id(1)
    offsets_c = tl.arange(0, block_chunks)
    offsets_d = feature_block * block_d + tl.arange(0, block_d)
    mask = (offsets_c[:, None] < num_chunks) & (offsets_d[None, :] < head_dim)
    partial_offsets = (batch_head * num_chunks + offsets_c[:, None]) * head_dim + offsets_d[None, :]
    key_partials = tl.load(key_partial_ptr + partial_offsets, mask=mask, other=0.0)
    output_offsets = batch_head * head_dim + offsets_d
    output_mask = offsets_d < head_dim
    tl.store(
        key_mean_ptr + output_offsets,
        tl.sum(key_partials, axis=0) / key_length,
        mask=output_mask,
    )
    if not is_causal:
        value_partials = tl.load(
            value_partial_ptr + partial_offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            value_mean_ptr + output_offsets,
            tl.sum(value_partials, axis=0) / key_length,
            mask=output_mask,
        )


def compute_kv_means(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the shared Piper K mean and optional non-causal V mean."""
    batch, heads, key_length, head_dim = key.shape
    num_chunks = int(triton.cdiv(key_length, _MEAN_CHUNK_N))
    partial_shape = (batch, heads, num_chunks, head_dim)
    key_partial = torch.empty(partial_shape, device=key.device, dtype=torch.float32)
    value_partial = (
        torch.empty_like(key_partial)
        if not is_causal
        else torch.empty((1,), device=value.device, dtype=torch.float32)
    )
    key_mean = torch.empty((batch, heads, head_dim), device=key.device, dtype=torch.float32)
    value_mean = (
        torch.empty_like(key_mean)
        if not is_causal
        else torch.empty((1,), device=value.device, dtype=torch.float32)
    )
    _kv_mean_partial_kernel[
        (
            num_chunks,
            int(triton.cdiv(head_dim, _MEAN_BLOCK_D)),
            batch * heads,
        )
    ](
        key,
        value,
        key_partial,
        value_partial,
        key_length,
        num_chunks,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        is_causal=is_causal,
        heads=heads,
        head_dim=head_dim,
        chunk_n=_MEAN_CHUNK_N,
        block_n=_MEAN_BLOCK_N,
        block_d=_MEAN_BLOCK_D,
        num_warps=4,
    )
    _kv_mean_finalize_kernel[(batch * heads, int(triton.cdiv(head_dim, _MEAN_BLOCK_D)))](
        key_partial,
        value_partial,
        key_mean,
        value_mean,
        key_length,
        num_chunks,
        is_causal=is_causal,
        head_dim=head_dim,
        block_chunks=triton.next_power_of_2(num_chunks),
        block_d=_MEAN_BLOCK_D,
        num_warps=4,
    )
    return key_mean, value_mean


@triton.jit
def quantize_value_per_key_block(
    value_ptr,
    value_mean_ptr,
    value_output_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    key_block,
    head,
    batch,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vob,
    stride_voh,
    stride_vod,
    stride_vok,
    is_causal: tl.constexpr,
    store_log_scale: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Quantize one per-key V block."""
    value_offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid_value = value_offsets_n < key_length
    batch_head = batch * heads + head
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + value_offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid_value[:, None],
        other=0.0,
    ).to(tl.float32)
    if not is_causal:
        value_mean = tl.load(value_mean_ptr + batch_head * head_dim + offsets_d)
        value = value - value_mean[None, :]
    value = tl.where(valid_value[:, None], value, 0.0)
    value_scale = tl.max(tl.abs(value), axis=1) / _V_INT8_RANGE + _SCALE_EPSILON
    value_quantized = qk_quantization.round_to_int8(value / value_scale[:, None])
    tl.store(
        value_scale_multiplier_ptr + batch_head * key_length + value_offsets_n,
        value_scale * _P_UINT8_RANGE,
        mask=valid_value,
    )
    if store_log_scale:
        tl.store(
            value_log_scale_ptr + batch_head * key_length + value_offsets_n,
            tl.log2(value_scale),
            mask=valid_value,
        )
    tl.store(
        value_output_ptr
        + batch * stride_vob
        + head * stride_voh
        + offsets_d[None, :] * stride_vod
        + value_offsets_n[:, None] * stride_vok,
        value_quantized,
        mask=valid_value[:, None],
    )
