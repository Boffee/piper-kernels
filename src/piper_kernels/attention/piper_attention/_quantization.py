"""Value quantization for Piper Attention preparation."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import triton
import triton.language as tl

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

_P_UINT8_RANGE = tl.constexpr(255.0)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)


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
