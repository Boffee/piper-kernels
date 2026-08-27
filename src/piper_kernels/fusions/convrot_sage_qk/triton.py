"""Reusable ConvRot projection and Sage-style INT8 Q/K fusion primitives."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

_HEAD_DIM = 128
_LOG2_E = tl.constexpr(1.4426950408889634)
_SCALE_EPSILON = tl.constexpr(1e-7)


@triton.jit
def project_rmsnorm_rope_tile(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    row_offsets,
    weight_offsets,
    sequence_offsets,
    rows,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    aligned_projection: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Return one FP32 normalized and rotated projection tile."""
    feature_offsets = tl.arange(0, head_dim)
    projection = convrot_backend.scaled_int8_matmul(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        row_offsets,
        weight_offsets,
        rows,
        output_features,
        input_features,
        block_m,
        block_n,
        block_k,
        aligned_projection,
    )
    projection = tl.reshape(projection, (block_m, heads_per_program, head_dim))
    inverse_rms = libdevice.rsqrt_rn(
        tl.sum(projection * projection, axis=2) / head_dim + norm_epsilon
    )
    norm_weight = tl.load(norm_weight_ptr + feature_offsets).to(tl.float32)
    normalized = projection * inverse_rms[:, :, None] * norm_weight[None, None, :]

    half_rotary_dim: tl.constexpr = rotary_dim // 2
    paired_features = tl.where(
        feature_offsets < half_rotary_dim,
        feature_offsets + half_rotary_dim,
        tl.where(
            feature_offsets < rotary_dim,
            feature_offsets - half_rotary_dim,
            feature_offsets,
        ),
    )
    paired_indices = tl.broadcast_to(
        paired_features[None, None, :],
        (block_m, heads_per_program, head_dim),
    )
    paired = tl.gather(normalized, paired_indices, axis=2)
    rotated = tl.where(feature_offsets[None, None, :] < half_rotary_dim, -paired, paired)
    rope_mask = feature_offsets < rotary_dim
    cos = tl.load(
        cos_ptr + sequence_offsets[:, None, None] * rotary_dim + feature_offsets[None, None, :],
        mask=rope_mask[None, None, :],
        other=1.0,
    )
    sin = tl.load(
        sin_ptr + sequence_offsets[:, None, None] * rotary_dim + feature_offsets[None, None, :],
        mask=rope_mask[None, None, :],
        other=0.0,
    )
    rotary = normalized * cos + rotated * sin
    return tl.where(
        rope_mask[None, None, :],
        rotary,
        normalized,
    )


@triton.jit
def quantize_query_tile(
    values,
    group_valid,
    softmax_scale: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Return Sage-style INT8 Q and base-2 recurrence scales for one tile."""
    smoothed = qk_quantization.rotate_signed_hadamard_heads(
        tl.reshape(values, (block_m * heads_per_program, head_dim)),
        head_dim,
    )
    smoothed = tl.permute(
        tl.reshape(smoothed, (block_m, heads_per_program, head_dim)),
        (1, 0, 2),
    )
    grouped = tl.reshape(
        smoothed,
        (
            heads_per_program,
            block_m // scale_rows,
            scale_rows,
            head_dim,
        ),
    )
    maximum = tl.max(tl.max(tl.abs(grouped), axis=3), axis=2)
    raw_scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(
        grouped / tl.where(group_valid, raw_scale, 1.0)[:, :, None, None]
    )
    stored_scale = tl.where(
        group_valid,
        raw_scale * (softmax_scale * _LOG2_E),
        0.0,
    )
    return tl.reshape(quantized, (heads_per_program, block_m, head_dim)), stored_scale


@triton.jit
def quantize_key_tile(
    values,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Return Sage-style INT8 K and K-tile scales for one projection tile."""
    smoothed = qk_quantization.rotate_signed_hadamard_heads(
        tl.reshape(values, (block_m * heads_per_program, head_dim)),
        head_dim,
    )
    smoothed = tl.permute(
        tl.reshape(smoothed, (block_m, heads_per_program, head_dim)),
        (1, 0, 2),
    )
    grouped = tl.reshape(
        smoothed,
        (
            heads_per_program,
            block_m // scale_rows,
            scale_rows,
            head_dim,
        ),
    )
    maximum = tl.max(tl.max(tl.abs(grouped), axis=3), axis=2)
    key_scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(grouped / key_scale[:, :, None, None])
    return tl.reshape(quantized, (heads_per_program, block_m, head_dim)), key_scale


def validate_qk_projection_inputs(  # noqa: PLR0912
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    valid_sequence_length: int,
    norm_epsilon: float,
    block_rows: int,
    name: str,
) -> tuple[int, int, int, int]:
    """Validate inputs to a fused ConvRot Q/K projection kernel."""
    if input_qdata.ndim != 3 or input_qdata.dtype is not torch.int8:
        raise ValueError(f"{name} projection input must be [batch,sequence,features] INT8")
    batch, sequence_length, input_features = input_qdata.shape
    if input_scale.shape != (batch, sequence_length) or input_scale.dtype is not torch.float32:
        raise ValueError(f"{name} projection input scale must be a batch/sequence FP32 matrix")
    if weight_qdata.ndim != 2 or weight_qdata.dtype is not torch.int8:
        raise ValueError(f"{name} projection weight must be a two-dimensional INT8 tensor")
    if weight_qdata.shape[1] != input_features or weight_qdata.shape[0] % _HEAD_DIM:
        raise ValueError(f"{name} projection weight must map the input to complete D128 heads")
    if weight_scale.shape != (weight_qdata.shape[0], 1) or weight_scale.dtype is not torch.float32:
        raise ValueError(
            f"{name} projection weight scale must be one FP32 value per output feature"
        )
    heads = weight_qdata.shape[0] // _HEAD_DIM
    if norm_weight.shape != (_HEAD_DIM,) or norm_weight.dtype is not torch.bfloat16:
        raise ValueError(f"{name} projection RMSNorm weight must be a BF16 D128 vector")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != sequence_length:
        raise ValueError(f"{name} projection RoPE cos/sin must match the physical sequence")
    rotary_dim = cos.shape[1]
    if rotary_dim < 2 or rotary_dim > _HEAD_DIM or rotary_dim % 2:
        raise ValueError(f"{name} projection rotary dimension must be even and fit D128")
    if cos.dtype is not torch.float32 or sin.dtype is not cos.dtype:
        raise ValueError(f"{name} projection RoPE cos/sin must use FP32")
    operands = input_qdata, input_scale, weight_qdata, weight_scale, norm_weight, cos, sin
    if any(operand.device != input_qdata.device for operand in operands):
        raise ValueError(f"{name} projection operands must share a device")
    if input_qdata.device.type != "cuda":
        raise ValueError(f"{name} projection fusion currently requires CUDA")
    if any(
        operand.layout is not torch.strided or not operand.is_contiguous() for operand in operands
    ):
        raise ValueError(f"{name} projection operands must be contiguous strided tensors")
    if sequence_length < block_rows or sequence_length % block_rows:
        raise ValueError(f"{name} projection physical sequence must be {block_rows}-row aligned")
    if not sequence_length - block_rows < valid_sequence_length <= sequence_length:
        raise ValueError(
            f"{name} projection supports padding only in the final {block_rows}-row block"
        )
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError(f"{name} projection RMSNorm epsilon must be finite and positive")
    return batch, sequence_length, heads, rotary_dim
