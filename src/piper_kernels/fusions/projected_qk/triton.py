"""Projection-independent fused Q/K transformation primitives."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def rmsnorm_rope_tile(
    projection,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    sequence_offsets,
    sequence_length,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    block_m: tl.constexpr,
    rsqrt_fn: tl.constexpr = None,  # pyright: ignore[reportArgumentType]
):
    """Apply FP32 RMSNorm and RoPE; execution may supply a rounding-specific rsqrt."""
    feature_offsets = tl.arange(0, head_dim)
    variance = tl.sum(projection * projection, axis=2) / head_dim + norm_epsilon
    # Resolve generic libdevice through the compiling target, not a Python
    # default that captures its non-executable declaration before JIT.
    inverse_rms = libdevice.rsqrt(variance) if rsqrt_fn is None else rsqrt_fn(variance)
    norm_weight = tl.load(norm_weight_ptr + feature_offsets).to(tl.float32)
    normalized = projection * inverse_rms[:, :, None] * norm_weight[None, None, :]  # pyright: ignore[reportOptionalSubscript]

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
    rotary_features = feature_offsets < rotary_dim
    if mask_ragged_tail:
        rope_load_mask = (sequence_offsets[:, None, None] < sequence_length) & rotary_features[
            None, None, :
        ]
    else:
        rope_load_mask = rotary_features[None, None, :]
    cos = tl.load(
        cos_ptr + sequence_offsets[:, None, None] * rotary_dim + feature_offsets[None, None, :],
        mask=rope_load_mask,
        other=1.0,
    )
    sin = tl.load(
        sin_ptr + sequence_offsets[:, None, None] * rotary_dim + feature_offsets[None, None, :],
        mask=rope_load_mask,
        other=0.0,
    )
    rotary = normalized * cos + rotated * sin
    return tl.where(
        rotary_features[None, None, :],
        rotary,
        normalized,
    )
