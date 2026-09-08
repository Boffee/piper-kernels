"""ConvRot INT8 projection adapter for Sage-style Q/K fusion."""

from __future__ import annotations

import triton
import triton.language as tl

from piper_kernels.fusions.projected_qk import triton as projected_qk
from piper_kernels.linear.convrot.int8._kernels import triton as convrot_int8_kernels


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
    sequence_length,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    aligned_projection: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    rsqrt_fn: tl.constexpr = None,  # pyright: ignore[reportArgumentType]
):
    """Return one FP32 normalized and rotated projection tile."""
    projection = convrot_int8_kernels.scaled_int8_matmul(
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
    return projected_qk.rmsnorm_rope_tile(
        projection,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        sequence_offsets,
        sequence_length,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        mask_ragged_tail,
        block_m,
        rsqrt_fn,
    )
