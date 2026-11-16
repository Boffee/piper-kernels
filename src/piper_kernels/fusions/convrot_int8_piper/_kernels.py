"""Fused FP32 projections, transforms, and dense Piper operand stores."""

# Triton device parameters are not Python runtime values.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
# pyright: reportAssignmentType=false, reportAttributeAccessIssue=false

import triton
import triton.language as tl

from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization
from piper_kernels.attention.piper_attention import _quantization
from piper_kernels.fusions.convrot_int8_projection.triton import (
    project_rmsnorm_rope_tile,
    project_tile,
    projection_tile_ids,
)


@triton.jit
def _project_query_kernel(  # noqa: PLR0913, PLR0917
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_ptr,
    query_scale_ptr,
    row_block_offset,
    chunk_start,
    chunk_rows,
    sequence_length,
    storage_length,
    batch_size,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    softmax_scale: tl.constexpr,
    heads_per_program: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    round_rsqrt_to_nearest: tl.constexpr,
    aligned_projection: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    per_thread_scales: tl.constexpr,
    bias_ptr=None,
):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = heads_per_program * head_dim
    row_block, head_block = projection_tile_ids(group_m)
    row_block += row_block_offset
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + tl.arange(0, block_n)
    transformed = project_rmsnorm_rope_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        batch * sequence_length + chunk_start + sequence_offsets,
        weight_offsets,
        chunk_start + sequence_offsets,
        batch_size * sequence_length,
        sequence_length,
        input_features,
        heads * head_dim,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        aligned_projection,
        mask_ragged_tail,
        block_m,
        block_n,
        block_k,
        round_rsqrt_to_nearest,
        bias_ptr=bias_ptr,
    )
    group_offsets = tl.arange(0, 2)
    if per_thread_scales:
        # Each row stores the scale of its per-thread group.
        row_valid = head_offsets[:, None] < heads
        if mask_ragged_tail:
            transformed = tl.where(
                sequence_offsets[:, None, None] < chunk_rows,
                transformed,
                0.0,
            )
            row_valid = row_valid & (sequence_offsets[None, :] < chunk_rows)
        quantized, scale = qk_quantization.quantize_query_per_thread_tile(
            transformed,
            row_valid,
            softmax_scale,
            heads_per_program,
            head_dim,
            block_m,
        )
    else:
        group_valid = head_offsets[:, None] < heads
        if mask_ragged_tail:
            transformed = tl.where(
                sequence_offsets[:, None, None] < chunk_rows,
                transformed,
                0.0,
            )
            group_valid = group_valid & (
                row_block * block_m + group_offsets[None, :] * 32 < chunk_rows
            )
        quantized, scale = qk_quantization.quantize_query_tile(
            transformed,
            group_valid,
            softmax_scale,
            heads_per_program,
            head_dim,
            block_m,
            32,
        )
    batch_heads = batch * heads + head_offsets.to(tl.int64)
    query_offsets = (
        batch_heads[:, None, None] * storage_length * head_dim
        + sequence_offsets[None, :, None] * head_dim
        + tl.arange(0, head_dim)[None, None, :]
    )
    tl.store(query_ptr + query_offsets, quantized, mask=head_offsets[:, None, None] < heads)
    if per_thread_scales:
        scale_offsets = batch_heads[:, None] * storage_length + sequence_offsets[None, :]
    else:
        scale_offsets = (
            batch_heads[:, None] * (storage_length // 32) + row_block * 2 + group_offsets[None, :]
        )
    tl.store(query_scale_ptr + scale_offsets, scale, mask=head_offsets[:, None] < heads)


@triton.jit
def _project_key_kernel(  # noqa: PLR0913, PLR0917
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    stored_ptr,
    rows,
    sequence_length,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    aligned_projection: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    round_rsqrt_to_nearest: tl.constexpr,
    group_m: tl.constexpr,
    bias_ptr=None,
):
    """Project RMSNorm/RoPE K rows into BF16 storage for K1024 chunk quantization."""
    row_block, head_block = projection_tile_ids(group_m)
    row_block += row_block_offset
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    feature_offsets = tl.arange(0, head_dim)
    weight_offsets = head_block * block_n + tl.arange(0, block_n)
    transformed = project_rmsnorm_rope_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        batch * sequence_length + sequence_offsets,
        weight_offsets,
        sequence_offsets,
        rows,
        sequence_length,
        input_features,
        heads * head_dim,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        aligned_projection,
        mask_ragged_tail,
        block_m,
        block_n,
        block_k,
        round_rsqrt_to_nearest,
        bias_ptr=bias_ptr,
    )
    stored = tl.permute(transformed, (1, 0, 2)).to(tl.bfloat16)
    batch_heads = batch * heads + head_offsets.to(tl.int64)
    offsets = (
        batch_heads[:, None, None] * sequence_length * head_dim
        + sequence_offsets[None, :, None].to(tl.int64) * head_dim
        + feature_offsets[None, None, :]
    )
    tl.store(
        stored_ptr + offsets,
        stored,
        (head_offsets[:, None, None] < heads) & (sequence_offsets[None, :, None] < sequence_length),
    )


@triton.jit(do_not_specialize=["sequence_length", "storage_length", "chunk_count"])
def _project_value_kernel(  # noqa: PLR0913, PLR0917
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    mean_ptr,
    value_ptr,
    multiplier_ptr,
    log_ptr,
    bias_ptr,
    sequence_length,
    storage_length,
    chunk_count,
    batch_size,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    heads_per_program: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    packed_amd: tl.constexpr,
    aligned_projection: tl.constexpr,
):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = heads_per_program * head_dim
    row_block, head_block = projection_tile_ids(group_m)
    row_block += row_block_offset
    batch = tl.program_id(2)
    rows = row_block * block_m + tl.arange(0, block_m)
    features = head_block * block_n + tl.arange(0, block_n)
    value = project_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        batch * sequence_length + rows,
        features,
        batch_size * sequence_length,
        input_features,
        heads * head_dim,
        aligned_projection,
        block_m,
        block_n,
        block_k,
        bias_ptr=bias_ptr,
    )
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    dim = tl.arange(0, head_dim)
    chunk = (row_block * block_m) // 1024
    mean_offsets = (
        (batch * heads + head_offsets[:, None].to(tl.int64)) * chunk_count.to(tl.int64)
        + chunk.to(tl.int64)
    ) * head_dim + dim[None, :]
    mean = tl.load(mean_ptr + mean_offsets, head_offsets[:, None] < heads, 0.0)
    value_3d = value.reshape((block_m, heads_per_program, head_dim)) - mean[None, :, :]
    value_3d = tl.where(rows[:, None, None] < sequence_length, value_3d, 0.0)
    codes, scales = _quantization.quantize_value_rows(
        value_3d.reshape((block_m * heads_per_program, head_dim))
    )
    codes = codes.reshape((block_m, heads_per_program, head_dim)).permute((1, 0, 2))
    scales = scales.reshape((block_m, heads_per_program)).T
    batch_heads = batch * heads + head_offsets.to(tl.int64)
    if packed_amd:
        token = tl.arange(0, block_m)
        packed_token = (token & ~24) | ((token & 8) << 1) | ((token & 16) >> 1)
        offsets = (
            (batch_heads[:, None, None] * (storage_length // 64) + row_block) * head_dim * 64
            + dim[None, None, :] * 64
            + packed_token[None, :, None]
        )
    else:
        offsets = (
            batch_heads[:, None, None] * head_dim + dim[None, None, :]
        ) * storage_length + rows[None, :, None]
    tl.store(value_ptr + offsets, codes, head_offsets[:, None, None] < heads)
    metadata = batch_heads[:, None] * storage_length + rows[None, :]
    tl.store(multiplier_ptr + metadata, scales * 255.0, head_offsets[:, None] < heads)
    logs = tl.log2(scales)
    if not packed_amd:
        logs = logs.to(tl.float16).to(tl.float32)
    tl.store(log_ptr + metadata, logs, head_offsets[:, None] < heads)
