"""Sparse-Piper-specific operand preparation primitives."""

from __future__ import annotations

import triton
import triton.language as tl

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

_P_UINT8_RANGE = tl.constexpr(255.0)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)


@triton.jit
def quantize_value_tile(
    values,
    value_mean,
    valid_rows,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Center and encode one projected tile in sparse-Piper's transposed V format."""
    centered = tl.where(
        valid_rows[:, None, None],
        values - value_mean[None, :, :],
        0.0,
    )
    grouped = tl.reshape(
        tl.permute(centered, (1, 0, 2)),
        (
            heads_per_program,
            block_m // scale_rows,
            scale_rows,
            head_dim,
        ),
    )
    maximum = tl.max(tl.max(tl.abs(grouped), axis=3), axis=2)
    value_scale = maximum / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(grouped / value_scale[:, :, None, None])
    return (
        tl.reshape(quantized, (heads_per_program, block_m, head_dim)),
        value_scale * _P_UINT8_RANGE,
    )


@triton.jit
def store_query_tile(
    values,
    query_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    batch,
    heads,
    head_offsets,
    sequence_offsets,
    logical_sequence_length,
    storage_sequence_length,
    query_block,
    softmax_scale: tl.constexpr,
    mean_pool_summary: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Quantize and store one transformed sparse-Piper query tile and route summary."""
    feature_offsets = tl.arange(0, head_dim)
    valid_rows = sequence_offsets < logical_sequence_length
    block_length = block_m
    if mask_block_lengths:
        block_length = tl.load(block_lengths_ptr + query_block)
        valid_rows = sequence_offsets - query_block * block_m < block_length
    if mask_block_lengths or mask_ragged_tail:
        values = tl.where(valid_rows[:, None, None], values, 0.0)
    if mean_pool_summary:
        if mask_block_lengths:
            valid_count = block_length
        else:
            valid_count = (
                logical_sequence_length - query_block * block_m if mask_ragged_tail else block_m
            )
        summary = tl.sum(values, axis=0) / valid_count
    elif mask_block_lengths or mask_ragged_tail:
        summary = tl.max(
            tl.where(valid_rows[:, None, None], values, -float("inf")), axis=0
        ) + tl.min(tl.where(valid_rows[:, None, None], values, float("inf")), axis=0)
    else:
        summary = tl.max(values, axis=0) + tl.min(values, axis=0)
    summary_offsets = (
        (batch * heads + head_offsets[:, None]) * (storage_sequence_length // block_m) + query_block
    ) * head_dim + feature_offsets[None, :]
    tl.store(
        query_summary_ptr + summary_offsets,
        summary,
        mask=head_offsets[:, None] < heads,
    )

    group_offsets = tl.arange(0, block_m // scale_rows)
    group_valid = head_offsets[:, None] < heads
    if mask_block_lengths:
        group_starts = group_offsets * scale_rows
        group_valid = group_valid & (group_starts[None, :] < block_length)
    elif mask_ragged_tail:
        group_starts = query_block * block_m + group_offsets * scale_rows
        group_valid = group_valid & (group_starts[None, :] < logical_sequence_length)
    quantized, stored_scale = qk_quantization.quantize_query_tile(
        values,
        group_valid,
        softmax_scale,
        heads_per_program,
        head_dim,
        block_m,
        scale_rows,
    )
    query_offsets = (
        batch * heads * storage_sequence_length * head_dim
        + head_offsets[:, None, None] * storage_sequence_length * head_dim
        + sequence_offsets[None, :, None] * head_dim
        + feature_offsets[None, None, :]
    )
    tl.store(
        query_ptr + query_offsets,
        quantized,
        mask=head_offsets[:, None, None] < heads,
    )
    scale_offsets = (
        (batch * heads + head_offsets[:, None]) * (storage_sequence_length // scale_rows)
        + query_block * (block_m // scale_rows)
        + group_offsets[None, :]
    )
    tl.store(
        query_scale_ptr + scale_offsets,
        stored_scale,
        mask=head_offsets[:, None] < heads,
    )


@triton.jit
def store_key_tile(
    values,
    key_ptr,
    key_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    batch,
    heads,
    head_offsets,
    sequence_offsets,
    logical_sequence_length,
    storage_sequence_length,
    row_block,
    mean_pool_summary: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Quantize and store one transformed sparse-Piper key tile and route summaries."""
    feature_offsets = tl.arange(0, head_dim)
    local_tile_offsets = tl.arange(0, block_m // scale_rows)
    tile_offsets = row_block * (block_m // scale_rows) + local_tile_offsets
    row_in_tile = tl.arange(0, scale_rows)
    if mask_block_lengths:
        lengths = tl.load(
            block_lengths_ptr + tile_offsets,
            mask=tile_offsets < storage_sequence_length // scale_rows,
            other=0,
        )
        valid = row_in_tile[None, :] < lengths[:, None]
    else:
        valid = tile_offsets[:, None] * scale_rows + row_in_tile[None, :] < logical_sequence_length
    values = tl.where(
        tl.reshape(valid, (block_m,))[:, None, None],
        values,
        0.0,
    )
    grouped = tl.reshape(
        tl.permute(values, (1, 0, 2)),
        (
            heads_per_program,
            block_m // scale_rows,
            scale_rows,
            head_dim,
        ),
    )
    if mean_pool_summary:
        valid_count = tl.sum(valid.to(tl.int32), axis=1)
        key_summary = tl.sum(grouped, axis=2) / valid_count[None, :, None]
    else:
        key_summary = tl.max(
            tl.where(valid[None, :, :, None], grouped, -float("inf")),
            axis=2,
        )
        key_aux = tl.min(
            tl.where(valid[None, :, :, None], grouped, float("inf")),
            axis=2,
        )

    quantized, key_scale = qk_quantization.quantize_key_tile(
        values,
        heads_per_program,
        head_dim,
        block_m,
        scale_rows,
    )
    key_offsets = (
        (batch * heads + head_offsets[:, None, None]) * storage_sequence_length * head_dim
        + sequence_offsets[None, :, None] * head_dim
        + feature_offsets[None, None, :]
    )
    tl.store(
        key_ptr + key_offsets,
        quantized,
        mask=(head_offsets[:, None, None] < heads)
        & (sequence_offsets[None, :, None] < storage_sequence_length),
    )
    tile_count = storage_sequence_length // scale_rows
    scale_offsets = (batch * heads + head_offsets[:, None]) * tile_count + tile_offsets[None, :]
    tile_mask = (head_offsets[:, None] < heads) & (tile_offsets[None, :] < tile_count)
    tl.store(key_scale_ptr + scale_offsets, key_scale, mask=tile_mask)
    summary_offsets = scale_offsets[:, :, None] * head_dim + feature_offsets[None, None, :]
    tl.store(key_summary_ptr + summary_offsets, key_summary, mask=tile_mask[:, :, None])
    if not mean_pool_summary:
        tl.store(key_aux_ptr + summary_offsets, key_aux, mask=tile_mask[:, :, None])


@triton.jit
def store_value_tile(
    values,
    value_mean_ptr,
    value_ptr,
    value_scale_ptr,
    block_mean_ptr,
    block_lengths_ptr,
    batch,
    heads,
    head_offsets,
    sequence_offsets,
    logical_sequence_length,
    storage_sequence_length,
    row_block,
    mask_block_lengths: tl.constexpr,
    emit_block_mean: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Center, quantize, and optionally mean-pool one sparse-Piper value tile."""
    feature_offsets = tl.arange(0, head_dim)
    value_mean = tl.load(
        value_mean_ptr
        + (batch * heads + head_offsets[:, None]) * head_dim
        + feature_offsets[None, :],
        mask=head_offsets[:, None] < heads,
        other=0.0,
    )
    tile_count = storage_sequence_length // scale_rows
    local_tile_offsets = tl.arange(0, block_m // scale_rows)
    tile_offsets = row_block * (block_m // scale_rows) + local_tile_offsets
    scale_offsets = (batch * heads + head_offsets[:, None]) * tile_count + tile_offsets[None, :]
    if mask_block_lengths:
        block_lengths = tl.load(
            block_lengths_ptr + tile_offsets,
            mask=tile_offsets < tile_count,
            other=0,
        )
        rows_in_tile = tl.arange(0, scale_rows)
        grouped_valid = rows_in_tile[None, :] < block_lengths[:, None]
        valid_rows = tl.reshape(grouped_valid, (block_m,))
    else:
        valid_rows = sequence_offsets < logical_sequence_length
    if emit_block_mean:
        value_blocks = tl.reshape(
            tl.permute(values, (1, 0, 2)),
            (
                heads_per_program,
                block_m // scale_rows,
                scale_rows,
                head_dim,
            ),
        )
        grouped_valid = tl.reshape(valid_rows, (block_m // scale_rows, scale_rows))
        valid_count = tl.maximum(tl.sum(grouped_valid.to(tl.int32), axis=1), 1)
        block_mean = (
            tl.sum(
                tl.where(grouped_valid[None, :, :, None], value_blocks, 0.0),
                axis=2,
            )
            / valid_count[None, :, None]
        )
        block_mean_offsets = scale_offsets[:, :, None] * head_dim + feature_offsets[None, None, :]
        tl.store(
            block_mean_ptr + block_mean_offsets,
            block_mean,
            mask=(head_offsets[:, None, None] < heads) & (tile_offsets[None, :, None] < tile_count),
        )
    quantized, value_scale_multiplier = quantize_value_tile(  # pyright: ignore[reportGeneralTypeIssues]
        values,
        value_mean,
        valid_rows,
        heads_per_program,
        head_dim,
        block_m,
        scale_rows,
    )
    value_offsets = (
        (batch * heads + head_offsets[:, None, None]) * head_dim * storage_sequence_length
        + feature_offsets[None, None, :] * storage_sequence_length
        + sequence_offsets[None, :, None]
    )
    tl.store(
        value_ptr + value_offsets,
        quantized,
        mask=(head_offsets[:, None, None] < heads)
        & (sequence_offsets[None, :, None] < storage_sequence_length),
    )

    tl.store(
        value_scale_ptr + scale_offsets,
        value_scale_multiplier,
        mask=(head_offsets[:, None] < heads) & (tile_offsets[None, :] < tile_count),
    )
