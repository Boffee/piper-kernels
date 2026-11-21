"""Sparse-Piper-specific operand preparation primitives."""

# Triton device-function return types are not represented in its Python stubs.
# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

import triton
import triton.language as tl

from piper_kernels.attention.kernels.piper.triton import pack_amd_wmma_token
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

from .layout import ValueFormat

_P_UINT8_RANGE = tl.constexpr(255.0)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)
# Keep per-tile mean restoration finite when a constant tile has tiny residual scale.
_VALUE_MEAN_SCALE_FLOOR = tl.constexpr(2.0**-80)
_AMD_VALUE_FORMAT = tl.constexpr(int(ValueFormat.AMD_WMMA))
_TRANSPOSED_VALUE_FORMAT = tl.constexpr(int(ValueFormat.TRANSPOSED))


@triton.jit
def summarize_block_tiles(
    values,
    valid_rows,
    mean_pool_summary: tl.constexpr,
    combine_extrema: tl.constexpr,
):
    """Reduce ``[heads,blocks,rows,features]`` through one routing policy."""
    valid = valid_rows[None, :, :, None]
    if mean_pool_summary:
        valid_count = tl.sum(valid_rows.to(tl.int32), axis=1)
        primary = tl.sum(tl.where(valid, values, 0.0), axis=2) / valid_count[None, :, None]  # pyright: ignore[reportArgumentType]
        auxiliary = primary
    else:
        maximum = tl.max(tl.where(valid, values, -float("inf")), axis=2)
        minimum = tl.min(tl.where(valid, values, float("inf")), axis=2)
        primary = maximum + minimum if combine_extrema else maximum  # pyright: ignore[reportOperatorIssue]
        auxiliary = minimum
    return primary, auxiliary


@triton.jit
def quantize_value_tile(
    values,
    valid_rows,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Per-K64 tile-center and encode V, returning codes, multipliers, and BF16 means."""
    num_tiles: tl.constexpr = block_m // scale_rows
    grouped = tl.reshape(
        tl.permute(values, (1, 0, 2)),
        (
            heads_per_program,
            num_tiles,
            scale_rows,
            head_dim,
        ),
    )
    grouped_valid = tl.reshape(valid_rows, (num_tiles, scale_rows))
    masked = tl.where(grouped_valid[None, :, :, None], grouped, 0.0)
    valid_count = tl.maximum(tl.sum(grouped_valid.to(tl.float32), axis=1), 1.0)
    mean_fp32 = tl.sum(masked, axis=2) / valid_count[None, :, None]
    mean_bf16 = mean_fp32.to(tl.bfloat16)
    centered = tl.where(
        grouped_valid[None, :, :, None],
        masked - mean_bf16.to(tl.float32)[:, :, None, :],
        0.0,
    )
    maximum = tl.max(tl.max(tl.abs(centered), axis=3), axis=2)
    mean_magnitude = tl.max(tl.abs(mean_bf16.to(tl.float32)), axis=2)
    value_scale = tl.maximum(
        maximum / _V_INT8_RANGE + _SCALE_EPSILON,
        mean_magnitude * _VALUE_MEAN_SCALE_FLOOR,
    )
    quantized = qk_quantization.round_to_int8(centered / value_scale[:, :, None, None])
    return (
        tl.reshape(quantized, (heads_per_program, block_m, head_dim)),
        value_scale * _P_UINT8_RANGE,
        mean_bf16,
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
    global_sequence_offsets,
    storage_sequence_offsets,
    logical_sequence_length,
    storage_sequence_length,
    global_query_block,
    storage_query_block,
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
    # Head-major operands can exceed 2**31 elements for long video sequences.
    head_offsets = head_offsets.to(tl.int64)
    feature_offsets = tl.arange(0, head_dim)
    valid_rows = global_sequence_offsets < logical_sequence_length
    block_length = block_m
    if mask_block_lengths:
        block_length = tl.load(block_lengths_ptr + global_query_block)
        valid_rows = valid_rows & (
            global_sequence_offsets - global_query_block * block_m < block_length
        )
    if mask_block_lengths or mask_ragged_tail:
        values = tl.where(valid_rows[:, None, None], values, 0.0)
    summary_values = tl.reshape(
        tl.permute(values, (1, 0, 2)),
        (heads_per_program, 1, block_m, head_dim),
    )
    summary, _summary_aux = summarize_block_tiles(
        summary_values,
        tl.reshape(valid_rows, (1, block_m)),
        mean_pool_summary,
        tl.constexpr(True),
    )
    summary = tl.reshape(summary, (heads_per_program, head_dim))
    summary_offsets = (
        (batch * heads + head_offsets[:, None]) * (storage_sequence_length // block_m)
        + storage_query_block
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
    if mask_block_lengths or mask_ragged_tail:
        group_starts = global_query_block * block_m + group_offsets * scale_rows
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
        (batch * heads + head_offsets[:, None, None]) * storage_sequence_length * head_dim
        + storage_sequence_offsets[None, :, None] * head_dim
        + feature_offsets[None, None, :]
    )
    tl.store(
        query_ptr + query_offsets,
        quantized,
        mask=head_offsets[:, None, None] < heads,
    )
    scale_offsets = (
        (batch * heads + head_offsets[:, None]) * (storage_sequence_length // scale_rows)
        + storage_query_block * (block_m // scale_rows)
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
    key_center_ptr,
    key_center_scale_ptr,
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
    """Quantize and store one K64-centered sparse-Piper key tile and route summaries."""
    head_offsets = head_offsets.to(tl.int64)
    feature_offsets = tl.arange(0, head_dim)
    local_tile_offsets = tl.arange(0, block_m // scale_rows)
    tile_offsets = (row_block * (block_m // scale_rows) + local_tile_offsets).to(tl.int64)
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
    key_summary, key_aux = summarize_block_tiles(
        grouped,
        valid,
        mean_pool_summary,
        tl.constexpr(False),
    )

    quantized, key_scale, center_0, center_1, center_scale_0, center_scale_1 = (
        qk_quantization.quantize_tile_centered_key_tile(
            values,
            valid,
            heads_per_program,
            head_dim,
            block_m,
            scale_rows,
        )
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
    center_offsets_0 = (scale_offsets * 2)[:, :, None] * head_dim + feature_offsets[None, None, :]
    center_offsets_1 = center_offsets_0 + head_dim
    tl.store(key_center_ptr + center_offsets_0, center_0, mask=tile_mask[:, :, None])
    tl.store(key_center_ptr + center_offsets_1, center_1, mask=tile_mask[:, :, None])
    tl.store(key_center_scale_ptr + scale_offsets * 2, center_scale_0, mask=tile_mask)
    tl.store(key_center_scale_ptr + scale_offsets * 2 + 1, center_scale_1, mask=tile_mask)
    summary_offsets = scale_offsets[:, :, None] * head_dim + feature_offsets[None, None, :]
    tl.store(key_summary_ptr + summary_offsets, key_summary, mask=tile_mask[:, :, None])
    if not mean_pool_summary:
        tl.store(key_aux_ptr + summary_offsets, key_aux, mask=tile_mask[:, :, None])


@triton.jit
def value_storage_offsets(
    head, row, feature, storage_sequence_length, head_dim: tl.constexpr, value_format: tl.constexpr
):
    """Map logical coordinates to the final V buffer; format IDs come from ValueFormat."""
    head = head.to(tl.int64)
    row = row.to(tl.int64)
    if value_format == _AMD_VALUE_FORMAT:
        token = pack_amd_wmma_token(row % 64)
        return (
            (head * (storage_sequence_length // 64) + row // 64) * head_dim + feature
        ) * 64 + token
    else:
        return (head * head_dim + feature) * storage_sequence_length + row


@triton.jit
def store_value_tile(
    values,
    value_ptr,
    value_scale_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    batch,
    heads,
    head_offsets,
    sequence_offsets,
    logical_sequence_length,
    storage_sequence_length,
    row_block,
    mask_block_lengths: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
    value_format: tl.constexpr = _TRANSPOSED_VALUE_FORMAT,
):
    """Per-K64 tile-center and quantize one sparse-Piper value tile."""
    head_offsets = head_offsets.to(tl.int64)
    feature_offsets = tl.arange(0, head_dim)
    tile_count = storage_sequence_length // scale_rows
    local_tile_offsets = tl.arange(0, block_m // scale_rows)
    tile_offsets = (row_block * (block_m // scale_rows) + local_tile_offsets).to(tl.int64)
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
    quantized, value_scale_multiplier, value_mean = quantize_value_tile(  # pyright: ignore[reportGeneralTypeIssues]
        values,
        valid_rows,
        heads_per_program,
        head_dim,
        block_m,
        scale_rows,
    )
    value_mean_offsets = scale_offsets[:, :, None] * head_dim + feature_offsets[None, None, :]
    tile_mask = (head_offsets[:, None] < heads) & (tile_offsets[None, :] < tile_count)
    tl.store(
        value_mean_ptr + value_mean_offsets,
        value_mean,
        mask=tile_mask[:, :, None],
    )
    value_offsets = value_storage_offsets(
        batch * heads + head_offsets[:, None, None],
        sequence_offsets[None, :, None],
        feature_offsets[None, None, :],
        storage_sequence_length,
        head_dim,
        value_format,
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
        mask=tile_mask,
    )
