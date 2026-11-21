"""RDNA4 Gluon kernels for fused ConvRot INT8 sparse-Piper Q/K/V projections.

Uses the shared 4x2 wave32 (128x128x64, 16 KB LDS) projection fragments and adds
sparse-only K64 routing summaries, two-term K64 centers, and optional ``block_lengths``
masking.
"""

# Gluon exposes low-level signatures and layouts that are not modeled by static type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._contracts import _MEAN_ROUTING
from piper_kernels.fusions.convrot_int8_projection._amd import fragments
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from .._layout import TILE_ROWS, resolve_row_block_launches
from .policy import GLUON_BLOCK_M

# One-barrier cross-M-warp column reduction layouts (distributes 128 columns across warps/lanes).
_MINMAX_REDUCE_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0]],
        lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 0]],
        warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]],
        block_bases=[],
        shape=[2, 4, 128],
    )
)
_MEAN_REDUCE_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0]],
        lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        warp_bases=[[0, 32], [0, 64], [0, 0]],
        block_bases=[],
        shape=[4, 128],
    )
)

_build_kernel_options = fragments.build_kernel_options
operands_aligned = fragments.operands_aligned


def _build_key_kernel_options(
    head_dim: int,
    rotary_dim: int,
    *,
    mask_rows: bool,
    mask_block_lengths: bool,
    mean_pool_summary: bool,
) -> dict[str, object]:
    """Keep 6-8 waves/EU on D64 and D128 R<=96 fast paths; relax ragged minmax tails and R128."""
    use_occupancy_hint = head_dim == 64 or (
        rotary_dim < 128 and (not mask_rows or mask_block_lengths or mean_pool_summary)
    )
    return _build_kernel_options(waves_per_eu="6,8" if use_occupancy_hint else None)


@gluon.jit
def _reduce_column_minmax(values_64x128, row_valid, mask_rows: gl.constexpr):
    """Exact columnwise ``(max, min)`` of ``[64, 128]`` using one cross-M-warp barrier."""
    if mask_rows:
        valid = gl.convert_layout(row_valid, gl.SliceLayout(1, values_64x128.type.layout))
        inf_pad = gl.where(valid[:, None], 0.0, float("inf"))
        max_input = gl.reshape(values_64x128 - inf_pad, [4, 16, 128])
        min_input = gl.reshape(values_64x128 + inf_pad, [4, 16, 128])
        warp_max = gl.max(max_input, axis=1)
        warp_min = gl.min(min_input, axis=1)
    else:
        grouped = gl.reshape(values_64x128, [4, 16, 128])
        warp_max = gl.max(grouped, axis=1)
        warp_min = gl.min(grouped, axis=1)
    packed = gl.permute(gl.join(warp_max, warp_min), [2, 0, 1])
    per_column = gl.convert_layout(packed, _MINMAX_REDUCE_LAYOUT)
    reg_max, reg_min = gl.split(gl.permute(per_column, [1, 2, 0]))
    return gl.max(reg_max, axis=0), gl.min(reg_min, axis=0)


@gluon.jit
def _reduce_column_mean(values_64x128, valid_count_f32):
    """Exact columnwise mean of zero-padded ``[64, 128]`` using one cross-M-warp barrier."""
    warp_sum = gl.sum(gl.reshape(values_64x128, [4, 16, 128]), axis=1)
    per_column = gl.convert_layout(warp_sum, _MEAN_REDUCE_LAYOUT)
    return gl.sum(per_column, axis=0) / valid_count_f32


@gluon.jit
def _store_tile_summary(
    summary_ptr,
    values,
    batch_head_index,
    tile_index,
    storage_tiles,
    head_dim: gl.constexpr,
):
    column_offsets = gl.arange(0, 128, values.type.layout)
    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    if head_dim == 64:
        head_offset = (column_offsets // 64).to(gl.int64)
        feature_offset = column_offsets % 64
        offsets = ((batch_head_64 + head_offset) * storage_tiles_64 + tile_index) * 64 + (
            feature_offset
        )
    else:
        offsets = (batch_head_64 * storage_tiles_64 + tile_index) * 128 + column_offsets
    gl.store(summary_ptr + offsets, values)


@gluon.jit
def _store_query_half(
    values,
    cos_ptr,
    sin_ptr,
    query_base_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    batch_head_index,
    storage_tile,
    global_tile,
    storage_tiles,
    seq_first_row,
    query_sequence_end,
    logical_sequence_length,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    softmax_scale: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    values, row_valid, block_length = fragments.rope_and_mask_half(
        values,
        cos_ptr,
        sin_ptr,
        seq_first_row,
        query_sequence_end,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mask_rows,
        block_lengths_ptr,
        global_tile,
        mask_block_lengths,
    )
    if mean_pool_summary:
        valid_count = (
            gl.sum(row_valid.to(gl.int32), axis=0).to(gl.float32)
            if mask_rows
            else fragments.GL_TILE_ROWS_F32
        )
        summary = _reduce_column_mean(values, valid_count)
    else:
        column_max, column_min = _reduce_column_minmax(values, row_valid, mask_rows)
        summary = column_max + column_min
    _store_tile_summary(
        query_summary_ptr, summary, batch_head_index, storage_tile, storage_tiles, head_dim
    )

    fragments.quantize_and_store_query_half(
        values,
        query_base_ptr,
        query_scale_ptr,
        batch_head_index,
        storage_tile,
        storage_tiles,
        seq_first_row,
        query_sequence_end,
        block_length,
        head_dim,
        softmax_scale,
        mask_block_lengths,
        mask_rows,
    )


@gluon.jit
def _store_key_half(
    values,
    cos_ptr,
    sin_ptr,
    key_base_ptr,
    key_scale_ptr,
    key_center_base_ptr,
    key_center_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    batch_head_index,
    tile_index,
    storage_tiles,
    seq_first_row,
    logical_sequence_length,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    values, row_valid, _block_length = fragments.rope_and_mask_half(
        values,
        cos_ptr,
        sin_ptr,
        seq_first_row,
        logical_sequence_length,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mask_rows,
        block_lengths_ptr,
        tile_index,
        mask_block_lengths,
    )
    if mean_pool_summary:
        valid_count = (
            gl.sum(row_valid.to(gl.int32), axis=0).to(gl.float32)
            if mask_rows
            else fragments.GL_TILE_ROWS_F32
        )
        key_summary = _reduce_column_mean(values, valid_count)
        _store_tile_summary(
            key_summary_ptr, key_summary, batch_head_index, tile_index, storage_tiles, head_dim
        )
    else:
        column_max, column_min = _reduce_column_minmax(values, row_valid, mask_rows)
        _store_tile_summary(
            key_summary_ptr, column_max, batch_head_index, tile_index, storage_tiles, head_dim
        )
        _store_tile_summary(
            key_aux_ptr, column_min, batch_head_index, tile_index, storage_tiles, head_dim
        )

    column_offsets = gl.arange(0, 128, gl.SliceLayout(0, fragments.TILE_HALF_LAYOUT))
    values = values * fragments.compute_hadamard_signs(column_offsets % head_dim)[None, :]
    hadamard_64 = fragments.apply_hadamard_64(gl.reshape(values, [64, 2, 64]))
    if head_dim == 64:
        rotated = gl.reshape(hadamard_64 * fragments.GL_D64_NORM, [64, 128])
    else:
        rotated = fragments.apply_hadamard_stage64(hadamard_64) * fragments.GL_D128_NORM

    rotated_max, rotated_min = _reduce_column_minmax(rotated, row_valid, mask_rows)
    if mask_rows:
        center = gl.where(rotated_max >= rotated_min, 0.5 * (rotated_max + rotated_min), 0.0)
    else:
        center = 0.5 * (rotated_max + rotated_min)

    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    storage_rows = storage_tiles * fragments.GL_TILE_ROWS

    # Quantize & store the 2-term rotated center in 1-VGPR/thread layout before centered keys.
    if head_dim == 64:
        center_heads = gl.reshape(center, [2, 64])
        center_scale_0 = fragments.compute_int8_scale(gl.max(gl.abs(center_heads), axis=1))
        center_codes_0 = fragments.round_to_int8(center_heads / center_scale_0[:, None])
        residual = center_heads - center_codes_0.to(gl.float32) * center_scale_0[:, None]
        center_scale_1 = fragments.compute_int8_scale(gl.max(gl.abs(residual), axis=1))
        center_codes_1 = fragments.round_to_int8(residual / center_scale_1[:, None])
        center_layout: gl.constexpr = center_codes_0.type.layout
        head_indices = gl.arange(0, 2, gl.SliceLayout(1, center_layout))
        feature_indices = gl.arange(0, 64, gl.SliceLayout(0, center_layout))
        center_offsets = (head_indices[:, None] * storage_tiles_64 + tile_index) * (
            2 * 64
        ) + feature_indices[None, :]
        cdna3.buffer_store(center_codes_0, key_center_base_ptr, center_offsets.to(gl.int32))
        cdna3.buffer_store(center_codes_1, key_center_base_ptr, (center_offsets + 64).to(gl.int32))
        scale_heads = gl.arange(0, 2, center_scale_0.type.layout)
        center_scale_offsets = (
            (batch_head_64 + scale_heads.to(gl.int64)) * storage_tiles_64 + tile_index
        ) * 2
        gl.store(key_center_scale_ptr + center_scale_offsets, center_scale_0)
        gl.store(key_center_scale_ptr + center_scale_offsets + 1, center_scale_1)
    else:
        center_scale_0 = fragments.compute_int8_scale(gl.max(gl.abs(center), axis=0))
        center_codes_0 = fragments.round_to_int8(center / center_scale_0)
        residual = center - center_codes_0.to(gl.float32) * center_scale_0
        center_scale_1 = fragments.compute_int8_scale(gl.max(gl.abs(residual), axis=0))
        center_codes_1 = fragments.round_to_int8(residual / center_scale_1)
        feature_indices = gl.arange(0, 128, center_codes_0.type.layout)
        center_offsets = tile_index * (2 * 128) + feature_indices
        cdna3.buffer_store(center_codes_0, key_center_base_ptr, center_offsets)
        cdna3.buffer_store(center_codes_1, key_center_base_ptr, center_offsets + 128)
        center_scale_offsets = (batch_head_64 * storage_tiles_64 + tile_index) * 2
        gl.store(key_center_scale_ptr + center_scale_offsets, center_scale_0)
        gl.store(key_center_scale_ptr + center_scale_offsets + 1, center_scale_1)

    center_row = gl.convert_layout(center, gl.SliceLayout(0, rotated.type.layout))
    centered = rotated - center_row[None, :]
    if mask_rows:
        row_valid_rot = gl.convert_layout(row_valid, gl.SliceLayout(1, centered.type.layout))
        centered = gl.where(row_valid_rot[:, None], centered, 0.0)
    if head_dim == 64:
        centered_heads = gl.reshape(centered, [64, 2, 64])
        key_scale = fragments.compute_int8_scale(
            gl.max(gl.max(gl.abs(centered_heads), axis=2), axis=0)
        )
        key_codes = gl.reshape(
            fragments.round_to_int8(centered_heads / key_scale[None, :, None]), [64, 128]
        )
        fragments.store_head_major(key_base_ptr, key_codes, tile_index, storage_rows, head_dim)
        scale_heads = gl.arange(0, 2, key_scale.type.layout)
        gl.store(
            key_scale_ptr
            + (batch_head_64 + scale_heads.to(gl.int64)) * storage_tiles_64
            + tile_index,
            key_scale,
        )
    else:
        key_scale = fragments.compute_int8_scale(gl.max(gl.max(gl.abs(centered), axis=1), axis=0))
        key_codes = fragments.round_to_int8(centered / key_scale)
        fragments.store_head_major(key_base_ptr, key_codes, tile_index, storage_rows, head_dim)
        gl.store(key_scale_ptr + batch_head_64 * storage_tiles_64 + tile_index, key_scale)


@gluon.jit
def _store_value_half(
    values,
    value_base_ptr,
    value_scale_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    batch_head_index,
    tile_index,
    storage_tiles,
    seq_first_row,
    logical_sequence_length,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    if mask_rows:
        rows_local = gl.arange(0, 64, gl.SliceLayout(1, fragments.TILE_HALF_LAYOUT))
        row_valid = (seq_first_row + rows_local) < logical_sequence_length
        if mask_block_lengths:
            block_length = gl.load(
                block_lengths_ptr + tile_index,
                mask=seq_first_row < logical_sequence_length,
                other=0,
            )
            row_valid = row_valid & (rows_local < block_length)
        values = gl.where(row_valid[:, None], values, 0.0)
        valid_count = gl.maximum(gl.sum(row_valid.to(gl.int32), axis=0), 1).to(gl.float32)
        mean_bf16 = _reduce_column_mean(values, valid_count).to(gl.bfloat16)
    else:
        row_valid = None
        mean_bf16 = _reduce_column_mean(values, fragments.GL_TILE_ROWS_F32).to(gl.bfloat16)
    _store_tile_summary(
        value_mean_ptr, mean_bf16, batch_head_index, tile_index, storage_tiles, head_dim
    )

    mean_f32 = mean_bf16.to(gl.float32)
    centered = (
        values - gl.convert_layout(mean_f32, gl.SliceLayout(0, fragments.TILE_HALF_LAYOUT))[None, :]
    )
    if mask_rows and row_valid is not None:
        centered = gl.where(row_valid[:, None], centered, 0.0)

    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    if head_dim == 64:
        centered_heads = gl.reshape(centered, [64, 2, 64])
        residual_scale = fragments.compute_int8_scale(
            gl.max(gl.max(gl.abs(centered_heads), axis=2), axis=0)
        )
        mean_magnitude = gl.convert_layout(
            gl.max(gl.abs(gl.reshape(mean_f32, [2, 64])), axis=1), residual_scale.type.layout
        )
        value_scale = gl.maximum(
            residual_scale, mean_magnitude * fragments.GL_VALUE_MEAN_SCALE_FLOOR
        )
        value_codes = gl.reshape(
            fragments.round_to_int8(centered_heads / value_scale[None, :, None]), [64, 128]
        )
        scale_heads = gl.arange(0, 2, value_scale.type.layout)
        gl.store(
            value_scale_ptr
            + (batch_head_64 + scale_heads.to(gl.int64)) * storage_tiles_64
            + tile_index,
            value_scale * fragments.GL_P_UINT8_RANGE,
        )
    else:
        residual_scale = fragments.compute_int8_scale(
            gl.max(gl.max(gl.abs(centered), axis=1), axis=0)
        )
        mean_magnitude = gl.max(gl.abs(mean_f32), axis=0)
        value_scale = gl.maximum(
            residual_scale, mean_magnitude * fragments.GL_VALUE_MEAN_SCALE_FLOOR
        )
        value_codes = fragments.round_to_int8(centered / value_scale)
        gl.store(
            value_scale_ptr + batch_head_64 * storage_tiles_64 + tile_index,
            value_scale * fragments.GL_P_UINT8_RANGE,
        )

    fragments.store_amd_wmma_value_codes(
        value_base_ptr, value_codes, tile_index, storage_tiles, head_dim
    )


@gluon.jit(do_not_specialize=fragments.DO_NOT_SPECIALIZE_QUERY)
def _query_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    chunk_start,
    query_sequence_end,
    storage_tiles,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    softmax_scale: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a Q window tile and emit Q32 INT8 rows, scales, and Q64 summaries."""
    half_top, half_bot, batch_head_index, batch_head_stride, storage_tile_top, seq_first_row = (
        fragments.project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            norm_weight_ptr,
            row_block_offset,
            chunk_start,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            norm_epsilon,
            True,
            mask_rows,
            group_m,
        )
    )
    query_base_ptr = query_ptr + batch_head_stride * fragments.GL_TILE_ROWS * head_dim
    chunk_tile_base = chunk_start // fragments.GL_TILE_ROWS
    _store_query_half(
        half_top,
        cos_ptr,
        sin_ptr,
        query_base_ptr,
        query_scale_ptr,
        query_summary_ptr,
        block_lengths_ptr,
        batch_head_index,
        storage_tile_top,
        chunk_tile_base + storage_tile_top,
        storage_tiles,
        seq_first_row,
        query_sequence_end,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        softmax_scale,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
    )
    storage_tile_bot = storage_tile_top + 1
    if (not mask_rows) or (
        (storage_tile_bot < storage_tiles)
        and (seq_first_row + fragments.GL_TILE_ROWS < query_sequence_end)
    ):
        _store_query_half(
            half_bot,
            cos_ptr,
            sin_ptr,
            query_base_ptr,
            query_scale_ptr,
            query_summary_ptr,
            block_lengths_ptr,
            batch_head_index,
            storage_tile_bot,
            chunk_tile_base + storage_tile_bot,
            storage_tiles,
            seq_first_row + fragments.GL_TILE_ROWS,
            query_sequence_end,
            logical_sequence_length,
            head_dim,
            rotary_dim,
            softmax_scale,
            mean_pool_summary,
            mask_block_lengths,
            mask_rows,
        )


@gluon.jit(do_not_specialize=fragments.DO_NOT_SPECIALIZE)
def _key_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    key_ptr,
    key_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a K tile and emit K64-centered INT8 keys, scales, centers, and summaries."""
    half_top, half_bot, batch_head_index, batch_head_stride, tile_top, seq_first_row = (
        fragments.project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            norm_weight_ptr,
            row_block_offset,
            0,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            norm_epsilon,
            True,
            mask_rows,
            group_m,
        )
    )
    key_base_ptr = key_ptr + batch_head_stride * fragments.GL_TILE_ROWS * head_dim
    key_center_base_ptr = key_center_ptr + batch_head_stride * (2 * head_dim)
    _store_key_half(
        half_top,
        cos_ptr,
        sin_ptr,
        key_base_ptr,
        key_scale_ptr,
        key_center_base_ptr,
        key_center_scale_ptr,
        key_summary_ptr,
        key_aux_ptr,
        block_lengths_ptr,
        batch_head_index,
        tile_top,
        storage_tiles,
        seq_first_row,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
    )
    tile_bot = tile_top + 1
    if (not mask_rows) or (
        (tile_bot < storage_tiles)
        and (seq_first_row + fragments.GL_TILE_ROWS < logical_sequence_length)
    ):
        _store_key_half(
            half_bot,
            cos_ptr,
            sin_ptr,
            key_base_ptr,
            key_scale_ptr,
            key_center_base_ptr,
            key_center_scale_ptr,
            key_summary_ptr,
            key_aux_ptr,
            block_lengths_ptr,
            batch_head_index,
            tile_bot,
            storage_tiles,
            seq_first_row + fragments.GL_TILE_ROWS,
            logical_sequence_length,
            head_dim,
            rotary_dim,
            mean_pool_summary,
            mask_block_lengths,
            mask_rows,
        )


@gluon.jit(do_not_specialize=fragments.DO_NOT_SPECIALIZE)
def _value_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    value_ptr,
    value_scale_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a V tile and emit K64 tile-centered AMD-WMMA INT8 values and BF16 means."""
    half_top, half_bot, batch_head_index, batch_head_stride, tile_top, seq_first_row = (
        fragments.project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            None,
            row_block_offset,
            0,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            1e-5,
            False,
            mask_rows,
            group_m,
        )
    )
    value_base_ptr = value_ptr + batch_head_stride * fragments.GL_TILE_ROWS * head_dim
    _store_value_half(
        half_top,
        value_base_ptr,
        value_scale_ptr,
        value_mean_ptr,
        block_lengths_ptr,
        batch_head_index,
        tile_top,
        storage_tiles,
        seq_first_row,
        logical_sequence_length,
        head_dim,
        mask_block_lengths,
        mask_rows,
    )
    tile_bot = tile_top + 1
    if (not mask_rows) or (
        (tile_bot < storage_tiles)
        and (seq_first_row + fragments.GL_TILE_ROWS < logical_sequence_length)
    ):
        _store_value_half(
            half_bot,
            value_base_ptr,
            value_scale_ptr,
            value_mean_ptr,
            block_lengths_ptr,
            batch_head_index,
            tile_bot,
            storage_tiles,
            seq_first_row + fragments.GL_TILE_ROWS,
            logical_sequence_length,
            head_dim,
            mask_block_lengths,
            mask_rows,
        )


def project_query(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    chunk_start: int,
    chunk_rows: int,
    out: QueryOutput,
    bias: torch.Tensor | None = None,
    execution_plan: ProjectionExecutionPlan,
) -> None:
    """Fill Q32 INT8 queries, scales, and Q64 summaries via RDNA4 Gluon kernels."""
    query, query_scale, query_summary = out
    batch, heads, storage_sequence_length, head_dim = query.shape
    if batch == 0 or chunk_rows == 0:
        return
    mask_block_lengths = block_lengths is not None
    col_blocks = (heads * head_dim) // 128
    options = _build_kernel_options()

    with device_context(input_qdata.device):
        for row_blocks, row_block_offset, aligned_rows in resolve_row_block_launches(
            chunk_rows, GLUON_BLOCK_M
        ):
            _query_kernel[(row_blocks * col_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                norm_weight,
                cos,
                sin,
                query,
                query_scale,
                query_summary,
                block_lengths if mask_block_lengths else query_scale,
                input_qdata.shape[1],
                row_block_offset,
                chunk_start,
                chunk_start + chunk_rows,
                storage_sequence_length // TILE_ROWS,
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=mask_block_lengths,
                mask_rows=(not aligned_rows) or mask_block_lengths,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                **options,
            )


def project_key(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    out: KeyOutput,
    bias: torch.Tensor | None = None,
    execution_plan: ProjectionExecutionPlan,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    """Fill K64-centered INT8 keys, scales, 2-term centers, and summaries via RDNA4 Gluon."""
    key, key_scale, key_center, key_center_scale, key_summary, key_aux = out
    batch, heads, storage_sequence_length, head_dim = key.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0 or sequence_length == 0:
        return
    mask_block_lengths = block_lengths is not None
    col_blocks = (heads * head_dim) // 128
    rotary_dim = cos.shape[1]
    mean_pool_summary = routing_mode == _MEAN_ROUTING

    with device_context(input_qdata.device):
        for row_blocks, row_block_offset, aligned_rows in resolve_row_block_launches(
            sequence_length,
            GLUON_BLOCK_M,
            start_row_block=start_row_block,
            num_row_blocks=num_row_blocks,
        ):
            mask_rows = (not aligned_rows) or mask_block_lengths
            options = _build_key_kernel_options(
                head_dim,
                rotary_dim,
                mask_rows=mask_rows,
                mask_block_lengths=mask_block_lengths,
                mean_pool_summary=mean_pool_summary,
            )
            _key_kernel[(row_blocks * col_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                norm_weight,
                cos,
                sin,
                key,
                key_scale,
                key_center,
                key_center_scale,
                key_summary,
                key_aux,
                block_lengths if mask_block_lengths else key_scale,
                sequence_length,
                row_block_offset,
                storage_sequence_length // TILE_ROWS,
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                norm_epsilon=norm_epsilon,
                mean_pool_summary=mean_pool_summary,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                **options,
            )


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    value_format: ValueFormat,
    out: ValueOutput,
    bias: torch.Tensor | None = None,
    execution_plan: ProjectionExecutionPlan,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    """Fill K64 tile-centered INT8 values, multipliers, and BF16 means via RDNA4 Gluon."""
    if value_format is not ValueFormat.AMD_WMMA:
        raise ValueError("RDNA4 Gluon V projection requires ValueFormat.AMD_WMMA storage")
    value, value_scale_multiplier, value_mean = out
    batch, heads, storage_tiles, head_dim = value_mean.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0 or sequence_length == 0:
        return
    mask_block_lengths = block_lengths is not None
    col_blocks = (heads * head_dim) // 128
    options = _build_kernel_options()

    with device_context(input_qdata.device):
        for row_blocks, row_block_offset, aligned_rows in resolve_row_block_launches(
            sequence_length,
            GLUON_BLOCK_M,
            start_row_block=start_row_block,
            num_row_blocks=num_row_blocks,
        ):
            _value_kernel[(row_blocks * col_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                value,
                value_scale_multiplier,
                value_mean,
                block_lengths if mask_block_lengths else value_scale_multiplier,
                sequence_length,
                row_block_offset,
                storage_tiles,
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                mask_block_lengths=mask_block_lengths,
                mask_rows=(not aligned_rows) or mask_block_lengths,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                **options,
            )
