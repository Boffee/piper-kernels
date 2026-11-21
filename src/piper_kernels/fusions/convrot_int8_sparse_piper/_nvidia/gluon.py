"""SM8x and SM120 Gluon kernels for the fused ConvRot INT8 sparse-Piper Q/K/V projections.

Each program projects 128 rows onto one D128 head with the shared projection
fragments, using ``cp.async`` copies on SM8x and TMA loads on SM120. Q emits
Q32 INT8 rows with routing summaries, and K emits K64-centered INT8 rows, scales,
2-term INT8 centers, and routing summaries in a single pass. Q/K programs run at
the 255-register limit. V projects its tiles through the shared value-tile fragment,
centers each K64 tile around its BF16 mean, and emits scaled transposed INT8 values
and means in a single pass.

The policy module selects shared Triton launchers for other shapes or unaligned operands.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._contracts import _MEAN_ROUTING
from piper_kernels.fusions.convrot_int8_projection._nvidia import fragments
from piper_kernels.fusions.convrot_int8_projection._nvidia._plan import NvidiaExecutionPlan

from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from .._layout import QUERY_SCALE_ROWS, TILE_ROWS

_HEAD_DIM = fragments.HEAD_DIM

_GL_HEAD_DIM = gl.constexpr(_HEAD_DIM)
_GL_BLOCK_M = gl.constexpr(fragments.BLOCK_M)
_GL_TILE_ROWS = gl.constexpr(TILE_ROWS)
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)
_GL_P_UINT8_RANGE = gl.constexpr(255.0)
_GL_VALUE_MEAN_SCALE_FLOOR = gl.constexpr(2.0**-80)
_GL_LOG2_E = gl.constexpr(1.4426950408889634)
_GL_FEATURE_LAYOUT = fragments.FEATURE_LAYOUT
_GL_SUMMARY_LAYOUT = gl.constexpr(
    gl.BlockedLayout([1, 4], [1, 32], [1, fragments.NUM_WARPS], [1, 0])
)

# Sequence lengths and offsets vary per call; specializing them would compile one
# kernel per divisibility class. Storage extents arrive as K64 tile counts, so the
# kernels still know that row strides are multiples of 64.
_DO_NOT_SPECIALIZE = ("logical_sequence_length", "row_block_offset", "storage_tiles")
_DO_NOT_SPECIALIZE_QUERY = (*_DO_NOT_SPECIALIZE, "chunk_start", "query_sequence_end")


@gluon.jit
def _minmax(maximum_0, minimum_0, maximum_1, minimum_1):
    """Combine (maximum, minimum) pairs in one reduction."""
    return gl.maximum(maximum_0, maximum_1), gl.minimum(minimum_0, minimum_1)


@gluon.jit
def _valid_rows(rows, positions, row_end, block_lengths_ptr, mask_block_lengths: gl.constexpr):
    """Intersect the row window with each K64 block's valid prefix.

    Block lengths are read at the clamped positions, which stay inside the sequence.
    """
    if mask_block_lengths:
        return (rows < row_end) & (
            rows % _GL_TILE_ROWS < gl.load(block_lengths_ptr + positions // _GL_TILE_ROWS)
        )
    return rows < row_end


@gluon.jit
def _summarize_block_means(block_values, block_mask, row_valid, mask_rows: gl.constexpr):
    """Reduce masked ``[blocks, 64, 128]`` values to per-block FP32 means."""
    blocks: gl.constexpr = block_values.shape[0]
    if mask_rows:
        summed_values = (
            block_values if block_mask is None else gl.where(block_mask, block_values, 0.0)
        )
        block_sum = gl.sum(summed_values, axis=1)
        # Count rows, not row-feature pairs: a per-feature count would hold 32 more
        # registers per thread through the reduction.
        row_counts = gl.reshape(row_valid.to(gl.int32), [blocks, _GL_TILE_ROWS])
        valid_count = gl.convert_layout(
            gl.sum(row_counts, axis=1), gl.SliceLayout(1, block_sum.type.layout)
        )
        return block_sum / valid_count[:, None]
    return gl.sum(block_values, axis=1) / _GL_TILE_ROWS


@gluon.jit
def _summarize_blocks(
    values,
    positions,
    features,
    first_row,
    row_end,
    block_lengths_ptr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    use_inf_pad: gl.constexpr,
):
    """Zero invalid rows and reduce each 64-row block to its routing summaries.

    Returns the masked tile with (primary, auxiliary) summaries: the block mean twice, or
    its maximum and minimum.
    """
    blocks: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    rows = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, _GL_FEATURE_LAYOUT))
    row_valid = _valid_rows(rows, positions, row_end, block_lengths_ptr, mask_block_lengths)
    if use_inf_pad:
        if mask_rows:
            values = gl.where(row_valid[:, None], values, 0.0)
        block_values = gl.reshape(values, [blocks, _GL_TILE_ROWS, _GL_HEAD_DIM])
        if mean_pool_summary:
            mean = _summarize_block_means(block_values, None, row_valid, mask_rows)
            return values, mean, mean, row_valid
        if mask_rows:
            block_valid_layout: gl.constexpr = gl.SliceLayout(2, block_values.type.layout)  # pyright: ignore[reportAttributeAccessIssue]
            block_valid = gl.convert_layout(
                gl.reshape(row_valid, [blocks, _GL_TILE_ROWS]), block_valid_layout
            )[:, :, None]
            # Padding invalid rows (which were just zeroed above) with +-inf via one mask
            # tensor keeps the key kernel within its 2-program SM8x/SM120 register limit.
            inf_pad = gl.where(block_valid, 0.0, float("inf"))
            maximum, minimum = gl.reduce(
                (block_values - inf_pad, block_values + inf_pad),
                1,
                _minmax,
            )
            return values, maximum, minimum, row_valid
        maximum, minimum = gl.reduce((block_values, block_values), 1, _minmax)
        return values, maximum, minimum, row_valid

    valid = row_valid[:, None] & (features >= 0)[None, :]
    if mask_rows:
        values = gl.where(valid, values, 0.0)
    block_values = gl.reshape(values, [blocks, _GL_TILE_ROWS, _GL_HEAD_DIM])
    block_valid = gl.reshape(valid, [blocks, _GL_TILE_ROWS, _GL_HEAD_DIM])
    if mean_pool_summary:
        mean = _summarize_block_means(block_values, block_valid, row_valid, mask_rows)
        return values, mean, mean, row_valid
    if mask_rows:
        maximum, minimum = gl.reduce(
            (
                gl.where(block_valid, block_values, -float("inf")),
                gl.where(block_valid, block_values, float("inf")),
            ),
            1,
            _minmax,
        )
    else:
        maximum, minimum = gl.reduce((block_values, block_values), 1, _minmax)
    return values, maximum, minimum, row_valid


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_QUERY)
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
    input_features: gl.constexpr,
    heads: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    softmax_scale: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a Q window tile and emit Q32 INT8 rows, scales, and Q64 summaries."""
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = chunk_start + row_block * _GL_BLOCK_M
    values, positions, features = fragments.project_rmsnorm_rope(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        batch,
        head,
        first_row,
        logical_sequence_length,
        input_features,
        rotary_dim,
        norm_epsilon,
        mask_rows,
    )

    values, maximum, minimum, _row_valid = _summarize_blocks(
        values,
        positions,
        features,
        first_row,
        query_sequence_end,
        block_lengths_ptr,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
        False,
    )
    summary = maximum
    if not mean_pool_summary:
        summary += minimum

    batch_head = batch * heads + head
    blocks: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    # Store routing summaries before the Hadamard transform so they do not stay live
    # alongside smoothed Q rows in the 255-register tail epilogue.
    fragments.store_summaries(
        query_summary_ptr + batch_head * storage_tiles * _GL_HEAD_DIM,
        summary,
        row_block * blocks,
        storage_tiles,
        mask_rows,
    )

    groups: gl.constexpr = _GL_BLOCK_M // _GL_QUERY_SCALE_ROWS
    smoothed = gl.reshape(
        fragments.signed_hadamard(values, features), [groups, _GL_QUERY_SCALE_ROWS, _GL_HEAD_DIM]
    )
    raw_scale = fragments.int8_scale(gl.max(gl.max(gl.abs(smoothed), axis=2), axis=1))
    if mask_rows:
        group_layout: gl.constexpr = raw_scale.type.layout  # pyright: ignore[reportAttributeAccessIssue]
        group_starts = first_row + gl.arange(0, groups, group_layout) * _GL_QUERY_SCALE_ROWS
        group_valid = _valid_rows(
            group_starts,
            gl.minimum(group_starts, logical_sequence_length - 1),
            query_sequence_end,
            block_lengths_ptr,
            mask_block_lengths,
        )
        raw_scale = gl.where(group_valid, raw_scale, 1.0)
        stored_scale = gl.where(group_valid, raw_scale * (softmax_scale * _GL_LOG2_E), 0.0)
    else:
        stored_scale = raw_scale * (softmax_scale * _GL_LOG2_E)
    quantized = fragments.round_to_int8(smoothed / raw_scale[:, None, None])

    storage_rows = storage_tiles * _GL_TILE_ROWS
    storage_groups = storage_rows // _GL_QUERY_SCALE_ROWS
    fragments.store_rows(
        query_ptr + batch_head.to(gl.int64) * storage_rows * _GL_HEAD_DIM,
        quantized,
        row_block * _GL_BLOCK_M,
        storage_rows,
        mask_rows,
    )
    fragments.store_scales(
        query_scale_ptr + batch_head * storage_groups,
        stored_scale,
        row_block * groups,
        storage_groups,
        mask_rows,
    )


@gluon.jit
def _tile_mean(values, row_valid, mask_rows: gl.constexpr, mask_values: gl.constexpr):
    """Compute per-K64 tile FP32 means; keys arrive pre-zeroed from _summarize_blocks."""
    if mask_rows:
        if mask_values:
            values = gl.where(row_valid[:, :, None], values, 0.0)
        tile_sum = gl.sum(values, axis=1)
        valid_count = gl.convert_layout(
            gl.maximum(gl.sum(row_valid.to(gl.int32), axis=1), 1),
            gl.SliceLayout(1, tile_sum.type.layout),
        )
        return tile_sum / valid_count[:, None]
    return gl.sum(values, axis=1) / _GL_TILE_ROWS


@gluon.jit
def _tile_midpoint(values, tile_row_valid, mask_rows: gl.constexpr):
    """Compute per-K64 tile FP32 midpoints over rotated keys."""
    if mask_rows:
        tile_max, tile_min = gl.reduce(
            (
                gl.where(tile_row_valid, values, -float("inf")),
                gl.where(tile_row_valid, values, float("inf")),
            ),
            1,
            _minmax,
        )
        # Empty tiles reduce to (-inf, +inf); finite non-empty tiles always satisfy min <= max.
        return gl.where(tile_max >= tile_min, 0.5 * (tile_max + tile_min), 0.0)
    tile_max, tile_min = gl.reduce((values, values), 1, _minmax)
    return 0.5 * (tile_max + tile_min)


@gluon.jit
def _store_key_centers(
    key_center_ptr,
    key_center_scale_ptr,
    center_0,
    center_1,
    center_scale_0,
    center_scale_1,
    batch_head,
    first_tile,
    storage_tiles,
    mask_rows: gl.constexpr,
):
    """Store [tiles, 2, D128] INT8 key centers and FP32 center scales."""
    tiles: gl.constexpr = center_0.shape[0]
    tile_row = batch_head.to(gl.int64) * storage_tiles
    tile_layout: gl.constexpr = gl.SliceLayout(1, _GL_SUMMARY_LAYOUT)
    tile_offsets = first_tile + gl.arange(0, tiles, tile_layout)
    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, _GL_SUMMARY_LAYOUT))
    center_base = (
        key_center_ptr + (tile_row + tile_offsets[:, None]) * (2 * _GL_HEAD_DIM) + features[None, :]
    )
    scale_base = key_center_scale_ptr + (tile_row + tile_offsets) * 2
    center_0 = gl.convert_layout(center_0, _GL_SUMMARY_LAYOUT)
    center_1 = gl.convert_layout(center_1, _GL_SUMMARY_LAYOUT)
    center_scale_0 = gl.convert_layout(center_scale_0, tile_layout)
    center_scale_1 = gl.convert_layout(center_scale_1, tile_layout)
    if mask_rows:
        tile_mask = tile_offsets < storage_tiles
        gl.store(center_base, center_0, mask=tile_mask[:, None])
        gl.store(center_base + _GL_HEAD_DIM, center_1, mask=tile_mask[:, None])
        gl.store(scale_base, center_scale_0, mask=tile_mask)
        gl.store(scale_base + 1, center_scale_1, mask=tile_mask)
    else:
        gl.store(center_base, center_0)
        gl.store(center_base + _GL_HEAD_DIM, center_1)
        gl.store(scale_base, center_scale_0)
        gl.store(scale_base + 1, center_scale_1)


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE)
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
    input_features: gl.constexpr,
    heads: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a K tile and emit K64-centered INT8 keys, scales, centers, and summaries."""
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = row_block * _GL_BLOCK_M
    values, positions, features = fragments.project_rmsnorm_rope(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        batch,
        head,
        first_row,
        logical_sequence_length,
        input_features,
        rotary_dim,
        norm_epsilon,
        mask_rows,
    )

    values, key_summary, key_aux, row_valid = _summarize_blocks(
        values,
        positions,
        features,
        first_row,
        logical_sequence_length,
        block_lengths_ptr,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
        True,
    )

    batch_head = batch * heads + head
    tiles: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    first_tile = row_block * tiles
    tile_row = batch_head * storage_tiles
    fragments.store_summaries(
        key_summary_ptr + tile_row * _GL_HEAD_DIM, key_summary, first_tile, storage_tiles, mask_rows
    )
    if not mean_pool_summary:
        fragments.store_summaries(
            key_aux_ptr + tile_row * _GL_HEAD_DIM, key_aux, first_tile, storage_tiles, mask_rows
        )

    rotated = gl.reshape(
        fragments.signed_hadamard(values, features), [tiles, _GL_TILE_ROWS, _GL_HEAD_DIM]
    )
    if mask_rows:
        row_mask_layout: gl.constexpr = gl.SliceLayout(2, rotated.type.layout)  # pyright: ignore[reportAttributeAccessIssue]
        tile_row_valid = gl.convert_layout(
            gl.reshape(row_valid, [tiles, _GL_TILE_ROWS]), row_mask_layout
        )[:, :, None]
    else:
        tile_row_valid = None
    rotated_center = _tile_midpoint(rotated, tile_row_valid, mask_rows)

    centered = rotated - rotated_center[:, None, :]
    if mask_rows:
        centered = gl.where(tile_row_valid, centered, 0.0)
    key_scale = fragments.int8_scale(gl.max(gl.max(gl.abs(centered), axis=2), axis=1))
    quantized = fragments.round_to_int8(centered / key_scale[:, None, None])

    storage_rows = storage_tiles * _GL_TILE_ROWS
    fragments.store_rows(
        key_ptr + batch_head.to(gl.int64) * storage_rows * _GL_HEAD_DIM,
        quantized,
        first_row,
        storage_rows,
        mask_rows,
    )
    fragments.store_scales(
        key_scale_ptr + tile_row,
        key_scale,
        first_tile,
        storage_tiles,
        mask_rows,
    )

    # Encode and store the 2-term rotated midpoint after freeing the 128-reg tile buffer.
    center_scale_0 = fragments.int8_scale(gl.max(gl.abs(rotated_center), axis=1))
    center_0 = fragments.round_to_int8(rotated_center / center_scale_0[:, None])
    residual = rotated_center - center_0.to(gl.float32) * center_scale_0[:, None]
    center_scale_1 = fragments.int8_scale(gl.max(gl.abs(residual), axis=1))
    center_1 = fragments.round_to_int8(residual / center_scale_1[:, None])
    _store_key_centers(
        key_center_ptr,
        key_center_scale_ptr,
        center_0,
        center_1,
        center_scale_0,
        center_scale_1,
        batch_head,
        first_tile,
        storage_tiles,
        mask_rows,
    )


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE)
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
    input_features: gl.constexpr,
    heads: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a V tile and emit K64 tile-centered transposed INT8 values and BF16 means."""
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    projection, sequence, tile_offsets, rows_in_tile, features = fragments.project_value_tiles(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        batch,
        head,
        row_block,
        logical_sequence_length,
        input_features,
        _GL_TILE_ROWS,
        mask_rows,
    )
    tiles: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    first_tile = row_block * tiles

    if mask_block_lengths:
        tile_length = gl.load(
            block_lengths_ptr + tile_offsets, mask=tile_offsets < storage_tiles, other=0
        )
        valid = rows_in_tile[None, :] < tile_length[:, None]
    else:
        valid = sequence < logical_sequence_length
    batch_head = batch * heads + head
    tile_row = batch_head * storage_tiles
    mean_fp32 = _tile_mean(projection, valid, mask_rows, True)
    value_mean = mean_fp32.to(gl.bfloat16)
    fragments.store_summaries(
        value_mean_ptr + tile_row * _GL_HEAD_DIM,
        value_mean,
        first_tile,
        storage_tiles,
        mask_rows,
    )
    centered = projection - value_mean.to(gl.float32)[:, None, :]
    if mask_rows:
        centered = gl.where(valid[:, :, None], centered, 0.0)
    residual_scale = fragments.int8_scale(gl.max(gl.max(gl.abs(centered), axis=2), axis=1))
    mean_magnitude = gl.convert_layout(
        gl.max(gl.abs(value_mean.to(gl.float32)), axis=1),
        residual_scale.type.layout,
    )
    value_scale = gl.maximum(
        residual_scale,
        mean_magnitude * _GL_VALUE_MEAN_SCALE_FLOOR,
    )
    quantized = fragments.round_to_int8(centered / value_scale[:, None, None])

    storage_rows = storage_tiles * _GL_TILE_ROWS
    fragments.store_value_codes(
        value_ptr + batch_head.to(gl.int64) * _GL_HEAD_DIM * storage_rows,
        quantized,
        sequence,
        features,
        tile_offsets,
        storage_tiles,
        mask_rows,
    )
    fragments.store_scales(
        value_scale_ptr + tile_row,
        value_scale * _GL_P_UINT8_RANGE,
        first_tile,
        storage_tiles,
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
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill Q32 INT8 queries, scales, and Q64 summaries for a supported query window."""
    query, query_scale, query_summary = out
    batch, heads, storage_sequence_length, _head_dim = query.shape
    if batch == 0:
        return
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):
        input_operand, weight_operand = fragments.resolve_operands(
            input_qdata, weight_qdata, execution_plan
        )

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _query_kernel[(heads, row_blocks, batch)](
                input_operand,
                input_scale,
                weight_operand,
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
                input_features=input_qdata.shape[2],
                heads=heads,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        fragments.launch_row_blocks(chunk_rows, launch, mask_full_tiles=mask_block_lengths)


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
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill K64-centered INT8 keys, scales, centers, and routing summaries."""
    key, key_scale, key_center, key_center_scale, key_summary, key_aux = out
    batch, heads, storage_sequence_length, _head_dim = key.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0:
        return
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):
        input_operand, weight_operand = fragments.resolve_operands(
            input_qdata, weight_qdata, execution_plan
        )

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _key_kernel[(heads, row_blocks, batch)](
                input_operand,
                input_scale,
                weight_operand,
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
                input_features=input_qdata.shape[2],
                heads=heads,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        fragments.launch_row_blocks(sequence_length, launch, mask_full_tiles=mask_block_lengths)


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
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill K64 tile-centered INT8 values, multipliers, and BF16 per-tile means."""
    if value_format is not ValueFormat.TRANSPOSED:
        raise ValueError("NVIDIA Gluon V projection requires transposed storage")
    value, value_scale_multiplier, value_mean = out
    batch, heads, _head_dim, storage_sequence_length = value.shape
    if batch == 0:
        return
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):
        input_operand, weight_operand = fragments.resolve_operands(
            input_qdata, weight_qdata, execution_plan
        )

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _value_kernel[(heads, row_blocks, batch)](
                input_operand,
                input_scale,
                weight_operand,
                weight_scale,
                bias,
                value,
                value_scale_multiplier,
                value_mean,
                block_lengths if mask_block_lengths else value_scale_multiplier,
                input_qdata.shape[1],
                row_block_offset,
                storage_sequence_length // TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        fragments.launch_row_blocks(
            input_qdata.shape[1], launch, mask_full_tiles=mask_block_lengths
        )
