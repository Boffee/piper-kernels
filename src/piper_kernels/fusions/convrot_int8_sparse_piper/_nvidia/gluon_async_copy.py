"""SM89 Gluon kernels for the fused ConvRot INT8 sparse-Piper Q/K/V projections.

Each program projects 128 rows onto one D128 head through the SM8x ConvRot INT8
GEMM's pipeline: ``cp.async`` copies stage K64 slices of the INT8 operands in shared
memory, and ``mma_v2`` accumulates exact INT32 products.

Q and K split warps only across rows and permute the weight rows, so each thread
holds 4 consecutive features of every row it owns, with the other feature bits in
registers or its lane quad. Loads of cos/sin are then 16 bytes wide, and RoPE
pairs stay in registers. Five of the seven Hadamard stages stay in registers;
the other two use butterfly shuffles.

V keeps a 2x2 warp grid. Its epilogue spreads rows across lanes, so the
transposed stores are coalesced.

Shapes outside ``supports_projection`` use the shared Triton launchers.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra import libdevice

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import (
    SIGNED_HADAMARD_MASK,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MEAN_ROUTING
from piper_kernels.linear.convrot.int8._nvidia.gluon_async_copy import (
    _accumulate as _accumulate_tiles,
)

from .. import _kernels
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from .._layout import QUERY_SCALE_ROWS, TILE_ROWS

_HEAD_DIM = 128
_BLOCK_M = 128
_BLOCK_K = 64
_NUM_STAGES = 3
_NUM_WARPS = 4
# The projected V mean is one row per head; its launcher reuses the Triton kernel.
_MEAN_BLOCK_K = 128

_GL_HEAD_DIM = gl.constexpr(_HEAD_DIM)
_GL_BLOCK_M = gl.constexpr(_BLOCK_M)
_GL_BLOCK_K = gl.constexpr(_BLOCK_K)
_GL_NUM_STAGES = gl.constexpr(_NUM_STAGES)
_GL_NUM_WARPS = gl.constexpr(_NUM_WARPS)
_GL_TILE_ROWS = gl.constexpr(TILE_ROWS)
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)
_GL_SCALE_EPSILON = gl.constexpr(1e-7)
_GL_INT8_RANGE = gl.constexpr(127.0)
_GL_P_UINT8_RANGE = gl.constexpr(255.0)
_GL_LOG2_E = gl.constexpr(1.4426950408889634)
_GL_HADAMARD_NORM = gl.constexpr(0.08838834764831845)
_GL_HADAMARD_WORD_0 = gl.constexpr(SIGNED_HADAMARD_MASK[0])
_GL_HADAMARD_WORD_1 = gl.constexpr(SIGNED_HADAMARD_MASK[1])
_GL_HADAMARD_WORD_2 = gl.constexpr(SIGNED_HADAMARD_MASK[2])
_GL_HADAMARD_WORD_3 = gl.constexpr(SIGNED_HADAMARD_MASK[3])
_GL_SHUFFLE_LANE_1 = gl.constexpr("shfl.sync.bfly.b32 $0, $1, 0x1, 0x1f, 0xffffffff;")
_GL_SHUFFLE_LANE_2 = gl.constexpr("shfl.sync.bfly.b32 $0, $1, 0x2, 0x1f, 0xffffffff;")
# Every thread copies 16 bytes of a K64 slice; each warp copies 8 rows per step.
_GL_COPY_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 16], [8, 4], [_NUM_WARPS, 1], [1, 0]))
_GL_COPY_ROWS = gl.constexpr(gl.SliceLayout(1, _GL_COPY_LAYOUT.value))
# Q/K rows leave in 16-byte stores; each warp writes 4 whole D128 rows.
_GL_STORE_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 16], [4, 8], [_NUM_WARPS, 1], [1, 0]))
_GL_SUMMARY_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 4], [1, 32], [1, _NUM_WARPS], [1, 0]))

# Sequence lengths and offsets vary per call; specializing them would compile one
# kernel per divisibility class. Storage extents arrive as K64 tile counts, so the
# kernels still know that row strides are multiples of 64.
_DO_NOT_SPECIALIZE = ("logical_sequence_length", "row_block_offset", "storage_tiles")
_DO_NOT_SPECIALIZE_QUERY = (*_DO_NOT_SPECIALIZE, "chunk_start", "query_sequence_end")


def supports_projection(input_qdata: torch.Tensor, head_dim: int, rotary_dim: int = 0) -> bool:
    """Return whether these kernels cover the operands; Q/K pass their RoPE width."""
    return (
        head_dim == _HEAD_DIM
        and input_qdata.shape[2] % _BLOCK_K == 0
        # RoPE pairs must differ only in feature bits 4-6, which each thread holds.
        and rotary_dim % 32 == 0
    )


@gluon.jit
def _project_int8(
    input_ptr,
    weight_ptr,
    input_rows,
    weight_rows,
    accumulator,
    input_features: gl.constexpr,
):
    """Accumulate ``input[rows] @ weight[weight_rows].T`` in the SM8x GEMM's copy pipeline."""
    mma_layout: gl.constexpr = accumulator.type.layout
    features = gl.arange(0, _GL_BLOCK_K, gl.SliceLayout(0, _GL_COPY_LAYOUT))
    input_pointers = (
        input_ptr + input_rows.to(gl.int64)[:, None] * input_features + features[None, :]
    )
    weight_pointers = (
        weight_ptr + weight_rows.to(gl.int64)[:, None] * input_features + features[None, :]
    )
    # Rows are in bounds and K64 slices are whole, so copies need no row or column masks.
    return _accumulate_tiles(
        input_pointers,
        weight_pointers,
        None,
        None,
        features,
        input_features,
        input_features // _GL_BLOCK_K,
        accumulator,
        _GL_BLOCK_K,
        _GL_NUM_STAGES,
        gl.DotOperandLayout(0, mma_layout, k_width=4),
        gl.DotOperandLayout(1, mma_layout, k_width=4),
        False,
    )


@gluon.jit
def _round_to_int8(values):
    """Round half away from zero and clamp to the symmetric INT8 range."""
    rounded = values + 0.5 * gl.where(values >= 0, 1.0, -1.0)
    return gl.maximum(-_GL_INT8_RANGE, gl.minimum(_GL_INT8_RANGE, rounded)).to(gl.int8)


@gluon.jit
def _minmax(maximum_0, minimum_0, maximum_1, minimum_1):
    """Combine (maximum, minimum) pairs in one reduction."""
    return gl.maximum(maximum_0, maximum_1), gl.minimum(minimum_0, minimum_1)


# Q/K feature placement. Accumulator column c holds feature c0->f0, c3->f1, c1->f2, c2->f3,
# c4-c6->f4-f6: thread registers then own f0, f1, and f4-f6 of each row, and the lane
# quad owns f2 and f3.


def _feature_layout() -> gl.DistributedLinearLayout:
    """The Q/K accumulator of a [4, 1] mma_v2 warp grid, indexed by feature."""
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 16], [0, 32], [0, 64], [16, 0]],
        lane_bases=[[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[],
        shape=[_BLOCK_M, _HEAD_DIM],
    )


_GL_FEATURE_LAYOUT = gl.constexpr(_feature_layout())


@gluon.jit
def _column_features(columns):
    """Return the head feature that each accumulator column projects."""
    return (
        (columns & 1) | (((columns >> 3) & 1) << 1) | (((columns >> 1) & 3) << 2) | (columns & 0x70)
    )


@gluon.jit
def _by_feature(accumulator):
    """Reindex the accumulator columns by feature without moving data between threads."""
    rows: gl.constexpr = accumulator.shape[0]
    bits = gl.reshape(accumulator, [rows, 2, 2, 2, 2, 2, 2, 2])
    ordered = gl.permute(bits, [0, 1, 2, 3, 5, 6, 4, 7])
    return gl.convert_layout(gl.reshape(ordered, [rows, _GL_HEAD_DIM]), _GL_FEATURE_LAYOUT)


@gluon.jit
def _rotated_group(groups, group: gl.constexpr, half_groups: gl.constexpr):
    """Return one 16-feature group's split-half RoPE partner."""
    if group < half_groups:
        return -groups[group + half_groups]
    if group < 2 * half_groups:
        return groups[group - half_groups]
    return groups[group]


@gluon.jit
def _rotate_half(values, rotary_dim: gl.constexpr):
    """Return split-half RoPE partners, negated below half the rotary width.

    Features 16g+r pair with 16(g +- rotary_dim/32)+r, a move in the register
    bits f4-f6 only.
    """
    rows: gl.constexpr = values.shape[0]
    half_groups: gl.constexpr = rotary_dim // 32
    by_group = gl.permute(gl.reshape(values, [rows, 2, 2, 2, 16]), [0, 4, 3, 2, 1])
    groups_0123, groups_4567 = gl.split(by_group)
    groups_01, groups_23 = gl.split(groups_0123)
    groups_45, groups_67 = gl.split(groups_4567)
    group_0, group_1 = gl.split(groups_01)
    group_2, group_3 = gl.split(groups_23)
    group_4, group_5 = gl.split(groups_45)
    group_6, group_7 = gl.split(groups_67)
    groups = (group_0, group_1, group_2, group_3, group_4, group_5, group_6, group_7)
    rotated = gl.join(
        gl.join(
            gl.join(_rotated_group(groups, 0, half_groups), _rotated_group(groups, 1, half_groups)),
            gl.join(_rotated_group(groups, 2, half_groups), _rotated_group(groups, 3, half_groups)),
        ),
        gl.join(
            gl.join(_rotated_group(groups, 4, half_groups), _rotated_group(groups, 5, half_groups)),
            gl.join(_rotated_group(groups, 6, half_groups), _rotated_group(groups, 7, half_groups)),
        ),
    )
    rotated = gl.reshape(gl.permute(rotated, [0, 4, 3, 2, 1]), [rows, _GL_HEAD_DIM])
    return gl.convert_layout(rotated, values.type.layout)


@gluon.jit
def _register_butterfly(values, distance: gl.constexpr):
    """One Hadamard stage over a feature bit that each thread holds in registers."""
    rows: gl.constexpr = values.shape[0]
    outer: gl.constexpr = _GL_HEAD_DIM // (2 * distance)
    pairs = gl.permute(gl.reshape(values, [rows, outer, 2, distance]), [0, 1, 3, 2])
    low, high = gl.split(pairs)
    transformed = gl.permute(gl.join(low + high, low - high), [0, 1, 3, 2])
    return gl.convert_layout(gl.reshape(transformed, [rows, _GL_HEAD_DIM]), values.type.layout)


@gluon.jit
def _lane_butterfly(values, features, distance: gl.constexpr, shuffle: gl.constexpr):
    """One Hadamard stage over a feature bit held by the lane quad."""
    partner = gl.inline_asm_elementwise(
        shuffle, "=r,r", [values], dtype=gl.float32, is_pure=True, pack=1
    )
    return gl.where(((features & distance) == 0)[None, :], values + partner, partner - values)


@gluon.jit
def _signed_hadamard(values, features):
    """Apply the signed, normalized D128 Hadamard in the Triton stage order."""
    word_group = features // 32
    words = gl.where(
        word_group == 0,
        _GL_HADAMARD_WORD_0,
        gl.where(
            word_group == 1,
            _GL_HADAMARD_WORD_1,
            gl.where(word_group == 2, _GL_HADAMARD_WORD_2, _GL_HADAMARD_WORD_3),
        ),
    ).to(gl.uint32)
    signs = gl.where(((words >> (features % 32).to(gl.uint32)) & 1) != 0, 1.0, -1.0)
    values *= signs[None, :]
    values = _register_butterfly(values, 1)
    values = _register_butterfly(values, 2)
    values = _lane_butterfly(values, features, 4, _GL_SHUFFLE_LANE_1)
    values = _lane_butterfly(values, features, 8, _GL_SHUFFLE_LANE_2)
    values = _register_butterfly(values, 16)
    values = _register_butterfly(values, 32)
    values = _register_butterfly(values, 64)
    return values * _GL_HADAMARD_NORM


@gluon.jit
def _project_rmsnorm_rope(
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
    input_features: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project 128 rows onto a D128 head, then apply FP32 RMSNorm and RoPE by feature.

    Returns the tile, its sequence positions, and its features. With ``mask_rows``,
    rows past the sequence end repeat its last row; callers mask their outputs.
    """
    positions = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, _GL_FEATURE_LAYOUT))
    copy_positions = first_row + gl.arange(0, _GL_BLOCK_M, _GL_COPY_ROWS)
    if mask_rows:
        positions = gl.minimum(positions, logical_sequence_length - 1)
        copy_positions = gl.minimum(copy_positions, logical_sequence_length - 1)
    input_row_start = batch * logical_sequence_length
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[_GL_NUM_WARPS, 1], instr_shape=[16, 8]
    )
    columns = gl.arange(0, _GL_HEAD_DIM, _GL_COPY_ROWS)
    accumulator = _project_int8(
        input_ptr,
        weight_ptr,
        input_row_start + copy_positions,
        head * _GL_HEAD_DIM + _column_features(columns),
        gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.int32, mma_layout),
        input_features,
    )
    accumulator = _by_feature(accumulator)

    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, _GL_FEATURE_LAYOUT))
    input_scale = gl.load(input_scale_ptr + input_row_start + positions)
    weight_scale = gl.load(weight_scale_ptr + head * _GL_HEAD_DIM + features)
    projection = accumulator.to(gl.float32) * input_scale[:, None] * weight_scale[None, :]
    if bias_ptr is not None:
        projection += gl.load(bias_ptr + head * _GL_HEAD_DIM + features).to(gl.float32)[None, :]

    variance = gl.sum(projection * projection, axis=1) / _GL_HEAD_DIM + norm_epsilon
    inverse_rms = libdevice.rsqrt_rn(variance)
    normalized = projection * inverse_rms[:, None]  # pyright: ignore[reportOptionalSubscript]
    if norm_weight_ptr is not None:
        normalized = normalized * gl.load(norm_weight_ptr + features).to(gl.float32)[None, :]
    rotary_features = features < rotary_dim
    rope_offsets = positions[:, None] * rotary_dim + features[None, :]
    cos = gl.load(cos_ptr + rope_offsets, mask=rotary_features[None, :], other=1.0)
    sin = gl.load(sin_ptr + rope_offsets, mask=rotary_features[None, :], other=0.0)
    rotary = normalized * cos + _rotate_half(normalized, rotary_dim) * sin
    return gl.where(rotary_features[None, :], rotary, normalized), positions, features


@gluon.jit
def _valid_rows(rows, positions, row_end, block_lengths_ptr, mask_block_lengths: gl.constexpr):
    """Mark rows before ``row_end`` or, with block lengths, in each K64 block's valid prefix.

    Block lengths are read at the clamped positions, which stay inside the sequence.
    """
    if mask_block_lengths:
        return rows % _GL_TILE_ROWS < gl.load(block_lengths_ptr + positions // _GL_TILE_ROWS)
    return rows < row_end


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
):
    """Zero invalid rows and reduce each 64-row block to its routing summaries.

    Returns the masked tile with (primary, auxiliary) summaries: the block mean twice, or
    its maximum and minimum.
    """
    blocks: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    rows = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, _GL_FEATURE_LAYOUT))
    valid = _valid_rows(rows, positions, row_end, block_lengths_ptr, mask_block_lengths)
    valid = valid[:, None] & (features >= 0)[None, :]
    if mask_rows:
        values = gl.where(valid, values, 0.0)
    block_values = gl.reshape(values, [blocks, _GL_TILE_ROWS, _GL_HEAD_DIM])
    block_valid = gl.reshape(valid, [blocks, _GL_TILE_ROWS, _GL_HEAD_DIM])
    if mean_pool_summary:
        if mask_rows:
            valid_count = gl.sum(block_valid.to(gl.int32), axis=1)
            block_sum = gl.sum(gl.where(block_valid, block_values, 0.0), axis=1)
            mean = block_sum / valid_count  # pyright: ignore[reportOperatorIssue]
        else:
            mean = gl.sum(block_values, axis=1) / _GL_TILE_ROWS
        return values, mean, mean
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
    return values, maximum, minimum


@gluon.jit
def _store_rows(pointer, values, first_row, storage_rows, mask_rows: gl.constexpr):
    """Store a 128-row INT8 Q/K tile of one [S, D128] head in 16-byte chunks."""
    values = gl.convert_layout(gl.reshape(values, [_GL_BLOCK_M, _GL_HEAD_DIM]), _GL_STORE_LAYOUT)
    rows = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, _GL_STORE_LAYOUT))
    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, _GL_STORE_LAYOUT))
    pointers = pointer + rows[:, None] * _GL_HEAD_DIM + features[None, :]
    if mask_rows:
        gl.store(pointers, values, mask=rows[:, None] < storage_rows)
    else:
        gl.store(pointers, values)


@gluon.jit
def _store_scales(pointer, scales, first_group, storage_groups, mask_rows: gl.constexpr):
    """Store one tile's per-group scales into one head's scale row."""
    layout: gl.constexpr = gl.SliceLayout(1, _GL_SUMMARY_LAYOUT)
    groups = first_group + gl.arange(0, scales.shape[0], layout)
    scales = gl.convert_layout(scales, layout)
    if mask_rows:
        gl.store(pointer + groups, scales, mask=groups < storage_groups)
    else:
        gl.store(pointer + groups, scales)


@gluon.jit
def _store_summaries(pointer, summaries, first_block, storage_blocks, mask_rows: gl.constexpr):
    """Store one tile's [blocks, D128] summaries or means into one head's block rows."""
    blocks = first_block + gl.arange(0, summaries.shape[0], gl.SliceLayout(1, _GL_SUMMARY_LAYOUT))
    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, _GL_SUMMARY_LAYOUT))
    pointers = pointer + blocks[:, None] * _GL_HEAD_DIM + features[None, :]
    summaries = gl.convert_layout(summaries, _GL_SUMMARY_LAYOUT)
    if mask_rows:
        gl.store(pointers, summaries, mask=(blocks < storage_blocks)[:, None])
    else:
        gl.store(pointers, summaries)


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
    values, positions, features = _project_rmsnorm_rope(
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

    values, maximum, minimum = _summarize_blocks(
        values,
        positions,
        features,
        first_row,
        query_sequence_end,
        block_lengths_ptr,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
    )
    summary = maximum
    if not mean_pool_summary:
        summary += minimum

    blocks: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    groups: gl.constexpr = _GL_BLOCK_M // _GL_QUERY_SCALE_ROWS
    smoothed = gl.reshape(
        _signed_hadamard(values, features), [groups, _GL_QUERY_SCALE_ROWS, _GL_HEAD_DIM]
    )
    raw_scale = (
        gl.max(gl.max(gl.abs(smoothed), axis=2), axis=1) / _GL_INT8_RANGE + _GL_SCALE_EPSILON
    )
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
    quantized = _round_to_int8(smoothed / raw_scale[:, None, None])

    batch_head = batch * heads + head
    storage_rows = storage_tiles * _GL_TILE_ROWS
    storage_groups = storage_rows // _GL_QUERY_SCALE_ROWS
    _store_rows(
        query_ptr + batch_head.to(gl.int64) * storage_rows * _GL_HEAD_DIM,
        quantized,
        row_block * _GL_BLOCK_M,
        storage_rows,
        mask_rows,
    )
    _store_scales(
        query_scale_ptr + batch_head * storage_groups,
        stored_scale,
        row_block * groups,
        storage_groups,
        mask_rows,
    )
    _store_summaries(
        query_summary_ptr + batch_head * storage_tiles * _GL_HEAD_DIM,
        summary,
        row_block * blocks,
        storage_tiles,
        mask_rows,
    )


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
    """Project a K tile once and emit K64 INT8 rows, scales, and routing summaries."""
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = row_block * _GL_BLOCK_M
    values, positions, features = _project_rmsnorm_rope(
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

    values, key_summary, key_aux = _summarize_blocks(
        values,
        positions,
        features,
        first_row,
        logical_sequence_length,
        block_lengths_ptr,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
    )

    tiles: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    smoothed = gl.reshape(_signed_hadamard(values, features), [tiles, _GL_TILE_ROWS, _GL_HEAD_DIM])
    key_scale = (
        gl.max(gl.max(gl.abs(smoothed), axis=2), axis=1) / _GL_INT8_RANGE + _GL_SCALE_EPSILON
    )
    quantized = _round_to_int8(smoothed / key_scale[:, None, None])

    batch_head = batch * heads + head
    storage_rows = storage_tiles * _GL_TILE_ROWS
    first_tile = row_block * tiles
    tile_row = batch_head * storage_tiles
    _store_rows(
        key_ptr + batch_head.to(gl.int64) * storage_rows * _GL_HEAD_DIM,
        quantized,
        first_row,
        storage_rows,
        mask_rows,
    )
    _store_scales(key_scale_ptr + tile_row, key_scale, first_tile, storage_tiles, mask_rows)
    _store_summaries(
        key_summary_ptr + tile_row * _GL_HEAD_DIM, key_summary, first_tile, storage_tiles, mask_rows
    )
    if not mean_pool_summary:
        _store_summaries(
            key_aux_ptr + tile_row * _GL_HEAD_DIM, key_aux, first_tile, storage_tiles, mask_rows
        )


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE)
def _value_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    value_mean_ptr,
    value_ptr,
    value_scale_ptr,
    block_mean_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    emit_block_mean: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a V tile and emit centered, K64-scaled, transposed INT8 values."""
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = row_block * _GL_BLOCK_M
    input_row_start = batch * logical_sequence_length
    copy_positions = first_row + gl.arange(0, _GL_BLOCK_M, _GL_COPY_ROWS)
    if mask_rows:
        copy_positions = gl.minimum(copy_positions, logical_sequence_length - 1)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[2, 2], instr_shape=[16, 8]
    )
    accumulator = _project_int8(
        input_ptr,
        weight_ptr,
        input_row_start + copy_positions,
        head * _GL_HEAD_DIM + gl.arange(0, _GL_HEAD_DIM, _GL_COPY_ROWS),
        gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.int32, mma_layout),
        input_features,
    )

    # [K64 tiles, rows, features] with lanes along rows for the transposed V stores.
    tiles: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 32, 1], [tiles, 1, _GL_NUM_WARPS // tiles], [1, 2, 0]
    )
    tile_rows_layout: gl.constexpr = gl.SliceLayout(2, layout)
    first_tile = row_block * tiles
    tile_offsets = first_tile + gl.arange(0, tiles, gl.SliceLayout(1, tile_rows_layout))
    rows_in_tile = gl.arange(0, _GL_TILE_ROWS, gl.SliceLayout(0, tile_rows_layout))
    sequence = tile_offsets[:, None] * _GL_TILE_ROWS + rows_in_tile[None, :]
    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    positions = sequence
    if mask_rows:
        positions = gl.minimum(positions, logical_sequence_length - 1)
    input_scale = gl.load(input_scale_ptr + input_row_start + positions)
    weight_scale = gl.load(weight_scale_ptr + head * _GL_HEAD_DIM + features)
    accumulator = gl.convert_layout(
        gl.reshape(accumulator, [tiles, _GL_TILE_ROWS, _GL_HEAD_DIM]), layout
    )
    projection = accumulator.to(gl.float32) * input_scale[:, :, None] * weight_scale[None, None, :]
    if bias_ptr is not None:
        bias = gl.load(bias_ptr + head * _GL_HEAD_DIM + features).to(gl.float32)
        projection += bias[None, None, :]

    if mask_block_lengths:
        tile_length = gl.load(
            block_lengths_ptr + tile_offsets, mask=tile_offsets < storage_tiles, other=0
        )
        valid = rows_in_tile[None, :] < tile_length[:, None]
    else:
        valid = sequence < logical_sequence_length
    batch_head = batch * heads + head
    tile_row = batch_head * storage_tiles
    if emit_block_mean:
        if mask_rows:
            valid_count = gl.maximum(gl.sum(valid.to(gl.int32), axis=1), 1)
            count_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(1, layout))
            block_sum = gl.sum(gl.where(valid[:, :, None], projection, 0.0), axis=1)
            block_mean = block_sum / gl.convert_layout(valid_count, count_layout)[:, None]
        else:
            block_mean = gl.sum(projection, axis=1) / _GL_TILE_ROWS
        _store_summaries(
            block_mean_ptr + tile_row * _GL_HEAD_DIM,
            block_mean,
            first_tile,
            storage_tiles,
            mask_rows,
        )

    value_mean = gl.load(value_mean_ptr + batch_head * _GL_HEAD_DIM + features)
    centered = projection - value_mean[None, None, :]
    if mask_rows:
        centered = gl.where(valid[:, :, None], centered, 0.0)
    value_scale = (
        gl.max(gl.max(gl.abs(centered), axis=2), axis=1) / _GL_INT8_RANGE + _GL_SCALE_EPSILON
    )
    quantized = _round_to_int8(centered / value_scale[:, None, None])

    storage_rows = storage_tiles * _GL_TILE_ROWS
    value_pointers = (
        value_ptr
        + batch_head.to(gl.int64) * _GL_HEAD_DIM * storage_rows
        + features[None, None, :].to(gl.int64) * storage_rows
        + sequence[:, :, None]
    )
    if mask_rows:
        gl.store(value_pointers, quantized, mask=(tile_offsets < storage_tiles)[:, None, None])
    else:
        gl.store(value_pointers, quantized)
    _store_scales(
        value_scale_ptr + tile_row,
        value_scale * _GL_P_UINT8_RANGE,
        first_tile,
        storage_tiles,
        mask_rows,
    )


def _launch_rows(sequence_rows: int, launch, *, mask_block_lengths: bool) -> None:
    """Launch full 128-row tiles, then one masked tile for the remainder.

    Block lengths can end any K64 block early, so they mask every tile. Launch grids put
    heads on the fastest axis, so the programs that run together share their input rows
    in L2; grouping several row blocks per head, as the Triton kernels do, measured 1-3%
    slower.
    """
    full_row_blocks = sequence_rows // _BLOCK_M
    if full_row_blocks:
        launch(full_row_blocks, 0, mask_rows=mask_block_lengths)
    if sequence_rows % _BLOCK_M:
        launch(1, full_row_blocks, mask_rows=True)


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
) -> None:
    """Fill Q32 INT8 queries, scales, and Q64 summaries for a supported query window."""
    query, query_scale, query_summary = out
    batch, heads, storage_sequence_length, _head_dim = query.shape
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _query_kernel[(heads, row_blocks, batch)](
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
                input_features=input_qdata.shape[2],
                heads=heads,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                num_warps=_NUM_WARPS,
            )

        _launch_rows(chunk_rows, launch, mask_block_lengths=mask_block_lengths)


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
) -> None:
    """Fill K64 INT8 keys, scales, and routing summaries for supported operands."""
    key, key_scale, key_summary, key_aux = out
    batch, heads, storage_sequence_length, _head_dim = key.shape
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _key_kernel[(heads, row_blocks, batch)](
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
                key_summary,
                key_aux,
                block_lengths if mask_block_lengths else key_scale,
                input_qdata.shape[1],
                row_block_offset,
                storage_sequence_length // TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                num_warps=_NUM_WARPS,
            )

        _launch_rows(input_qdata.shape[1], launch, mask_block_lengths=mask_block_lengths)


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    emit_block_mean: bool,
    out: ValueOutput,
    bias: torch.Tensor | None = None,
) -> None:
    """Fill centered K64-scaled INT8 values, the projected mean, and block means."""
    value, value_scale_multiplier, value_mean, block_mean = out
    batch, heads, _head_dim, storage_sequence_length = value.shape
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _value_kernel[(heads, row_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                value_mean,
                value,
                value_scale_multiplier,
                block_mean,
                block_lengths if mask_block_lengths else value_mean,
                input_qdata.shape[1],
                row_block_offset,
                storage_sequence_length // TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                mask_block_lengths=mask_block_lengths,
                emit_block_mean=emit_block_mean,
                mask_rows=mask_rows,
                num_warps=_NUM_WARPS,
            )

        _kernels._project_prepared_input_mean_kernel[(heads, batch)](
            input_mean,
            weight_qdata,
            weight_scale,
            value_mean,
            bias_ptr=bias,
            input_features=input_qdata.shape[2],
            output_features=heads * _HEAD_DIM,
            block_n=_HEAD_DIM,
            block_k=_MEAN_BLOCK_K,
            num_warps=_NUM_WARPS,
        )
        _launch_rows(input_qdata.shape[1], launch, mask_block_lengths=mask_block_lengths)
