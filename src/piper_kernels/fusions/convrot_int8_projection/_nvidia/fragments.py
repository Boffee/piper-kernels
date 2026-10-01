"""Gluon fragments shared by the SM89 fused ConvRot INT8 dense and sparse Piper projections.

Each program projects 128 rows onto one D128 head through the SM8x ConvRot INT8
GEMM's pipeline: ``cp.async`` copies stage K64 slices of the INT8 operands in shared
memory, and ``mma_v2`` accumulates exact INT32 products.

Q and K split warps only across rows and permute the weight rows, so each thread
holds 4 consecutive features of every row it owns, with the other feature bits in
registers or its lane quad. Loads of cos/sin are then 16 bytes wide, and RoPE
pairs stay in registers. The signed Hadamard runs five of its seven stages in
registers and the other two as butterfly shuffles. K stores BF16 rows and FP32
tile sums for the shared encoder, which centers K by its global mean.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra import libdevice

from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import (
    SIGNED_HADAMARD_MASK,
)
from piper_kernels.linear.convrot.int8._nvidia.gluon_async_copy import (
    _accumulate as _accumulate_tiles,
)

from ._plan import GLUON_BLOCK_K, GLUON_BLOCK_M, GLUON_NUM_STAGES, GLUON_NUM_WARPS

HEAD_DIM = 128
BLOCK_M = GLUON_BLOCK_M
BLOCK_K = GLUON_BLOCK_K
NUM_STAGES = GLUON_NUM_STAGES
NUM_WARPS = GLUON_NUM_WARPS

_GL_HEAD_DIM = gl.constexpr(HEAD_DIM)
_GL_BLOCK_M = gl.constexpr(BLOCK_M)
_GL_BLOCK_K = gl.constexpr(BLOCK_K)
_GL_NUM_STAGES = gl.constexpr(NUM_STAGES)
_GL_NUM_WARPS = gl.constexpr(NUM_WARPS)
_GL_INT8_RANGE = gl.constexpr(127.0)
_GL_HADAMARD_NORM = gl.constexpr(0.08838834764831845)
_GL_HADAMARD_WORD_0 = gl.constexpr(SIGNED_HADAMARD_MASK[0])
_GL_HADAMARD_WORD_1 = gl.constexpr(SIGNED_HADAMARD_MASK[1])
_GL_HADAMARD_WORD_2 = gl.constexpr(SIGNED_HADAMARD_MASK[2])
_GL_HADAMARD_WORD_3 = gl.constexpr(SIGNED_HADAMARD_MASK[3])
_GL_SHUFFLE_LANE_1 = gl.constexpr("shfl.sync.bfly.b32 $0, $1, 0x1, 0x1f, 0xffffffff;")
_GL_SHUFFLE_LANE_2 = gl.constexpr("shfl.sync.bfly.b32 $0, $1, 0x2, 0x1f, 0xffffffff;")
# Every thread copies 16 bytes of a K64 slice; each warp copies 8 rows per step.
_GL_COPY_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 16], [8, 4], [NUM_WARPS, 1], [1, 0]))
COPY_ROWS = gl.constexpr(gl.SliceLayout(1, _GL_COPY_LAYOUT.value))
# Each thread stores 16 consecutive features of a Q/K row; each warp writes 4 whole rows.
_GL_STORE_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 16], [4, 8], [NUM_WARPS, 1], [1, 0]))
_GL_SUMMARY_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 4], [1, 32], [1, NUM_WARPS], [1, 0]))


@gluon.jit
def project_int8(
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
def round_to_int8(values):
    """Round half away from zero and clamp to the symmetric INT8 range."""
    rounded = values + 0.5 * gl.where(values >= 0, 1.0, -1.0)
    return gl.maximum(-_GL_INT8_RANGE, gl.minimum(_GL_INT8_RANGE, rounded)).to(gl.int8)


# Q/K feature placement. Accumulator column c holds feature c0->f0, c3->f1, c1->f2, c2->f3,
# c4-c6->f4-f6: thread registers then own f0, f1, and f4-f6 of each row, and the lane
# quad owns f2 and f3. Rows 8i + t (i = 0..3) of each 32-row group share a thread.


def _feature_layout() -> gl.DistributedLinearLayout:
    """The Q/K accumulator of a [4, 1] mma_v2 warp grid, indexed by feature."""
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 16], [0, 32], [0, 64], [16, 0]],
        lane_bases=[[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[],
        shape=[BLOCK_M, HEAD_DIM],
    )


FEATURE_LAYOUT = gl.constexpr(_feature_layout())


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
    return gl.convert_layout(gl.reshape(ordered, [rows, _GL_HEAD_DIM]), FEATURE_LAYOUT)


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
def signed_hadamard(values, features):
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
def project_rmsnorm_rope(
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
    positions = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, FEATURE_LAYOUT))
    copy_positions = first_row + gl.arange(0, _GL_BLOCK_M, COPY_ROWS)
    if mask_rows:
        positions = gl.minimum(positions, logical_sequence_length - 1)
        copy_positions = gl.minimum(copy_positions, logical_sequence_length - 1)
    input_row_start = batch * logical_sequence_length
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[_GL_NUM_WARPS, 1], instr_shape=[16, 8]
    )
    columns = gl.arange(0, _GL_HEAD_DIM, COPY_ROWS)
    accumulator = project_int8(
        input_ptr,
        weight_ptr,
        input_row_start + copy_positions,
        head * _GL_HEAD_DIM + _column_features(columns),
        gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.int32, mma_layout),
        input_features,
    )
    accumulator = _by_feature(accumulator)

    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, FEATURE_LAYOUT))
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
def store_rows(pointer, values, first_row, storage_rows, mask_rows: gl.constexpr):
    """Store a 128-row Q/K tile of one [S, D128] head in contiguous feature chunks."""
    values = gl.convert_layout(gl.reshape(values, [_GL_BLOCK_M, _GL_HEAD_DIM]), _GL_STORE_LAYOUT)
    rows = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, _GL_STORE_LAYOUT))
    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, _GL_STORE_LAYOUT))
    pointers = pointer + rows[:, None] * _GL_HEAD_DIM + features[None, :]
    if mask_rows:
        gl.store(pointers, values, mask=rows[:, None] < storage_rows)
    else:
        gl.store(pointers, values)


@gluon.jit
def store_scales(pointer, scales, first_group, storage_groups, mask_rows: gl.constexpr):
    """Store one tile's per-group scales into one head's scale row."""
    layout: gl.constexpr = gl.SliceLayout(1, _GL_SUMMARY_LAYOUT)
    groups = first_group + gl.arange(0, scales.shape[0], layout)
    scales = gl.convert_layout(scales, layout)
    if mask_rows:
        gl.store(pointer + groups, scales, mask=groups < storage_groups)
    else:
        gl.store(pointer + groups, scales)


@gluon.jit
def store_summaries(pointer, summaries, first_block, storage_blocks, mask_rows: gl.constexpr):
    """Store one tile's [blocks, D128] summaries or means into one head's block rows."""
    blocks = first_block + gl.arange(0, summaries.shape[0], gl.SliceLayout(1, _GL_SUMMARY_LAYOUT))
    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, _GL_SUMMARY_LAYOUT))
    pointers = pointer + blocks[:, None] * _GL_HEAD_DIM + features[None, :]
    summaries = gl.convert_layout(summaries, _GL_SUMMARY_LAYOUT)
    if mask_rows:
        gl.store(pointers, summaries, mask=(blocks < storage_blocks)[:, None])
    else:
        gl.store(pointers, summaries)


@gluon.jit
def store_centered_key(
    stored_ptr,
    partial_ptr,
    values,
    batch,
    heads: gl.constexpr,
    head,
    row_block,
    first_row,
    storage_tiles,
    tile_rows: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Store K tile ``row_block``, whose first row is ``first_row``, for the centered encoder.

    Rows are stored as BF16, with FP32 sums of each stored tile's values; the shared
    encoder reduces the sums to the global mean that centers K.
    """
    tiles: gl.constexpr = _GL_BLOCK_M // tile_rows
    stored = values.to(gl.bfloat16)
    stored_tiles = gl.reshape(stored.to(gl.float32), [tiles, tile_rows, _GL_HEAD_DIM])
    tile_sums = gl.sum(stored_tiles, axis=1)
    batch_head = batch * heads + head
    storage_rows = storage_tiles * tile_rows
    store_rows(
        stored_ptr + batch_head.to(gl.int64) * storage_rows * _GL_HEAD_DIM,
        stored,
        first_row,
        storage_rows,
        mask_rows,
    )
    store_summaries(
        partial_ptr + batch_head * storage_tiles * _GL_HEAD_DIM,
        tile_sums,
        row_block * tiles,
        storage_tiles,
        mask_rows,
    )
