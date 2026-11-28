"""Unified NVIDIA paired-K128 Gluon kernel for sparse Piper Attention (TMA & ``cp.async``).

SM120 (``use_tma=True``) loads operands through TMA descriptors and runs a
split-D64 recurrence for D128: Q/K dots stream in four D32 slices, 16-tile
K-bias batches reuse ``key_center_shared`` as an FP32 cache, PV accumulation
splits across D64 halves with ``rescale_packed``, and V-mean restoration buffers
16 pairs (32 tiles) via BF16 tensor-core strips reusing query shared memory.

SM120 D64 (``use_tma=True``) and SM8x D64/D128 (``use_tma=False`` via
``cp.async`` commit groups) share one single-accumulator recurrence driver and
one ``is_final: gl.constexpr`` pair-step helper.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0912, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.piper._nvidia.fragments import (
    packed_float32_to_uint8,
    rescale_packed,
    uint8_int8_mma,
)
from piper_kernels.attention.kernels.piper.gluon import (
    load_tile_column,
    split_bf16_pair,
    store_tile_column,
)
from piper_kernels.attention.kernels.sparse_piper.gluon import pair_tile_offsets
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS, TILE_ROWS

from .._prepared import (
    _DO_NOT_SPECIALIZE_ARGUMENTS,
    _ContextLauncher,
    _PreparedSparsePiperContext,
    _PreparedSparsePiperQuery,
    _ValidatedAttentionLaunch,
)
from . import policy
from ._recurrence import (
    _K_BIAS_GROUP_PAIRS,
    K_BIAS_GROUP_TILES,
    accumulate_value_mean_pair,
    build_k_bias_cache,
    compute_pair_k_bias,
    compute_pair_softmax_bounds,
    piper_probability_pair,
    piper_pv_pair,
    resolve_batch_tiles,
    store_attention_output,
    store_attention_output_strip,
)

_BLOCK_N = TILE_ROWS
_ASYNC_COPY_WARPS = 4
# ``cp.async`` moves at most 16 bytes per thread and instruction.
_COPY_ALIGNMENT = 16

_GL_BLOCK_N = gl.constexpr(_BLOCK_N)
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)
_GL_COPY_BYTES = gl.constexpr(_COPY_ALIGNMENT)

# SM120 D128 split-accumulator chunk constants and layouts.
_VALUE_MEAN_GROUP_PAIRS = gl.constexpr(16)
_VALUE_MEAN_GROUP_TILES = gl.constexpr(32)
_SHARED_TILE_LAYOUT = gl.constexpr(gl.BlockedLayout([1], [32], [4], [0]))
_VALUE_MEAN_LOAD_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0]))
_SCRATCH_SHARED_LAYOUT = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))


# ---------------------------------------------------------------------------
# TMA and cp.async load primitives
# ---------------------------------------------------------------------------
@gluon.jit
def _issue_tma(descriptor, offsets, shared, barrier):
    mbarrier.expect(barrier, descriptor.block_type.nbytes)
    tma.async_load(descriptor, offsets, barrier, shared)


@gluon.jit
def _issue_tma_pair(descriptor, offsets_0, offsets_1, shared_0, shared_1, barrier):
    mbarrier.expect(barrier, descriptor.block_type.nbytes * 2)
    tma.async_load(descriptor, offsets_0, barrier, shared_0)
    tma.async_load(descriptor, offsets_1, barrier, shared_1)


@gluon.jit
def _copy_rows(base_ptr, rows, shared, copy_layout: gl.constexpr):
    """Copy INT8 rows of one contiguous feature width into shared memory."""
    width: gl.constexpr = shared.shape[1]
    features = gl.arange(0, width, gl.SliceLayout(0, copy_layout))
    async_copy.async_load(shared, base_ptr + rows[:, None] * width + features[None, :])


@gluon.jit
def _copy_key_pair(key_ptr, start_n_0, start_n_1, key_shared_pair, copy_layout: gl.constexpr):
    """Gather two routed K64 row tiles into one paired [K128, D] buffer."""
    head_dim: gl.constexpr = key_shared_pair.shape[1]
    rows = gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(1, copy_layout))
    _copy_rows(
        key_ptr + start_n_0 * head_dim, rows, key_shared_pair.slice(0, _GL_BLOCK_N), copy_layout
    )
    _copy_rows(
        key_ptr + start_n_1 * head_dim,
        rows,
        key_shared_pair.slice(_GL_BLOCK_N, _GL_BLOCK_N),
        copy_layout,
    )


@gluon.jit
def _copy_value_pair(
    value_ptr,
    storage_sequence_length,
    start_n_0,
    start_n_1,
    value_shared_pair,
    copy_layout: gl.constexpr,
):
    """Copy the transposed [D, K64] V tiles of one pair."""
    head_dim: gl.constexpr = value_shared_pair.shape[1]
    features = gl.arange(0, head_dim, gl.SliceLayout(1, copy_layout))
    keys = gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(0, copy_layout))
    offsets = features[:, None] * storage_sequence_length + keys[None, :]
    async_copy.async_load(value_shared_pair.index(0), value_ptr + start_n_0 + offsets)
    async_copy.async_load(value_shared_pair.index(1), value_ptr + start_n_1 + offsets)


# ---------------------------------------------------------------------------
# CTA coordinate and route-count setup
# ---------------------------------------------------------------------------
@gluon.jit
def _schedule_program_coords(
    head,
    local_query_block,
    heads,
    global_query_block_offset,
    sparse_query_blocks,
    group_dense_suffix: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
):
    """Group dense-suffix CTAs first across heads so sparse waves finish together."""
    if group_dense_suffix and has_dense_query_suffix:
        query_blocks = gl.num_programs(0)
        sparse = gl.minimum(
            gl.maximum(sparse_query_blocks - global_query_block_offset, 0), query_blocks
        )
        dense = query_blocks - sparse
        linear = head * query_blocks + local_query_block
        dense_programs = heads * dense
        sparse_linear = linear - dense_programs
        in_dense = linear < dense_programs
        head = gl.where(
            in_dense,
            linear // gl.maximum(dense, 1),
            sparse_linear // gl.maximum(sparse, 1),
        )
        local_query_block = gl.where(
            in_dense,
            sparse + linear % gl.maximum(dense, 1),
            sparse_linear % gl.maximum(sparse, 1),
        )
    return head, local_query_block


@gluon.jit
def _resolve_sparse_route_counts(
    head,
    head_keep_blocks_ptr,
    global_query_block,
    sparse_key_blocks,
    sparse_query_blocks,
    skip_dense_routing: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
):
    """Determine selected KV tile count and routing flag for one CTA."""
    if skip_dense_routing:
        routed_sparse_tile_count = sparse_key_blocks
        selected_sparse_tile_count = sparse_key_blocks
        use_sparse_routes = False
    else:
        routed_sparse_tile_count = gl.load(head_keep_blocks_ptr + head)
        if has_dense_query_suffix:
            use_sparse_routes = global_query_block < sparse_query_blocks
            selected_sparse_tile_count = gl.where(
                use_sparse_routes,
                routed_sparse_tile_count,
                sparse_key_blocks,
            )
        else:
            use_sparse_routes = True
            selected_sparse_tile_count = routed_sparse_tile_count
    return routed_sparse_tile_count, selected_sparse_tile_count, use_sparse_routes


# ---------------------------------------------------------------------------
# SM120-only D128 split-accumulator recurrence
# ---------------------------------------------------------------------------
@gluon.jit
def _dot_d128_slices(
    query_0,
    query_1,
    query_2,
    query_3,
    rhs_shared,
    accumulator,
    key_layout: gl.constexpr,
):
    """Accumulate an INT32 D128 dot product across four D32 shared-memory slices."""
    accumulator = mma_v2(
        query_0, rhs_shared.slice(0, 32, dim=1).permute([1, 0]).load(key_layout), accumulator
    )
    accumulator = mma_v2(
        query_1, rhs_shared.slice(32, 32, dim=1).permute([1, 0]).load(key_layout), accumulator
    )
    accumulator = mma_v2(
        query_2, rhs_shared.slice(64, 32, dim=1).permute([1, 0]).load(key_layout), accumulator
    )
    return mma_v2(
        query_3, rhs_shared.slice(96, 32, dim=1).permute([1, 0]).load(key_layout), accumulator
    )


@gluon.jit
def _compute_k_bias_batch_d128(
    query_0,
    query_1,
    query_2,
    query_3,
    query_scale,
    key_center_ptr,
    key_center_scale_ptr,
    key_center_shared,
    bias_shared,
    tiles_shared,
    route_base,
    pair_index,
    tile_count,
    routed_sparse_tile_count,
    selected_sparse_tile_count,
    sparse_key_blocks,
    stride_rr,
    use_sparse_routes,
    skip_dense_routing: gl.constexpr,
    kv_batch_head,
    sequence_tiles,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
):
    """Compute a 16-tile D128 K-bias batch using D32 query slices and cache it in SMEM.

    Reuses the 4 KiB ``key_center_shared`` allocation (dead after the center
    dots finish) as ``bias_shared`` (`[64, 16]` FP32) until the end-of-chunk
    V-mean flush, freeing 8 live accumulator registers across the main loop.
    """
    gl.static_assert(
        mma_layout.warps_per_cta == [4, 1]  # pyright: ignore[reportAttributeAccessIssue]
    )
    gl.static_assert(query_0.shape == [_GL_BLOCK_N, 32])
    gl.static_assert(key_center_shared.shape == [2 * K_BIAS_GROUP_TILES, 128])
    gl.static_assert(bias_shared.shape == [_GL_BLOCK_N, K_BIAS_GROUP_TILES])
    gl.static_assert(tiles_shared.shape == [2, K_BIAS_GROUP_TILES])

    tile_batch_shared = tiles_shared.index((pair_index // _K_BIAS_GROUP_PAIRS) & 1)
    resolved_tiles = resolve_batch_tiles(
        pair_index * 2,
        tile_count,
        route_base,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        _SHARED_TILE_LAYOUT,
    )
    gl.barrier()
    tile_batch_shared.store(resolved_tiles)
    gl.barrier()

    load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    tile_layout: gl.constexpr = gl.SliceLayout(1, load_layout)
    feature_layout: gl.constexpr = gl.SliceLayout(0, load_layout)
    slots_doubled = gl.arange(0, 2 * K_BIAS_GROUP_TILES, tile_layout)
    tiles = tile_batch_shared.gather(slots_doubled // 2, 0)
    kv_head_base = kv_batch_head.to(gl.int64) * sequence_tiles
    feature_offsets = gl.arange(0, 128, feature_layout)
    center_rows = (kv_head_base + tiles.to(gl.int64)) * 2 + (slots_doubled % 2)
    centers = gl.load(key_center_ptr + center_rows[:, None] * 128 + feature_offsets[None, :])
    key_center_shared.store(centers)
    gl.barrier()
    dots = gl.zeros([_GL_BLOCK_N, 2 * K_BIAS_GROUP_TILES], gl.int32, mma_layout)
    dots = _dot_d128_slices(query_0, query_1, query_2, query_3, key_center_shared, dots, key_layout)
    dot_0, dot_1 = gl.split(gl.reshape(dots, [_GL_BLOCK_N, K_BIAS_GROUP_TILES, 2]))
    dot_0 = gl.convert_layout(dot_0, mma_layout)
    dot_1 = gl.convert_layout(dot_1, mma_layout)

    scale_tiles = tile_batch_shared.load(gl.SliceLayout(0, mma_layout))
    scale_base = (kv_head_base + scale_tiles.to(gl.int64)) * 2
    scale_0 = gl.load(key_center_scale_ptr + scale_base)
    scale_1 = gl.load(key_center_scale_ptr + scale_base + 1)
    bias_batch = (
        dot_0.to(gl.float32) * scale_0[None, :] + dot_1.to(gl.float32) * scale_1[None, :]
    ) * query_scale[:, None]

    # Finish every warp's center loads before aliasing the buffer for FP32 biases.
    gl.barrier()
    bias_shared.store(bias_batch)
    gl.barrier()


@gluon.jit
def _compute_pair_k_bias_d128(
    pair_index,
    query_0,
    query_1,
    query_2,
    query_3,
    query_scale,
    key_center_ptr,
    key_center_scale_ptr,
    key_center_shared,
    bias_shared,
    tiles_shared,
    route_base,
    tile_count,
    routed_sparse_tile_count,
    selected_sparse_tile_count,
    sparse_key_blocks,
    stride_rr,
    use_sparse_routes,
    skip_dense_routing: gl.constexpr,
    kv_batch_head,
    sequence_tiles,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
    row_layout: gl.constexpr,
):
    """Refresh the 8-pair D128 SMEM K-bias cache on boundaries and return one pair."""
    local_pair = pair_index % _K_BIAS_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
    if local_pair == 0:
        _compute_k_bias_batch_d128(
            query_0,
            query_1,
            query_2,
            query_3,
            query_scale,
            key_center_ptr,
            key_center_scale_ptr,
            key_center_shared,
            bias_shared,
            tiles_shared,
            route_base,
            pair_index,
            tile_count,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
            skip_dense_routing,
            kv_batch_head,
            sequence_tiles,
            mma_layout,
            key_layout,
        )
    return (
        load_tile_column(bias_shared, local_pair * 2, row_layout),
        load_tile_column(bias_shared, local_pair * 2 + 1, row_layout),
    )


@gluon.jit
def _qk_pair_d128(
    query_0,
    query_1,
    query_2,
    query_3,
    key_shared_0,
    key_shared_1,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
):
    """Compute INT32 Q @ K^T scores for two K64 tiles via four D32 fragments."""
    zero_scores = gl.zeros([_GL_BLOCK_N, _GL_BLOCK_N], gl.int32, mma_layout)
    integer_scores_0 = _dot_d128_slices(
        query_0, query_1, query_2, query_3, key_shared_0, zero_scores, key_layout
    )
    integer_scores_1 = _dot_d128_slices(
        query_0, query_1, query_2, query_3, key_shared_1, zero_scores, key_layout
    )
    return integer_scores_0, integer_scores_1


@gluon.jit
def _piper_probability_pair_d128(
    integer_scores_0,
    integer_scores_1,
    query_scale,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    block_lengths_ptr,
    k_bias_0,
    k_bias_1,
    denominator,
    running_max,
    kv_batch_head,
    start_n_0,
    start_n_1,
    has_second,
    sequence_tiles,
    logical_sequence_length,
    mma_layout: gl.constexpr,
    probability_layout: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    mask_duplicate: gl.constexpr,
):
    """Run paired online softmax and sequentially quantize tile 0 then tile 1 to UINT8."""
    row_layout: gl.constexpr = running_max.type.layout
    (
        integer_scores_0,
        integer_scores_1,
        score_scale_0,
        score_scale_1,
        value_scale_multiplier_0,
        value_scale_multiplier_1,
        valid_keys_0,
        valid_keys_1,
        block_max,
        next_max,
        old_weight,
        current_weight,
    ) = compute_pair_softmax_bounds(
        integer_scores_0,
        integer_scores_1,
        query_scale,
        key_scale_ptr,
        value_scale_multiplier_ptr,
        block_lengths_ptr,
        k_bias_0,
        k_bias_1,
        running_max,
        kv_batch_head,
        start_n_0,
        start_n_1,
        has_second,
        sequence_tiles,
        logical_sequence_length,
        mma_layout,
        mask_block_lengths,
        mask_ragged_tail,
        mask_duplicate,
    )

    # Materialize tile 0 completely before expanding tile 1 so only one
    # [64, 64] FP32 probability tile occupies registers at a time.
    shifted_0 = (
        integer_scores_0.to(gl.float32) * score_scale_0[:, None] + (k_bias_0 - block_max)[:, None]
    )
    if mask_block_lengths or mask_ragged_tail:
        shifted_0 = gl.where(valid_keys_0[None, :], shifted_0, -float("inf"))
    probabilities_0 = gl.exp2(shifted_0)
    mass_0 = gl.convert_layout(gl.sum(probabilities_0, axis=1), row_layout)
    probability_uint8_0 = gl.convert_layout(
        packed_float32_to_uint8(probabilities_0 * value_scale_multiplier_0 + 0.5),
        probability_layout,
    )

    shifted_1 = (
        integer_scores_1.to(gl.float32) * score_scale_1[:, None] + (k_bias_1 - block_max)[:, None]
    )
    if mask_block_lengths or mask_ragged_tail:
        shifted_1 = gl.where(valid_keys_1[None, :], shifted_1, -float("inf"))
    elif mask_duplicate:
        shifted_1 = gl.where(has_second, shifted_1, -float("inf"))
    probabilities_1 = gl.exp2(shifted_1)
    mass_1 = gl.convert_layout(gl.sum(probabilities_1, axis=1), row_layout)
    if mask_duplicate:
        mass_1 = gl.where(has_second, mass_1, 0.0)
    probability_uint8_1 = gl.convert_layout(
        packed_float32_to_uint8(probabilities_1 * value_scale_multiplier_1 + 0.5),
        probability_layout,
    )

    denominator = denominator * old_weight + (mass_0 + mass_1) * current_weight
    return (
        probability_uint8_0,
        probability_uint8_1,
        denominator,
        next_max,
        old_weight,
        current_weight,
        mass_0,
        mass_1,
        block_max,
    )


@gluon.jit
def _accumulate_pv_strip_d128(
    probability_uint8_0,
    probability_uint8_1,
    value_shared_0,
    value_shared_1,
    accumulator_strip,
    old_weight,
    current_weight,
    strip_offset: gl.constexpr,
    mma_layout: gl.constexpr,
    value_layout: gl.constexpr,
):
    """Accumulate one D64 strip of two K64 PV tiles into an FP32 half-accumulator."""
    partial = uint8_int8_mma(
        probability_uint8_0,
        value_shared_0.slice(strip_offset, 64).permute([1, 0]).load(value_layout),
        gl.zeros([_GL_BLOCK_N, 64], gl.int32, mma_layout),
    )
    partial = uint8_int8_mma(
        probability_uint8_1,
        value_shared_1.slice(strip_offset, 64).permute([1, 0]).load(value_layout),
        partial,
    )
    return rescale_packed(partial, accumulator_strip, old_weight, current_weight)


@gluon.jit
def _piper_pv_pair_d128(
    probability_uint8_0,
    probability_uint8_1,
    value_shared_0,
    value_shared_1,
    accumulator_0,
    accumulator_1,
    old_weight,
    current_weight,
    mma_layout: gl.constexpr,
    value_layout: gl.constexpr,
):
    """Accumulate two K64 PV tiles sequentially across split [64, 64] feature halves."""
    accumulator_0 = _accumulate_pv_strip_d128(
        probability_uint8_0,
        probability_uint8_1,
        value_shared_0,
        value_shared_1,
        accumulator_0,
        old_weight,
        current_weight,
        0,
        mma_layout,
        value_layout,
    )
    accumulator_1 = _accumulate_pv_strip_d128(
        probability_uint8_0,
        probability_uint8_1,
        value_shared_0,
        value_shared_1,
        accumulator_1,
        old_weight,
        current_weight,
        64,
        mma_layout,
        value_layout,
    )
    return accumulator_0, accumulator_1


@gluon.jit
def _accumulate_value_mean_strip_d128(
    accumulator_slice,
    lhs_high,
    lhs_low,
    value_mean_shared,
    mean_ptrs_0,
    mean_ptrs_1,
    valid_0,
    valid_1,
    strip_offset: gl.constexpr,
    all_valid: gl.constexpr,
    feature_layout: gl.constexpr,
    rhs_layout: gl.constexpr,
):
    """Load one D64 slice of up to 32 BF16 V means into shared memory and accumulate."""
    half_dim: gl.constexpr = accumulator_slice.shape[1]
    feature_offsets = strip_offset + gl.arange(0, half_dim, feature_layout)
    if all_valid:
        means_0 = gl.load(mean_ptrs_0 + feature_offsets[None, :])
        means_1 = gl.load(mean_ptrs_1 + feature_offsets[None, :])
    else:
        means_0 = gl.load(mean_ptrs_0 + feature_offsets[None, :], mask=valid_0[:, None], other=0.0)
        means_1 = gl.load(mean_ptrs_1 + feature_offsets[None, :], mask=valid_1[:, None], other=0.0)
    value_mean_shared.index(0).store(means_0)
    value_mean_shared.index(1).store(means_1)
    gl.barrier()
    strip_means = value_mean_shared.reinterpret(
        gl.bfloat16,
        [_VALUE_MEAN_GROUP_TILES, half_dim],
        gl.NVMMASharedLayout.get_default_for([_VALUE_MEAN_GROUP_TILES, half_dim], gl.bfloat16),
    ).load(rhs_layout)
    accumulator_slice = mma_v2(lhs_high, strip_means, accumulator_slice)
    return mma_v2(lhs_low, strip_means, accumulator_slice)


@gluon.jit
def _accumulate_chunk_value_means_d128(
    accumulator_0,
    accumulator_1,
    tile_masses_shared,
    pair_max_shared,
    tiles_shared,
    key_center_shared,
    running_max,
    value_mean_ptr,
    kv_batch_head,
    sequence_tiles,
    num_valid_tiles,
    all_valid: gl.constexpr,
    mma_layout: gl.constexpr,
):
    """Flush up to 32 buffered per-tile V means into split D64 accumulators."""
    gl.static_assert(_VALUE_MEAN_GROUP_PAIRS == 2 * _K_BIAS_GROUP_PAIRS)
    gl.static_assert(_VALUE_MEAN_GROUP_TILES == 2 * K_BIAS_GROUP_TILES)
    gl.static_assert(
        mma_layout.warps_per_cta == [4, 1]  # pyright: ignore[reportAttributeAccessIssue]
    )
    gl.static_assert(accumulator_0.shape == [_GL_BLOCK_N, 64])
    gl.static_assert(accumulator_1.shape == [_GL_BLOCK_N, 64])
    gl.static_assert(tile_masses_shared.shape == [_GL_BLOCK_N, _VALUE_MEAN_GROUP_TILES])
    gl.static_assert(pair_max_shared.shape == [_GL_BLOCK_N, _VALUE_MEAN_GROUP_PAIRS])
    gl.static_assert(tiles_shared.shape == [2, K_BIAS_GROUP_TILES])
    gl.static_assert(key_center_shared.shape[0] * key_center_shared.shape[1] >= 2 * 32 * 64)

    # Ensure all warps have written tile masses/maxes and finished reading
    # bias_shared before key_center_shared is overwritten with V means.
    gl.barrier()
    tile_masses = tile_masses_shared.load(mma_layout)
    pair_max_columns = pair_max_shared.load(mma_layout)
    pair_max = gl.convert_layout(
        gl.reshape(
            gl.join(pair_max_columns, pair_max_columns), [_GL_BLOCK_N, _VALUE_MEAN_GROUP_TILES]
        ),
        mma_layout,
    )
    if all_valid:
        has_valid_scores = pair_max != -float("inf")
    else:
        tile_slots = gl.arange(0, _VALUE_MEAN_GROUP_TILES, gl.SliceLayout(0, mma_layout))[None, :]
        has_valid_scores = (tile_slots < num_valid_tiles) & (pair_max != -float("inf"))
    pair_decay = gl.where(has_valid_scores, gl.exp2(pair_max - running_max[:, None]), 0.0)
    scaled_masses = gl.where(has_valid_scores, tile_masses * (pair_decay * 255.0), 0.0)

    masses_high, masses_low = split_bf16_pair(scaled_masses)
    lhs_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=2)
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=2)
    lhs_high = gl.convert_layout(masses_high, lhs_layout)
    lhs_low = gl.convert_layout(masses_low, lhs_layout)

    tile_load_layout: gl.constexpr = gl.SliceLayout(1, _VALUE_MEAN_LOAD_LAYOUT)
    feature_layout: gl.constexpr = gl.SliceLayout(0, _VALUE_MEAN_LOAD_LAYOUT)
    tiles_0 = tiles_shared.index(0).load(tile_load_layout)
    tiles_1 = tiles_shared.index(1).load(tile_load_layout)
    if all_valid:
        valid_0 = gl.full([K_BIAS_GROUP_TILES], True, gl.int1, tile_load_layout)
        valid_1 = valid_0
        safe_tiles_0 = tiles_0
        safe_tiles_1 = tiles_1
    else:
        mean_slots = gl.arange(0, K_BIAS_GROUP_TILES, tile_load_layout)
        valid_0 = mean_slots < num_valid_tiles
        valid_1 = K_BIAS_GROUP_TILES + mean_slots < num_valid_tiles
        safe_tiles_0 = gl.where(valid_0, tiles_0, 0)
        safe_tiles_1 = gl.where(valid_1, tiles_1, 0)

    base = kv_batch_head.to(gl.int64) * sequence_tiles
    mean_ptrs_0 = value_mean_ptr + (base + safe_tiles_0.to(gl.int64))[:, None] * 128
    mean_ptrs_1 = value_mean_ptr + (base + safe_tiles_1.to(gl.int64))[:, None] * 128
    value_mean_shared = key_center_shared.reinterpret(
        gl.bfloat16,
        [2, K_BIAS_GROUP_TILES, 64],
        gl.NVMMASharedLayout.get_default_for([K_BIAS_GROUP_TILES, 64], gl.bfloat16),
    )
    accumulator_0 = _accumulate_value_mean_strip_d128(
        accumulator_0,
        lhs_high,
        lhs_low,
        value_mean_shared,
        mean_ptrs_0,
        mean_ptrs_1,
        valid_0,
        valid_1,
        0,
        all_valid,
        feature_layout,
        rhs_layout,
    )
    gl.barrier()
    accumulator_1 = _accumulate_value_mean_strip_d128(
        accumulator_1,
        lhs_high,
        lhs_low,
        value_mean_shared,
        mean_ptrs_0,
        mean_ptrs_1,
        valid_0,
        valid_1,
        64,
        all_valid,
        feature_layout,
        rhs_layout,
    )
    return accumulator_0, accumulator_1


@gluon.jit
def _accumulate_value_mean_step_d128(
    accumulator_0,
    accumulator_1,
    tile_masses_shared,
    pair_max_shared,
    tiles_shared,
    key_center_shared,
    running_max,
    value_mean_ptr,
    mass_0,
    mass_1,
    block_max,
    pair_index,
    tile_count,
    kv_batch_head,
    sequence_tiles,
    is_final_pair: gl.constexpr,
    mma_layout: gl.constexpr,
):
    """Record one pair's V-mean metadata and flush 32-tile chunks or the final tail."""
    local_pair = pair_index % _VALUE_MEAN_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
    store_tile_column(tile_masses_shared, mass_0, local_pair * 2)
    store_tile_column(tile_masses_shared, mass_1, local_pair * 2 + 1)
    store_tile_column(pair_max_shared, block_max, local_pair)
    if is_final_pair:
        chunk_offset = (
            pair_index // _VALUE_MEAN_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
        ) * _VALUE_MEAN_GROUP_TILES
        accumulator_0, accumulator_1 = _accumulate_chunk_value_means_d128(
            accumulator_0,
            accumulator_1,
            tile_masses_shared,
            pair_max_shared,
            tiles_shared,
            key_center_shared,
            running_max,
            value_mean_ptr,
            kv_batch_head,
            sequence_tiles,
            tile_count - chunk_offset,
            False,
            mma_layout,
        )
    elif local_pair == _VALUE_MEAN_GROUP_PAIRS - 1:
        accumulator_0, accumulator_1 = _accumulate_chunk_value_means_d128(
            accumulator_0,
            accumulator_1,
            tile_masses_shared,
            pair_max_shared,
            tiles_shared,
            key_center_shared,
            running_max,
            value_mean_ptr,
            kv_batch_head,
            sequence_tiles,
            _VALUE_MEAN_GROUP_TILES,
            True,
            mma_layout,
        )
    return accumulator_0, accumulator_1


@gluon.jit
def _d128_pair_step(
    pair_index,
    accumulator_0,
    accumulator_1,
    denominator,
    running_max,
    start_n_0,
    start_n_1,
    query_0,
    query_1,
    query_2,
    query_3,
    key_desc,
    value_desc,
    key_shared_0,
    key_shared_1,
    value_shared_0,
    value_shared_1,
    key_center_shared,
    bias_shared,
    tile_masses_shared,
    pair_max_shared,
    tiles_shared,
    key_barrier,
    value_barrier,
    query_scale,
    key_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    value_scale_multiplier_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    route_base,
    kv_batch_head,
    storage_sequence_length,
    logical_sequence_length,
    sequence_tiles,
    tile_count,
    routed_sparse_tile_count,
    selected_sparse_tile_count,
    sparse_key_blocks,
    use_sparse_routes,
    stride_rr,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
    probability_layout: gl.constexpr,
    value_layout: gl.constexpr,
    row_layout: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    skip_dense_routing: gl.constexpr,
    is_final: gl.constexpr,
):
    """Run one paired-K128 SM120 D128 step (interior prefetch or peeled final tail)."""
    k_bias_0, k_bias_1 = _compute_pair_k_bias_d128(
        pair_index,
        query_0,
        query_1,
        query_2,
        query_3,
        query_scale,
        key_center_ptr,
        key_center_scale_ptr,
        key_center_shared,
        bias_shared,
        tiles_shared,
        route_base,
        tile_count,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        kv_batch_head,
        sequence_tiles,
        mma_layout,
        key_layout,
        row_layout,
    )
    if not is_final:
        next_n_0, next_n_1 = pair_tile_offsets(
            route_base,
            pair_index + 1,
            tile_count,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
            skip_dense_routing,
            _GL_BLOCK_N,
        )
    else:
        next_n_0, next_n_1 = start_n_0, start_n_1
    phase = pair_index & 1
    has_second = (pair_index * 2 + 1 < tile_count) if is_final else True
    mbarrier.wait(key_barrier, phase=phase)
    integer_scores_0, integer_scores_1 = _qk_pair_d128(
        query_0,
        query_1,
        query_2,
        query_3,
        key_shared_0,
        key_shared_1,
        mma_layout,
        key_layout,
    )
    if not is_final:
        gl.barrier()
        _issue_tma_pair(
            key_desc,
            [kv_batch_head * storage_sequence_length + next_n_0, 0],
            [kv_batch_head * storage_sequence_length + next_n_1, 0],
            key_shared_0,
            key_shared_1,
            key_barrier,
        )
    (
        probability_uint8_0,
        probability_uint8_1,
        denominator,
        running_max,
        old_weight,
        current_weight,
        mass_0,
        mass_1,
        block_max,
    ) = _piper_probability_pair_d128(
        integer_scores_0,
        integer_scores_1,
        query_scale,
        key_scale_ptr,
        value_scale_multiplier_ptr,
        block_lengths_ptr,
        k_bias_0,
        k_bias_1,
        denominator,
        running_max,
        kv_batch_head,
        start_n_0,
        start_n_1,
        has_second,
        sequence_tiles,
        logical_sequence_length,
        mma_layout,
        probability_layout,
        mask_block_lengths,
        mask_ragged_tail if is_final else False,
        is_final,
    )
    mbarrier.wait(value_barrier, phase=phase)
    accumulator_0, accumulator_1 = _piper_pv_pair_d128(
        probability_uint8_0,
        probability_uint8_1,
        value_shared_0,
        value_shared_1,
        accumulator_0,
        accumulator_1,
        old_weight,
        current_weight,
        mma_layout,
        value_layout,
    )
    if not is_final:
        gl.barrier()
        _issue_tma_pair(
            value_desc,
            [kv_batch_head * 128, next_n_0],
            [kv_batch_head * 128, next_n_1],
            value_shared_0,
            value_shared_1,
            value_barrier,
        )
    accumulator_0, accumulator_1 = _accumulate_value_mean_step_d128(
        accumulator_0,
        accumulator_1,
        tile_masses_shared,
        pair_max_shared,
        tiles_shared,
        key_center_shared,
        running_max,
        value_mean_ptr,
        mass_0,
        mass_1,
        block_max,
        pair_index,
        tile_count,
        kv_batch_head,
        sequence_tiles,
        is_final,
        mma_layout,
    )
    return accumulator_0, accumulator_1, denominator, running_max, next_n_0, next_n_1


# ---------------------------------------------------------------------------
# Unified single-accumulator recurrence (SM120 D64 TMA + SM8x D64/D128 cp.async)
# ---------------------------------------------------------------------------
@gluon.jit
def _single_accumulator_pair_step(
    pair_index,
    accumulator,
    denominator,
    running_max,
    bias_cache_0,
    bias_cache_1,
    start_n_0,
    start_n_1,
    key_barrier,
    value_barrier,
    query,
    key_operand,
    value_operand,
    key_shared_pair,
    value_shared_pair,
    key_center_shared,
    query_scale,
    key_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    value_scale_multiplier_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    route_base,
    kv_batch_head,
    storage_sequence_length,
    logical_sequence_length,
    sequence_tiles,
    tile_count,
    routed_sparse_tile_count,
    selected_sparse_tile_count,
    sparse_key_blocks,
    use_sparse_routes,
    stride_rr,
    head_dim: gl.constexpr,
    mma_warps: gl.constexpr,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
    probability_layout: gl.constexpr,
    value_layout: gl.constexpr,
    row_layout: gl.constexpr,
    row_copy_layout: gl.constexpr,
    value_copy_layout: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    skip_dense_routing: gl.constexpr,
    use_tma: gl.constexpr,
    is_final: gl.constexpr,
):
    """Run one single-accumulator paired-K128 step for either TMA or ``cp.async``."""
    k_bias_0, k_bias_1, bias_cache_0, bias_cache_1 = compute_pair_k_bias(
        pair_index,
        bias_cache_0,
        bias_cache_1,
        query,
        query_scale,
        key_center_ptr,
        key_center_scale_ptr,
        key_center_shared,
        route_base,
        tile_count,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        kv_batch_head,
        sequence_tiles,
        head_dim,
        mma_warps,
        mma_layout,
        key_layout,
        row_layout,
    )
    phase = pair_index & 1
    has_second = (pair_index * 2 + 1 < tile_count) if is_final else True
    if use_tma:
        mbarrier.wait(key_barrier, phase=phase)
    else:
        # The consumed pair's K group is followed only by its V group.
        async_copy.wait_group(1)
        gl.barrier()
    (
        probability,
        denominator,
        running_max,
        old_weight,
        current_weight,
        mass_0,
        mass_1,
        _block_max,
    ) = piper_probability_pair(
        query,
        key_shared_pair,
        query_scale,
        key_scale_ptr,
        value_scale_multiplier_ptr,
        block_lengths_ptr,
        k_bias_0,
        k_bias_1,
        denominator,
        running_max,
        kv_batch_head,
        start_n_0,
        start_n_1,
        has_second,
        sequence_tiles,
        logical_sequence_length,
        mma_layout,
        key_layout,
        probability_layout,
        mask_block_lengths,
        mask_ragged_tail if is_final else False,
        is_final,
    )
    if not is_final:
        next_n_0, next_n_1 = pair_tile_offsets(
            route_base,
            pair_index + 1,
            tile_count,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
            skip_dense_routing,
            _GL_BLOCK_N,
        )
        gl.barrier()
        if use_tma:
            _issue_tma_pair(
                key_operand,
                [kv_batch_head * storage_sequence_length + next_n_0, 0],
                [kv_batch_head * storage_sequence_length + next_n_1, 0],
                key_shared_pair.slice(0, _GL_BLOCK_N),
                key_shared_pair.slice(_GL_BLOCK_N, _GL_BLOCK_N),
                key_barrier,
            )
        else:
            _copy_key_pair(key_operand, next_n_0, next_n_1, key_shared_pair, row_copy_layout)
            async_copy.commit_group()
    else:
        next_n_0, next_n_1 = start_n_0, start_n_1

    if use_tma:
        mbarrier.wait(value_barrier, phase=phase)
    else:
        # Interior: next pair's K group follows this V group; final: none follows.
        async_copy.wait_group(0 if is_final else 1)
        gl.barrier()
    accumulator = piper_pv_pair(
        probability,
        value_shared_pair,
        accumulator,
        old_weight,
        current_weight,
        mma_layout,
        value_layout,
    )
    if not is_final:
        gl.barrier()
        if use_tma:
            _issue_tma_pair(
                value_operand,
                [kv_batch_head * head_dim, next_n_0],
                [kv_batch_head * head_dim, next_n_1],
                value_shared_pair.index(0),
                value_shared_pair.index(1),
                value_barrier,
            )
        else:
            _copy_value_pair(
                value_operand,
                storage_sequence_length,
                next_n_0,
                next_n_1,
                value_shared_pair,
                value_copy_layout,
            )
            async_copy.commit_group()
    accumulator = accumulate_value_mean_pair(
        accumulator,
        value_mean_ptr,
        mass_0,
        mass_1,
        current_weight,
        kv_batch_head,
        start_n_0,
        start_n_1,
        sequence_tiles,
        mma_layout,
    )
    return accumulator, denominator, running_max, bias_cache_0, bias_cache_1, next_n_0, next_n_1


# ---------------------------------------------------------------------------
# Unified NVIDIA sparse attention kernel
# ---------------------------------------------------------------------------
@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_ARGUMENTS)
def _sparse_piper_attention_kernel(
    query_operand,
    key_operand,
    value_operand,
    query_scale_ptr,
    key_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    value_scale_multiplier_ptr,
    value_mean_ptr,
    coarse_output_ptr,
    coarse_gate_ptr,
    block_lengths_ptr,
    routes_ptr,
    head_keep_blocks_ptr,
    route_head_offsets_ptr,
    output_ptr,
    query_block_offset,
    global_query_block_offset,
    query_storage_sequence_length,
    storage_sequence_length,
    logical_sequence_length,
    sparse_key_blocks,
    sparse_query_blocks,
    stride_rb,
    stride_rq,
    stride_rr,
    stride_ob,
    stride_oh,
    stride_on,
    stride_cb,
    stride_ch,
    stride_cq,
    stride_gb,
    stride_gh,
    stride_gn,
    heads,
    query_heads_per_kv_head: gl.constexpr,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
    block_m: gl.constexpr,
    mma_warps: gl.constexpr,
    skip_dense_routing: gl.constexpr,
    mask_output_tail: gl.constexpr,
    use_tma: gl.constexpr,
    output_sequence_length,
):
    """Pair native logical K64 tiles in one shared Piper probability coordinate."""
    gl.static_assert(
        block_m == 64 or (use_tma and block_m == 128 and head_dim == 64 and skip_dense_routing)
    )
    gl.static_assert(
        mma_warps == 4 or (use_tma and mma_warps == 2 and head_dim == 64 and block_m == 64)
    )
    gl.static_assert(not apply_coarse_residual or block_m == 64)
    is_d128_tma: gl.constexpr = use_tma and head_dim == 128
    gl.static_assert(not (head_dim == 128 and skip_dense_routing))

    if use_tma:
        head, local_query_block = _schedule_program_coords(
            gl.program_id(1),
            gl.program_id(0) * (block_m // _GL_BLOCK_N),
            heads,
            global_query_block_offset,
            sparse_query_blocks,
            is_d128_tma,
            has_dense_query_suffix,
        )
    else:
        head = gl.program_id(1)
        local_query_block = gl.program_id(0)
    query_block = query_block_offset + local_query_block
    global_query_block = global_query_block_offset + local_query_block
    batch = gl.program_id(2)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // query_heads_per_kv_head) + head // query_heads_per_kv_head
    start_m = query_block * _GL_BLOCK_N
    output_start_m = local_query_block * _GL_BLOCK_N
    if skip_dense_routing:  # noqa: SIM108
        route_head_offset = 0
    else:
        route_head_offset = gl.load(route_head_offsets_ptr + head)
    route_base = (
        routes_ptr + batch * stride_rb + query_block * stride_rq + route_head_offset * stride_rr
    )

    # The packed PV rescale is validated for each selected MMA register layout.
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[mma_warps, 1],
        instr_shape=[16, 8],
    )
    query_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    key_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)

    center_rows: gl.constexpr = (2 if is_d128_tma else 1) * K_BIAS_GROUP_TILES
    key_center_shared = gl.allocate_shared_memory(
        gl.int8,
        [center_rows, head_dim],
        gl.NVMMASharedLayout.get_default_for([center_rows, head_dim], gl.int8),
    )

    if use_tma:
        query_shared = gl.allocate_shared_memory(
            query_operand.dtype, [block_m, head_dim], query_operand.layout
        )
        key_shared_pair = gl.allocate_shared_memory(
            key_operand.dtype, [2 * _GL_BLOCK_N, head_dim], key_operand.layout
        )
        value_shared_pair = gl.allocate_shared_memory(
            value_operand.dtype, [2, head_dim, _GL_BLOCK_N], value_operand.layout
        )
        if is_d128_tma:
            pair_max_shared = gl.allocate_shared_memory(
                gl.float32,
                [_GL_BLOCK_N, _VALUE_MEAN_GROUP_PAIRS],
                _SCRATCH_SHARED_LAYOUT,
            )
            tiles_shared = gl.allocate_shared_memory(
                gl.int32,
                [2, K_BIAS_GROUP_TILES],
                gl.SwizzledSharedLayout(1, 1, 1, [0]),
            )
        else:
            pair_max_shared = None
            tiles_shared = None
        query_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        key_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        value_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(query_barrier, count=1)
        mbarrier.init(key_barrier, count=1)
        mbarrier.init(value_barrier, count=1)
        gl.barrier()
        row_copy_layout: gl.constexpr = None
        value_copy_layout: gl.constexpr = None
        query_base_ptr = None
        key_base_ptr = key_operand
        value_base_ptr = value_operand
    else:
        # Head-major storage can exceed signed 32-bit element offsets.
        query_base_ptr = (
            query_operand + batch_head.to(gl.int64) * query_storage_sequence_length * head_dim
        )
        key_base_ptr = key_operand + kv_batch_head.to(gl.int64) * storage_sequence_length * head_dim
        value_base_ptr = (
            value_operand + kv_batch_head.to(gl.int64) * head_dim * storage_sequence_length
        )
        row_copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, _GL_COPY_BYTES],
            [32 // (head_dim // _GL_COPY_BYTES), head_dim // _GL_COPY_BYTES],
            [mma_warps, 1],
            [1, 0],
        )
        value_copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, _GL_COPY_BYTES],
            [32 // (_GL_BLOCK_N // _GL_COPY_BYTES), _GL_BLOCK_N // _GL_COPY_BYTES],
            [mma_warps, 1],
            [1, 0],
        )
        row_shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
            [_GL_BLOCK_N, head_dim], gl.int8
        )
        value_shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
            [head_dim, _GL_BLOCK_N], gl.int8
        )
        query_shared = gl.allocate_shared_memory(
            gl.int8, [_GL_BLOCK_N, head_dim], row_shared_layout
        )
        key_shared_pair = gl.allocate_shared_memory(
            gl.int8, [2 * _GL_BLOCK_N, head_dim], row_shared_layout
        )
        value_shared_pair = gl.allocate_shared_memory(
            gl.int8, [2, head_dim, _GL_BLOCK_N], value_shared_layout
        )
        pair_max_shared = None
        tiles_shared = None
        query_barrier = None
        key_barrier = None
        value_barrier = None

    routed_sparse_tile_count, selected_sparse_tile_count, use_sparse_routes = (
        _resolve_sparse_route_counts(
            head,
            head_keep_blocks_ptr,
            global_query_block,
            sparse_key_blocks,
            sparse_query_blocks,
            skip_dense_routing,
            has_dense_query_suffix,
        )
    )
    sequence_tiles = storage_sequence_length // _GL_BLOCK_N
    dense_tile_count = sequence_tiles - sparse_key_blocks
    tile_count = selected_sparse_tile_count + dense_tile_count
    pair_count = gl.cdiv(tile_count, 2)
    initial_n_0, initial_n_1 = pair_tile_offsets(
        route_base,
        0,
        tile_count,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        _GL_BLOCK_N,
    )

    if use_tma:
        key_shared_0 = key_shared_pair.slice(0, _GL_BLOCK_N)
        key_shared_1 = key_shared_pair.slice(_GL_BLOCK_N, _GL_BLOCK_N)
        value_shared_0 = value_shared_pair.index(0)
        value_shared_1 = value_shared_pair.index(1)
        _issue_tma(
            query_operand,
            [batch_head * query_storage_sequence_length + start_m, 0],
            query_shared,
            query_barrier,
        )
        _issue_tma_pair(
            key_operand,
            [kv_batch_head * storage_sequence_length + initial_n_0, 0],
            [kv_batch_head * storage_sequence_length + initial_n_1, 0],
            key_shared_0,
            key_shared_1,
            key_barrier,
        )
        _issue_tma_pair(
            value_operand,
            [kv_batch_head * head_dim, initial_n_0],
            [kv_batch_head * head_dim, initial_n_1],
            value_shared_0,
            value_shared_1,
            value_barrier,
        )
        mbarrier.wait(query_barrier, phase=0)
    else:
        query_rows = start_m + gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(1, row_copy_layout))
        _copy_rows(query_base_ptr, query_rows, query_shared, row_copy_layout)
        _copy_key_pair(key_base_ptr, initial_n_0, initial_n_1, key_shared_pair, row_copy_layout)
        async_copy.commit_group()
        _copy_value_pair(
            value_base_ptr,
            storage_sequence_length,
            initial_n_0,
            initial_n_1,
            value_shared_pair,
            value_copy_layout,
        )
        async_copy.commit_group()
        # Q shares the first pair's K group; only its V group may remain in flight.
        async_copy.wait_group(1)
        gl.barrier()

    offsets_m = gl.arange(0, block_m, row_layout)
    query_scale_stride = query_storage_sequence_length // _GL_QUERY_SCALE_ROWS
    if block_m == 128:
        query_scale = gl.load(
            query_scale_ptr
            + batch_head * query_scale_stride
            + (start_m + offsets_m) // _GL_QUERY_SCALE_ROWS,
            mask=start_m + offsets_m < query_storage_sequence_length,
            other=0.0,
        )
    else:
        query_scale = gl.load(
            query_scale_ptr
            + batch_head * query_scale_stride
            + (start_m + offsets_m) // _GL_QUERY_SCALE_ROWS
        )

    start_n_0, start_n_1 = initial_n_0, initial_n_1
    if is_d128_tma:
        key_shared_0 = key_shared_pair.slice(0, _GL_BLOCK_N)
        key_shared_1 = key_shared_pair.slice(_GL_BLOCK_N, _GL_BLOCK_N)
        value_shared_0 = value_shared_pair.index(0)
        value_shared_1 = value_shared_pair.index(1)
        query_0 = query_shared.slice(0, 32, dim=1).load(query_layout)
        query_1 = query_shared.slice(32, 32, dim=1).load(query_layout)
        query_2 = query_shared.slice(64, 32, dim=1).load(query_layout)
        query_3 = query_shared.slice(96, 32, dim=1).load(query_layout)
        # Wait for every warp to finish loading Q before reusing query_shared for tile masses.
        gl.barrier()
        tile_masses_shared = query_shared.reinterpret(
            gl.float32,
            [_GL_BLOCK_N, _VALUE_MEAN_GROUP_TILES],
            _SCRATCH_SHARED_LAYOUT,
        )
        bias_shared = key_center_shared.reinterpret(
            gl.float32,
            [_GL_BLOCK_N, K_BIAS_GROUP_TILES],
            _SCRATCH_SHARED_LAYOUT,
        )
        accumulator_0 = gl.zeros([_GL_BLOCK_N, 64], gl.float32, mma_layout)
        accumulator_1 = gl.zeros([_GL_BLOCK_N, 64], gl.float32, mma_layout)
        denominator = gl.zeros([_GL_BLOCK_N], gl.float32, row_layout)
        running_max = gl.full([_GL_BLOCK_N], -float("inf"), gl.float32, row_layout)

        d128_step_args = (
            query_0,
            query_1,
            query_2,
            query_3,
            key_operand,
            value_operand,
            key_shared_0,
            key_shared_1,
            value_shared_0,
            value_shared_1,
            key_center_shared,
            bias_shared,
            tile_masses_shared,
            pair_max_shared,
            tiles_shared,
            key_barrier,
            value_barrier,
            query_scale,
            key_scale_ptr,
            key_center_ptr,
            key_center_scale_ptr,
            value_scale_multiplier_ptr,
            value_mean_ptr,
            block_lengths_ptr,
            route_base,
            kv_batch_head,
            storage_sequence_length,
            logical_sequence_length,
            sequence_tiles,
            tile_count,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            use_sparse_routes,
            stride_rr,
        )
        d128_step_options = gl.constexpr(
            (
                mma_layout,
                key_layout,
                probability_layout,
                value_layout,
                row_layout,
                mask_block_lengths,
                mask_ragged_tail,
                skip_dense_routing,
            )
        )
        for pair_index in range(pair_count - 1):
            accumulator_0, accumulator_1, denominator, running_max, start_n_0, start_n_1 = (
                _d128_pair_step(
                    pair_index,
                    accumulator_0,
                    accumulator_1,
                    denominator,
                    running_max,
                    start_n_0,
                    start_n_1,
                    *d128_step_args,
                    *d128_step_options,
                    False,
                )
            )
        accumulator_0, accumulator_1, denominator, running_max, start_n_0, start_n_1 = (
            _d128_pair_step(
                pair_count - 1,
                accumulator_0,
                accumulator_1,
                denominator,
                running_max,
                start_n_0,
                start_n_1,
                *d128_step_args,
                *d128_step_options,
                True,
            )
        )

        denom_scale = (gl.maximum(denominator, 1e-30) * 255.0)[:, None]
        valid_queries = global_query_block * _GL_BLOCK_N + offsets_m < logical_sequence_length
        strip_common_args = (
            denom_scale,
            valid_queries,
            coarse_output_ptr,
            coarse_gate_ptr,
            output_ptr,
            batch,
            head,
            query_block,
            output_start_m,
            offsets_m,
            stride_ob,
            stride_oh,
            stride_on,
            stride_cb,
            stride_ch,
            stride_cq,
            stride_gb,
            stride_gh,
            stride_gn,
        )
        strip_options = gl.constexpr(
            (
                mma_layout,
                mask_ragged_tail,
                apply_coarse_residual,
                mask_output_tail,
            )
        )
        store_attention_output_strip(accumulator_0, *strip_common_args, 0, *strip_options)
        store_attention_output_strip(accumulator_1, *strip_common_args, 64, *strip_options)
    else:
        query = query_shared.load(query_layout)
        accumulator = gl.zeros([block_m, head_dim], gl.float32, mma_layout)
        denominator = gl.zeros([block_m], gl.float32, row_layout)
        running_max = gl.full([block_m], -float("inf"), gl.float32, row_layout)
        bias_cache_0, bias_cache_1 = build_k_bias_cache(block_m, mma_layout)
        single_step_args = (
            query,
            key_base_ptr,
            value_base_ptr,
            key_shared_pair,
            value_shared_pair,
            key_center_shared,
            query_scale,
            key_scale_ptr,
            key_center_ptr,
            key_center_scale_ptr,
            value_scale_multiplier_ptr,
            value_mean_ptr,
            block_lengths_ptr,
            route_base,
            kv_batch_head,
            storage_sequence_length,
            logical_sequence_length,
            sequence_tiles,
            tile_count,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            use_sparse_routes,
            stride_rr,
        )
        single_step_options = gl.constexpr(
            (
                head_dim,
                mma_warps,
                mma_layout,
                key_layout,
                probability_layout,
                value_layout,
                row_layout,
                row_copy_layout,
                value_copy_layout,
                mask_block_lengths,
                mask_ragged_tail,
                skip_dense_routing,
                use_tma,
            )
        )

        for pair_index in range(pair_count - 1):
            (
                accumulator,
                denominator,
                running_max,
                bias_cache_0,
                bias_cache_1,
                start_n_0,
                start_n_1,
            ) = _single_accumulator_pair_step(
                pair_index,
                accumulator,
                denominator,
                running_max,
                bias_cache_0,
                bias_cache_1,
                start_n_0,
                start_n_1,
                key_barrier,
                value_barrier,
                *single_step_args,
                *single_step_options,
                False,
            )

        (
            accumulator,
            denominator,
            running_max,
            bias_cache_0,
            bias_cache_1,
            start_n_0,
            start_n_1,
        ) = _single_accumulator_pair_step(
            pair_count - 1,
            accumulator,
            denominator,
            running_max,
            bias_cache_0,
            bias_cache_1,
            start_n_0,
            start_n_1,
            key_barrier,
            value_barrier,
            *single_step_args,
            *single_step_options,
            True,
        )

        store_attention_output(
            accumulator,
            denominator,
            coarse_output_ptr,
            coarse_gate_ptr,
            output_ptr,
            batch,
            head,
            query_block,
            global_query_block,
            output_start_m,
            offsets_m,
            logical_sequence_length,
            output_sequence_length,
            stride_ob,
            stride_oh,
            stride_on,
            stride_cb,
            stride_ch,
            stride_cq,
            stride_gb,
            stride_gh,
            stride_gn,
            mma_layout,
            mask_ragged_tail,
            apply_coarse_residual,
            mask_output_tail,
        )

    if use_tma:
        # Every warp must finish its final waits before any warp invalidates the
        # shared barriers.
        gl.barrier()
        mbarrier.invalidate(query_barrier)
        mbarrier.invalidate(key_barrier)
        mbarrier.invalidate(value_barrier)


# ---------------------------------------------------------------------------
# Host binding and shared launch helpers
# ---------------------------------------------------------------------------
def _descriptor(data: torch.Tensor, shape: list[int], block_shape: list[int]) -> TensorDescriptor:
    with device_context(data.device):
        return TensorDescriptor(
            data,
            shape,
            [shape[1], 1],
            block_shape,
            gl.NVMMASharedLayout.get_default_for(block_shape, gl.int8),
        )


def _validate_copy_storage(*tensors: torch.Tensor) -> None:
    """Check the alignment required by pointer-based 16-byte copies."""
    if any(not tensor.is_contiguous() or tensor.data_ptr() % _COPY_ALIGNMENT for tensor in tensors):
        raise ValueError("cp.async sparse Piper requires contiguous 16-byte-aligned Q/K/V storage")


def _launch_kernel(
    context: _PreparedSparsePiperContext,
    query_state: _PreparedSparsePiperQuery,
    output: torch.Tensor,
    launch: _ValidatedAttentionLaunch,
    *,
    query_operand: torch.Tensor | TensorDescriptor,
    key_operand: torch.Tensor | TensorDescriptor,
    value_operand: torch.Tensor | TensorDescriptor,
    block_m: int,
    num_warps: int,
    use_tma: bool,
    compile_options: dict[str, int],
) -> None:
    with device_context(output.device):
        install_uint8_int8_dot_hook()
        grid = (
            (launch.query_rows + block_m - 1) // block_m,
            launch.heads,
            launch.batch,
        )
        _sparse_piper_attention_kernel[grid](
            query_operand,
            key_operand,
            value_operand,
            query_state.scale,
            context.key_scale,
            context.key_center,
            context.key_center_scale,
            context.value_scale_multiplier,
            context.value_mean,
            launch.coarse_output,
            launch.coarse_gate,
            launch.block_lengths,
            query_state.routes,
            context.head_keep_blocks,
            context.route_head_offsets,
            output,
            launch.query_block_offset,
            launch.global_query_block_offset,
            launch.query_storage_sequence_length,
            launch.storage_sequence_length,
            launch.logical_sequence_length,
            launch.sparse_key_blocks,
            launch.sparse_query_blocks,
            *launch.route_strides,
            *launch.output_strides,
            *launch.coarse_strides,
            *launch.gate_strides,
            launch.heads,
            launch.heads // context.key.shape[1],
            launch.head_dim,
            launch.mask_block_lengths,
            launch.mask_ragged_tail,
            launch.has_dense_query_suffix,
            launch.apply_coarse_residual,
            block_m,
            num_warps,
            launch.skip_dense_routing,
            launch.output_sequence_length % block_m != 0,
            use_tma,
            launch.output_sequence_length,
            num_warps=num_warps,
            num_stages=1,
            **compile_options,
        )


def bind_tma_context(context: _PreparedSparsePiperContext) -> _ContextLauncher:
    """Prepare K/V TMA descriptors once; Q descriptors belong to each query window."""
    if context.routes_per_query == 0 and context.key.shape[-1] != 64:
        raise ValueError("skip_dense_routing requires NVIDIA D64 attention")
    batch, heads, sequence, head_dim = context.key.shape
    key_descriptor = _descriptor(
        context.key, [batch * heads * sequence, head_dim], [_BLOCK_N, head_dim]
    )
    value_descriptor = _descriptor(
        context.value, [batch * heads * head_dim, sequence], [head_dim, _BLOCK_N]
    )

    def _launch(
        bound_context: _PreparedSparsePiperContext,
        query_state: _PreparedSparsePiperQuery,
        output: torch.Tensor,
        launch: _ValidatedAttentionLaunch,
    ) -> None:
        block_m, num_warps = policy.select_sm120_attention_schedule(
            launch.head_dim,
            launch.query_rows,
            launch.storage_sequence_length,
            skip_dense_routing=launch.skip_dense_routing,
            has_coarse_residual=launch.apply_coarse_residual,
            selected_key_rows=(
                bound_context.routes_per_query // launch.heads * _BLOCK_N
                + launch.storage_sequence_length
                - launch.sparse_key_blocks * _BLOCK_N
            ),
        )
        query_descriptor = _descriptor(
            query_state.data,
            [launch.batch * launch.heads * launch.query_storage_sequence_length, launch.head_dim],
            [block_m, launch.head_dim],
        )
        _launch_kernel(
            bound_context,
            query_state,
            output,
            launch,
            query_operand=query_descriptor,
            key_operand=key_descriptor,
            value_operand=value_descriptor,
            block_m=block_m,
            num_warps=num_warps,
            use_tma=True,
            compile_options={},
        )

    return _ContextLauncher(context, _launch)


def bind_async_copy_context(context: _PreparedSparsePiperContext) -> _ContextLauncher:
    """Bind immutable sequence-global storage for SM8x ``cp.async`` execution."""
    if context.routes_per_query == 0 and context.key.shape[-1] != 64:
        raise ValueError("skip_dense_routing requires NVIDIA D64 attention")
    _validate_copy_storage(context.key, context.value)

    def _launch(
        bound_context: _PreparedSparsePiperContext,
        query_state: _PreparedSparsePiperQuery,
        output: torch.Tensor,
        launch: _ValidatedAttentionLaunch,
    ) -> None:
        _validate_copy_storage(query_state.data)
        max_registers = policy.select_sm8x_max_registers(launch.head_dim)
        compile_options = {} if max_registers is None else {"maxnreg": max_registers}
        _launch_kernel(
            bound_context,
            query_state,
            output,
            launch,
            query_operand=query_state.data,
            key_operand=bound_context.key,
            value_operand=bound_context.value,
            block_m=_BLOCK_N,
            num_warps=_ASYNC_COPY_WARPS,
            use_tma=False,
            compile_options=compile_options,
        )

    return _ContextLauncher(context, _launch)
