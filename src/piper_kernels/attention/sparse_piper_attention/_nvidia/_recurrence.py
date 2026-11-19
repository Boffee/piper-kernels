"""Paired-K128 recurrence arithmetic shared by the NVIDIA sparse Piper kernels.

Every function here consumes register tensors or shared-memory descriptors that
the calling kernel has already filled, so the SM120 TMA kernel and the SM8x
``cp.async`` kernel differ only in how operands reach shared memory. Keeping the
arithmetic in one place preserves one numerical contract across targets:
grouped INT8 Q/K, centered tile-scaled INT8 V, UINT8 probabilities, paired K128
online softmax, per-pair FP32 or 32-tile BF16 tensor-core V-mean restoration, and
an FP32 numerator and denominator rounded once at the store.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from piper_kernels.attention.kernels.piper._nvidia.fragments import (
    packed_float32_to_uint8,
    rescale_packed,
    uint8_int8_mma,
)
from piper_kernels.attention.kernels.sparse_piper.gluon import tile_offset
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS

_LOG2_255 = 7.994353436858858

_GL_BLOCK_N = gl.constexpr(TILE_ROWS)
_GL_LOG2_255 = gl.constexpr(_LOG2_255)
_GL_VALUE_LOG_BOUND_CORRECTION = gl.constexpr(0.086085)
K_BIAS_GROUP_TILES = gl.constexpr(16)
_K_BIAS_GROUP_PAIRS = gl.constexpr(8)
VALUE_MEAN_GROUP_PAIRS = gl.constexpr(16)
VALUE_MEAN_GROUP_TILES = gl.constexpr(32)
_SHARED_TILE_LAYOUT = gl.constexpr(gl.BlockedLayout([1], [32], [4], [0]))
_TILE_COLUMN_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 1], [16, 2], [4, 1], [0, 1]))
_VALUE_MEAN_LOAD_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0]))
SCRATCH_SHARED_LAYOUT = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))


@gluon.jit
def _fma_fp32(lhs, rhs, addend):
    return gl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[lhs, rhs, addend],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _resolve_batch_tiles(
    batch_tile_start,
    tile_count,
    route_base,
    routed_sparse_tile_count,
    selected_sparse_tile_count,
    sparse_key_blocks,
    stride_rr,
    use_sparse_routes,
    skip_dense_routing: gl.constexpr,
    layout: gl.constexpr,
):
    positions = gl.minimum(
        batch_tile_start + gl.arange(0, K_BIAS_GROUP_TILES, layout),
        tile_count - 1,
    )
    return tile_offset(
        route_base,
        positions,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
    )


@gluon.jit
def build_k_bias_cache(block_m: gl.constexpr, mma_layout: gl.constexpr):
    """Build the initial 8-pair K-bias batch cache ahead of the recurrence loop."""
    dummy = gl.zeros([block_m, K_BIAS_GROUP_TILES], gl.float32, mma_layout)
    return gl.split(gl.reshape(dummy, [block_m, _K_BIAS_GROUP_PAIRS, 2]))


@gluon.jit
def _compute_k_bias_batch(
    query,
    query_scale,
    key_center_ptr,
    key_center_scale_ptr,
    key_center_shared,
    route_base,
    batch_tile_start,
    tile_count,
    routed_sparse_tile_count,
    selected_sparse_tile_count,
    sparse_key_blocks,
    stride_rr,
    use_sparse_routes,
    skip_dense_routing: gl.constexpr,
    kv_batch_head,
    sequence_tiles,
    head_dim: gl.constexpr,
    mma_warps: gl.constexpr,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
):
    """Compute Q @ rotated_center^T score biases for one batch of 16 routed/dense tiles."""
    gl.static_assert(key_center_shared.shape[0] == K_BIAS_GROUP_TILES)
    elements_d: gl.constexpr = 8 if (mma_warps == 4 and head_dim == 64) else 16
    threads_d: gl.constexpr = head_dim // elements_d
    threads_m: gl.constexpr = 32 // threads_d
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, elements_d], [threads_m, threads_d], [mma_warps, 1], [1, 0]
    )
    tile_layout: gl.constexpr = gl.SliceLayout(1, load_layout)
    feature_layout: gl.constexpr = gl.SliceLayout(0, load_layout)
    tiles = _resolve_batch_tiles(
        batch_tile_start,
        tile_count,
        route_base,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        tile_layout,
    )

    kv_head_base = kv_batch_head.to(gl.int64) * sequence_tiles
    feature_offsets = gl.arange(0, head_dim, feature_layout)
    center_pointers_0 = (
        key_center_ptr
        + (kv_head_base + tiles[:, None].to(gl.int64)) * (2 * head_dim)
        + feature_offsets[None, :]
    )
    center_pointers_1 = center_pointers_0 + head_dim
    zero_acc = gl.zeros([query.shape[0], K_BIAS_GROUP_TILES], gl.int32, mma_layout)
    center_0 = gl.load(center_pointers_0)
    key_center_shared.store(center_0)
    gl.barrier()
    dot_0 = mma_v2(query, key_center_shared.permute([1, 0]).load(key_layout), zero_acc)
    center_1 = gl.load(center_pointers_1)
    gl.barrier()
    key_center_shared.store(center_1)
    gl.barrier()
    dot_1 = mma_v2(query, key_center_shared.permute([1, 0]).load(key_layout), zero_acc)

    scale_tiles = _resolve_batch_tiles(
        batch_tile_start,
        tile_count,
        route_base,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        gl.SliceLayout(0, mma_layout),
    )
    scale_base = (kv_head_base + scale_tiles.to(gl.int64)) * 2
    scale_0 = gl.load(key_center_scale_ptr + scale_base)
    scale_1 = gl.load(key_center_scale_ptr + scale_base + 1)
    return (
        dot_0.to(gl.float32) * scale_0[None, :] + dot_1.to(gl.float32) * scale_1[None, :]
    ) * query_scale[:, None]


@gluon.jit
def _extract_pair_k_bias(bias_cache_0, bias_cache_1, local_pair, row_layout: gl.constexpr):
    """Select one pair's tile biases from the active 16-tile bias batch."""
    pair_columns: gl.constexpr = gl.arange(
        0, _K_BIAS_GROUP_PAIRS, gl.SliceLayout(0, bias_cache_0.type.layout)
    )[None, :]
    k_bias_0 = gl.convert_layout(
        gl.sum(gl.where(pair_columns == local_pair, bias_cache_0, 0.0), axis=1), row_layout
    )
    k_bias_1 = gl.convert_layout(
        gl.sum(gl.where(pair_columns == local_pair, bias_cache_1, 0.0), axis=1), row_layout
    )
    return k_bias_0, k_bias_1


@gluon.jit
def compute_pair_k_bias(
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
    skip_dense_routing: gl.constexpr,
    kv_batch_head,
    sequence_tiles,
    head_dim: gl.constexpr,
    mma_warps: gl.constexpr,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
    row_layout: gl.constexpr,
):
    """Refresh the 8-pair K-bias batch on boundaries and return this pair's biases."""
    local_pair = pair_index % _K_BIAS_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
    if local_pair == 0:
        bias_batch = _compute_k_bias_batch(
            query,
            query_scale,
            key_center_ptr,
            key_center_scale_ptr,
            key_center_shared,
            route_base,
            pair_index * 2,
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
        )
        bias_cache_0, bias_cache_1 = gl.split(
            gl.reshape(bias_batch, [query.shape[0], _K_BIAS_GROUP_PAIRS, 2])
        )
    k_bias_0, k_bias_1 = _extract_pair_k_bias(bias_cache_0, bias_cache_1, local_pair, row_layout)
    return k_bias_0, k_bias_1, bias_cache_0, bias_cache_1


@gluon.jit
def _compute_pair_softmax_bounds(
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
    mma_layout: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    mask_duplicate: gl.constexpr,
):
    """Load pair scales, mask invalid keys, and compute online-softmax bounds."""
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    offsets_n = gl.arange(0, _GL_BLOCK_N, column_layout)
    tile_0 = start_n_0 // _GL_BLOCK_N
    tile_1 = start_n_1 // _GL_BLOCK_N
    key_scale_0 = gl.load(key_scale_ptr + kv_batch_head * sequence_tiles + tile_0)
    key_scale_1 = gl.load(key_scale_ptr + kv_batch_head * sequence_tiles + tile_1)
    score_scale_0 = query_scale * key_scale_0
    score_scale_1 = query_scale * key_scale_1

    valid_keys_0 = gl.full([_GL_BLOCK_N], True, gl.int1, column_layout)
    valid_keys_1 = valid_keys_0
    if mask_block_lengths:
        valid_keys_0 = offsets_n < gl.load(block_lengths_ptr + tile_0)
        valid_keys_1 = offsets_n < gl.load(block_lengths_ptr + tile_1)
    elif mask_ragged_tail:
        valid_keys_0 = start_n_0 + offsets_n < logical_sequence_length
        valid_keys_1 = start_n_1 + offsets_n < logical_sequence_length
    if mask_block_lengths or mask_ragged_tail:
        if mask_duplicate:
            valid_keys_1 &= has_second
        integer_scores_0 = gl.where(valid_keys_0[None, :], integer_scores_0, -2147483648)
        integer_scores_1 = gl.where(valid_keys_1[None, :], integer_scores_1, -2147483648)

    score_max_0 = gl.max(integer_scores_0, axis=1).to(gl.float32) * score_scale_0 + k_bias_0  # pyright: ignore[reportAttributeAccessIssue]
    score_max_1 = gl.max(integer_scores_1, axis=1).to(gl.float32) * score_scale_1 + k_bias_1  # pyright: ignore[reportAttributeAccessIssue]
    if mask_duplicate:
        score_max_1 = gl.where(has_second, score_max_1, -float("inf"))

    value_scale_multiplier_0 = gl.load(
        value_scale_multiplier_ptr + kv_batch_head * sequence_tiles + tile_0
    ).to(gl.float32)
    value_scale_multiplier_1 = gl.load(
        value_scale_multiplier_ptr + kv_batch_head * sequence_tiles + tile_1
    ).to(gl.float32)
    value_log_scale_0 = value_scale_multiplier_0.to(gl.int32, bitcast=True).to(gl.float32) * (
        1.0 / 8388608.0
    ) - (127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION)
    value_log_scale_1 = value_scale_multiplier_1.to(gl.int32, bitcast=True).to(gl.float32) * (
        1.0 / 8388608.0
    ) - (127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION)
    block_max = gl.maximum(
        score_max_0 + value_log_scale_0,
        score_max_1 + value_log_scale_1,
    )
    next_max = gl.maximum(running_max, block_max)
    old_weight = gl.exp2(running_max - next_max)
    current_weight = gl.exp2(block_max - next_max)
    return (
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
    )


@gluon.jit
def piper_probability_pair(
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
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
    probability_layout: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    mask_duplicate: gl.constexpr,
):
    """Advance one shared Piper coordinate over two independently scaled K64 tiles."""
    block_m: gl.constexpr = query.shape[0]
    row_layout: gl.constexpr = running_max.type.layout
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    key = key_shared_pair.permute([1, 0]).load(key_layout)
    integer_scores = mma_v2(
        query,
        key,
        gl.zeros([block_m, 2 * _GL_BLOCK_N], gl.int32, mma_layout),
    )
    integer_scores_0, integer_scores_1 = gl.split(
        gl.permute(
            gl.reshape(integer_scores, [block_m, 2, _GL_BLOCK_N]),
            [0, 2, 1],
        )
    )
    integer_scores_0 = gl.convert_layout(integer_scores_0, mma_layout)
    integer_scores_1 = gl.convert_layout(integer_scores_1, mma_layout)
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
    ) = _compute_pair_softmax_bounds(
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
    delta_0 = k_bias_0 - block_max
    delta_1 = k_bias_1 - block_max
    shifted_0 = integer_scores_0.to(gl.float32) * score_scale_0[:, None] + delta_0[:, None]
    shifted_1 = integer_scores_1.to(gl.float32) * score_scale_1[:, None] + delta_1[:, None]
    shifted_scores = gl.reshape(
        gl.permute(gl.join(shifted_0, shifted_1), [0, 2, 1]),
        [block_m, 2 * _GL_BLOCK_N],
    )
    shifted_scores = gl.convert_layout(shifted_scores, mma_layout)
    paired_columns = gl.arange(0, 2 * _GL_BLOCK_N, column_layout)
    value_scale_multiplier = gl.where(
        paired_columns < _GL_BLOCK_N,
        value_scale_multiplier_0,
        value_scale_multiplier_1,
    )
    if mask_block_lengths or mask_ragged_tail:
        valid_keys = gl.reshape(
            gl.permute(gl.join(valid_keys_0, valid_keys_1), [1, 0]), [2 * _GL_BLOCK_N]
        )
        valid_keys = gl.convert_layout(valid_keys, column_layout)
        shifted_scores = gl.where(valid_keys[None, :], shifted_scores, -float("inf"))
    elif mask_duplicate:
        valid_keys = (paired_columns < _GL_BLOCK_N) | has_second
        shifted_scores = gl.where(valid_keys[None, :], shifted_scores, -float("inf"))
    probabilities = gl.exp2(shifted_scores)
    probability_uint8 = packed_float32_to_uint8(
        probabilities * value_scale_multiplier[None, :] + 0.5
    )
    probability_uint8 = gl.convert_layout(probability_uint8, probability_layout)
    p0, p1 = gl.split(gl.permute(gl.reshape(probabilities, [block_m, 2, _GL_BLOCK_N]), [0, 2, 1]))
    mass_0 = gl.convert_layout(gl.sum(p0, axis=1), row_layout)
    mass_1 = gl.convert_layout(gl.sum(p1, axis=1), row_layout)
    if mask_duplicate:
        mass_1 = gl.where(has_second, mass_1, 0.0)
    denominator = denominator * old_weight + (mass_0 + mass_1) * current_weight
    return (
        probability_uint8,
        denominator,
        next_max,
        old_weight,
        current_weight,
        mass_0,
        mass_1,
        block_max,
    )


@gluon.jit
def piper_pv_pair(
    probability_uint8,
    value_shared_pair,
    accumulator,
    old_weight,
    current_weight,
    mma_layout: gl.constexpr,
    value_layout: gl.constexpr,
):
    """Accumulate two K64 PV tiles in INT32 and update the FP32 numerator."""
    head_dim: gl.constexpr = accumulator.shape[1]
    block_m: gl.constexpr = accumulator.shape[0]
    value_0 = value_shared_pair.index(0).permute([1, 0]).load(value_layout)
    value_1 = value_shared_pair.index(1).permute([1, 0]).load(value_layout)
    value = gl.reshape(
        gl.permute(gl.join(value_0, value_1), [2, 0, 1]),
        [2 * _GL_BLOCK_N, head_dim],
    )
    value = gl.convert_layout(value, value_layout)
    # Each product sum is bounded by 128 * 255 * 128 = 4,177,920 in magnitude:
    # safe in INT32 and exactly representable when converted to FP32.
    partial = uint8_int8_mma(
        probability_uint8,
        value,
        gl.zeros([block_m, head_dim], gl.int32, mma_layout),
    )
    return rescale_packed(partial, accumulator, old_weight, current_weight)


@gluon.jit
def accumulate_value_mean_pair(
    accumulator,
    value_mean_ptr,
    mass_0,
    mass_1,
    current_weight,
    kv_batch_head,
    start_n_0,
    start_n_1,
    sequence_tiles,
    mma_layout: gl.constexpr,
):
    """Add per-K64 tile V mean restoration directly into the FP32 numerator."""
    head_dim: gl.constexpr = accumulator.shape[1]
    offsets_d = gl.arange(0, head_dim, gl.SliceLayout(0, mma_layout))
    base = kv_batch_head.to(gl.int64) * sequence_tiles
    tile_0 = (start_n_0 // _GL_BLOCK_N).to(gl.int64)
    tile_1 = (start_n_1 // _GL_BLOCK_N).to(gl.int64)
    value_mean_0 = gl.load(value_mean_ptr + (base + tile_0) * head_dim + offsets_d).to(gl.float32)
    value_mean_1 = gl.load(value_mean_ptr + (base + tile_1) * head_dim + offsets_d).to(gl.float32)
    weight_0 = (mass_0 * current_weight * 255.0)[:, None]
    weight_1 = (mass_1 * current_weight * 255.0)[:, None]
    return accumulator + weight_0 * value_mean_0[None, :] + weight_1 * value_mean_1[None, :]


@gluon.jit
def _store_attention_output_strip(
    accumulator_strip,
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
    strip_offset: gl.constexpr,
    mma_layout: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
    mask_output_tail: gl.constexpr,
):
    """Normalize one head-width strip, add an optional coarse residual, and store."""
    width: gl.constexpr = accumulator_strip.shape[1]
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    offsets_d = strip_offset + gl.arange(0, width, column_layout)
    output = accumulator_strip / denom_scale
    if apply_coarse_residual:
        coarse = gl.load(
            coarse_output_ptr
            + batch * stride_cb
            + head * stride_ch
            + query_block * stride_cq
            + offsets_d
        ).to(gl.float32)
        gate_offsets = (
            batch * stride_gb
            + head * stride_gh
            + (output_start_m + offsets_m[:, None]) * stride_gn
            + offsets_d[None, :]
        )
        if mask_ragged_tail:
            gate = gl.load(
                coarse_gate_ptr + gate_offsets,
                mask=valid_queries[:, None],
                other=0.0,
            ).to(gl.float32)
        else:
            gate = gl.load(coarse_gate_ptr + gate_offsets).to(gl.float32)
        # Round once at the output store, after combining both FP32 terms.
        output = _fma_fp32(gate, coarse[None, :], output)
    output_offsets = (
        batch * stride_ob
        + head * stride_oh
        + (output_start_m + offsets_m[:, None]) * stride_on
        + offsets_d[None, :]
    )
    if mask_output_tail:
        gl.store(
            output_ptr + output_offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=valid_queries[:, None],
        )
    else:
        gl.store(output_ptr + output_offsets, output.to(output_ptr.dtype.element_ty))


@gluon.jit
def store_attention_output(
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
    mma_layout: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
    mask_output_tail: gl.constexpr,
):
    """Normalize, add an optional coarse residual, and store once."""
    block_m: gl.constexpr = accumulator.shape[0]
    denom_scale = (gl.maximum(denominator, 1e-30) * 255.0)[:, None]
    valid_queries = global_query_block * _GL_BLOCK_N + offsets_m < logical_sequence_length
    if block_m == 128:
        valid_queries = output_start_m + offsets_m < output_sequence_length
    _store_attention_output_strip(
        accumulator,
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
        0,
        mma_layout,
        mask_ragged_tail,
        apply_coarse_residual,
        mask_output_tail,
    )


@gluon.jit
def store_attention_output_d128(
    accumulator_0,
    accumulator_1,
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
    stride_ob,
    stride_oh,
    stride_on,
    stride_cb,
    stride_ch,
    stride_cq,
    stride_gb,
    stride_gh,
    stride_gn,
    mma_layout: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
    mask_output_tail: gl.constexpr,
):
    """Normalize and store a split-D64 pair of D128 accumulators."""
    denom_scale = (gl.maximum(denominator, 1e-30) * 255.0)[:, None]
    valid_queries = global_query_block * _GL_BLOCK_N + offsets_m < logical_sequence_length
    _store_attention_output_strip(
        accumulator_0,
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
        0,
        mma_layout,
        mask_ragged_tail,
        apply_coarse_residual,
        mask_output_tail,
    )
    _store_attention_output_strip(
        accumulator_1,
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
        64,
        mma_layout,
        mask_ragged_tail,
        apply_coarse_residual,
        mask_output_tail,
    )


@gluon.jit
def _store_tile_column(shared, values, column_index):
    """Store one [64]-row FP32 metric vector into column `column_index` of scratch SMEM."""
    row_layout: gl.constexpr = gl.SliceLayout(1, _TILE_COLUMN_LAYOUT)
    shared.scatter(
        gl.convert_layout(values, row_layout)[:, None],
        gl.full([_GL_BLOCK_N, 1], column_index, gl.int32, _TILE_COLUMN_LAYOUT),
        1,
    )


@gluon.jit
def _split_bf16_pair(value):
    """Split finite non-negative FP32 values into two BF16 terms (high + low)."""
    bits = value.to(gl.uint32, bitcast=True)
    high = (bits >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    high_fp32 = (bits & 0xFFFF0000).to(gl.float32, bitcast=True)
    residual_bits = (value - high_fp32).to(gl.uint32, bitcast=True)
    low = ((residual_bits + 0x8000) >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    return high, low


@gluon.jit
def _accumulate_value_mean_strip(
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
        [VALUE_MEAN_GROUP_TILES, half_dim],
        gl.NVMMASharedLayout.get_default_for([VALUE_MEAN_GROUP_TILES, half_dim], gl.bfloat16),
    ).load(rhs_layout)
    accumulator_slice = mma_v2(lhs_high, strip_means, accumulator_slice)
    return mma_v2(lhs_low, strip_means, accumulator_slice)


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
    resolved_tiles = _resolve_batch_tiles(
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
    # Wait for prior users of bias_shared / value_mean_shared / tile_batch_shared
    # before staging the next 16 tile indices and loading their centers.
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
    dots = _dot_d128_slices(
        query_0,
        query_1,
        query_2,
        query_3,
        key_center_shared,
        dots,
        key_layout,
    )
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
def _load_pair_k_bias_d128(bias_shared, local_pair, row_layout: gl.constexpr):
    """Gather one pair's K-bias row vectors from cached shared memory."""
    # _TILE_COLUMN_LAYOUT places the column index along the fast 2-thread axis
    # and avoids extra layout-conversion spills seen with mma_layout.
    column_0 = gl.full([_GL_BLOCK_N, 1], local_pair * 2, gl.int32, _TILE_COLUMN_LAYOUT)
    column_1 = gl.full([_GL_BLOCK_N, 1], local_pair * 2 + 1, gl.int32, _TILE_COLUMN_LAYOUT)
    bias_0 = gl.reshape(bias_shared.gather(column_0, 1), [_GL_BLOCK_N])
    bias_1 = gl.reshape(bias_shared.gather(column_1, 1), [_GL_BLOCK_N])
    return gl.convert_layout(bias_0, row_layout), gl.convert_layout(bias_1, row_layout)


@gluon.jit
def compute_pair_k_bias_d128(
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
    return _load_pair_k_bias_d128(bias_shared, local_pair, row_layout)


@gluon.jit
def qk_pair_d128(
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
def piper_probability_pair_d128(
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
    ) = _compute_pair_softmax_bounds(
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
def piper_pv_pair_d128(
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
def _record_value_mean_pair_d128(
    tile_masses_shared,
    pair_max_shared,
    mass_0,
    mass_1,
    block_max,
    local_pair,
):
    """Record one pair's unscaled tile masses and block max into chunk scratch SMEM."""
    _store_tile_column(tile_masses_shared, mass_0, local_pair * 2)
    _store_tile_column(tile_masses_shared, mass_1, local_pair * 2 + 1)
    _store_tile_column(pair_max_shared, block_max, local_pair)


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
    gl.static_assert(VALUE_MEAN_GROUP_PAIRS == 2 * _K_BIAS_GROUP_PAIRS)
    gl.static_assert(VALUE_MEAN_GROUP_TILES == 2 * K_BIAS_GROUP_TILES)
    gl.static_assert(
        mma_layout.warps_per_cta == [4, 1]  # pyright: ignore[reportAttributeAccessIssue]
    )
    gl.static_assert(accumulator_0.shape == [_GL_BLOCK_N, 64])
    gl.static_assert(accumulator_1.shape == [_GL_BLOCK_N, 64])
    gl.static_assert(tile_masses_shared.shape == [_GL_BLOCK_N, VALUE_MEAN_GROUP_TILES])
    gl.static_assert(pair_max_shared.shape == [_GL_BLOCK_N, VALUE_MEAN_GROUP_PAIRS])
    gl.static_assert(tiles_shared.shape == [2, K_BIAS_GROUP_TILES])
    gl.static_assert(key_center_shared.shape[0] * key_center_shared.shape[1] >= 2 * 32 * 64)

    # Ensure all warps have written tile masses/maxes and finished reading
    # bias_shared before key_center_shared is overwritten with V means.
    gl.barrier()
    tile_masses = tile_masses_shared.load(mma_layout)
    pair_max_columns = pair_max_shared.load(mma_layout)
    pair_max = gl.convert_layout(
        gl.reshape(
            gl.join(pair_max_columns, pair_max_columns), [_GL_BLOCK_N, VALUE_MEAN_GROUP_TILES]
        ),
        mma_layout,
    )
    if all_valid:
        has_valid_scores = pair_max != -float("inf")
    else:
        tile_slots = gl.arange(0, VALUE_MEAN_GROUP_TILES, gl.SliceLayout(0, mma_layout))[None, :]
        has_valid_scores = (tile_slots < num_valid_tiles) & (pair_max != -float("inf"))
    pair_decay = gl.where(has_valid_scores, gl.exp2(pair_max - running_max[:, None]), 0.0)
    scaled_masses = gl.where(has_valid_scores, tile_masses * (pair_decay * 255.0), 0.0)

    masses_high, masses_low = _split_bf16_pair(scaled_masses)
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
    accumulator_0 = _accumulate_value_mean_strip(
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
    accumulator_1 = _accumulate_value_mean_strip(
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
def accumulate_value_mean_step_d128(
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
    local_pair = pair_index % VALUE_MEAN_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
    _record_value_mean_pair_d128(
        tile_masses_shared, pair_max_shared, mass_0, mass_1, block_max, local_pair
    )
    if is_final_pair:
        chunk_offset = (
            pair_index // VALUE_MEAN_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
        ) * VALUE_MEAN_GROUP_TILES
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
    elif local_pair == VALUE_MEAN_GROUP_PAIRS - 1:
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
            VALUE_MEAN_GROUP_TILES,
            True,
            mma_layout,
        )
    return accumulator_0, accumulator_1
