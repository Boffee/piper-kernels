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
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
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
_VALUE_MEAN_GROUP_PAIRS = gl.constexpr(16)
_VALUE_MEAN_GROUP_TILES = gl.constexpr(32)
_SHARED_TILE_LAYOUT = gl.constexpr(gl.BlockedLayout([1], [32], [4], [0]))
_TILE_COLUMN_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 1], [16, 2], [4, 1], [0, 1]))
_VALUE_MEAN_LOAD_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0]))


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
def _packed_fma_fp32(lhs, rhs, addend):
    """Apply independently rounded FP32 FMAs to adjacent SM120 register lanes."""
    return gl.inline_asm_elementwise(
        asm="""
        {
            .reg .b64 a, b, c;
            mov.b64 a, {$2, $3};
            mov.b64 b, {$4, $5};
            mov.b64 c, {$6, $7};
            fma.rn.f32x2 c, a, b, c;
            mov.b64 {$0, $1}, c;
        }
        """,
        constraints="=f,=f,f,f,f,f,f,f",
        args=[lhs, rhs, addend],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
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
    terms_per_tile: gl.constexpr = 1,
):
    positions = gl.minimum(
        batch_tile_start
        + gl.arange(0, K_BIAS_GROUP_TILES * terms_per_tile, layout) // terms_per_tile,
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
    tile_batch_shared=None,
    use_shared_tiles: gl.constexpr = False,
):
    """Compute Q @ rotated_center^T score biases for one batch of 16 routed/dense tiles."""
    elements_d: gl.constexpr = 8 if (mma_warps == 4 and head_dim == 64) else 16
    threads_d: gl.constexpr = head_dim // elements_d
    threads_m: gl.constexpr = 32 // threads_d
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, elements_d], [threads_m, threads_d], [mma_warps, 1], [1, 0]
    )
    tile_layout: gl.constexpr = gl.SliceLayout(1, load_layout)
    feature_layout: gl.constexpr = gl.SliceLayout(0, load_layout)
    terms_per_tile: gl.constexpr = key_center_shared.shape[0] // K_BIAS_GROUP_TILES
    if use_shared_tiles:
        tiles = tile_batch_shared.gather(  # pyright: ignore[reportOptionalMemberAccess]
            gl.arange(0, K_BIAS_GROUP_TILES * terms_per_tile, tile_layout) // terms_per_tile, 0
        )
    else:
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
            terms_per_tile,
        )

    kv_head_base = kv_batch_head.to(gl.int64) * sequence_tiles
    feature_offsets = gl.arange(0, head_dim, feature_layout)
    if terms_per_tile == 2:
        # Interleaved column pairs hold the two independently scaled center terms.
        terms = gl.arange(0, 2 * K_BIAS_GROUP_TILES, tile_layout) % 2
        center_rows = (kv_head_base + tiles.to(gl.int64)) * 2 + terms
        centers = gl.load(
            key_center_ptr + center_rows[:, None] * head_dim + feature_offsets[None, :]
        )
        key_center_shared.store(centers)
        gl.barrier()
        dots = mma_v2(
            query,
            key_center_shared.permute([1, 0]).load(key_layout),
            gl.zeros([query.shape[0], 2 * K_BIAS_GROUP_TILES], gl.int32, mma_layout),
        )
        dot_0, dot_1 = gl.split(gl.reshape(dots, [query.shape[0], K_BIAS_GROUP_TILES, 2]))
        dot_0 = gl.convert_layout(dot_0, mma_layout)
        dot_1 = gl.convert_layout(dot_1, mma_layout)
    else:
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

    if use_shared_tiles:
        scale_tiles = tile_batch_shared.load(  # pyright: ignore[reportOptionalMemberAccess]
            gl.SliceLayout(0, mma_layout)
        )
    else:
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
    tiles_shared=None,
    use_shared_tiles: gl.constexpr = False,
):
    """Refresh the 8-pair K-bias batch on boundaries and return this pair's biases."""
    local_pair = pair_index % _K_BIAS_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
    if local_pair == 0:
        batch_tile_start = pair_index * 2
        if use_shared_tiles:
            tile_batch_shared = tiles_shared.index(  # pyright: ignore[reportOptionalMemberAccess]
                (pair_index // _K_BIAS_GROUP_PAIRS) & 1
            )
            resolved_tiles = _resolve_batch_tiles(
                batch_tile_start,
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
        else:
            tile_batch_shared = tiles_shared
        bias_batch = _compute_k_bias_batch(
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
            skip_dense_routing,
            kv_batch_head,
            sequence_tiles,
            head_dim,
            mma_warps,
            mma_layout,
            key_layout,
            tile_batch_shared,
            use_shared_tiles,
        )
        bias_cache_0, bias_cache_1 = gl.split(
            gl.reshape(bias_batch, [query.shape[0], _K_BIAS_GROUP_PAIRS, 2])
        )
    k_bias_0, k_bias_1 = _extract_pair_k_bias(bias_cache_0, bias_cache_1, local_pair, row_layout)
    return k_bias_0, k_bias_1, bias_cache_0, bias_cache_1


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
    batch_head,
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
    packed_fp32: gl.constexpr = False,
):
    """Advance one shared Piper coordinate over two independently scaled K64 tiles."""
    block_m: gl.constexpr = query.shape[0]
    row_layout: gl.constexpr = running_max.type.layout
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
    key_scale_0 = gl.load(key_scale_ptr + batch_head * sequence_tiles + start_n_0 // _GL_BLOCK_N)
    key_scale_1 = gl.load(key_scale_ptr + batch_head * sequence_tiles + start_n_1 // _GL_BLOCK_N)
    score_scale_0 = query_scale * key_scale_0
    score_scale_1 = query_scale * key_scale_1
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    offsets_n = gl.arange(0, _GL_BLOCK_N, column_layout)
    valid_keys_0 = gl.full([_GL_BLOCK_N], True, gl.int1, column_layout)
    valid_keys_1 = valid_keys_0
    if mask_block_lengths:
        block_length_0 = gl.load(block_lengths_ptr + start_n_0 // _GL_BLOCK_N)
        block_length_1 = gl.load(block_lengths_ptr + start_n_1 // _GL_BLOCK_N)
        valid_keys_0 = offsets_n < block_length_0
        valid_keys_1 = offsets_n < block_length_1
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
        value_scale_multiplier_ptr + batch_head * sequence_tiles + start_n_0 // _GL_BLOCK_N
    ).to(gl.float32)
    value_scale_multiplier_1 = gl.load(
        value_scale_multiplier_ptr + batch_head * sequence_tiles + start_n_1 // _GL_BLOCK_N
    ).to(gl.float32)
    multiplier_bits_0 = value_scale_multiplier_0.to(gl.int32, bitcast=True)
    multiplier_bits_1 = value_scale_multiplier_1.to(gl.int32, bitcast=True)
    value_log_scale_0 = multiplier_bits_0.to(gl.float32) * (1.0 / 8388608.0) - (
        127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION
    )
    value_log_scale_1 = multiplier_bits_1.to(gl.float32) * (1.0 / 8388608.0) - (
        127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION
    )
    block_max = gl.maximum(
        score_max_0 + value_log_scale_0,
        score_max_1 + value_log_scale_1,
    )
    next_max = gl.maximum(running_max, block_max)
    old_weight = gl.exp2(running_max - next_max)
    current_weight = gl.exp2(block_max - next_max)
    delta_0 = k_bias_0 - block_max
    delta_1 = k_bias_1 - block_max
    if packed_fp32:
        shifted_0 = _packed_fma_fp32(
            integer_scores_0.to(gl.float32), score_scale_0[:, None], delta_0[:, None]
        )
        shifted_1 = _packed_fma_fp32(
            integer_scores_1.to(gl.float32), score_scale_1[:, None], delta_1[:, None]
        )
    else:
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
    packed_fp32: gl.constexpr = False,
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
    if packed_fp32:
        result = _packed_fma_fp32(
            partial.to(gl.float32), current_weight[:, None], accumulator * old_weight[:, None]
        )
    else:
        result = rescale_packed(partial, accumulator, old_weight, current_weight)
    return result


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
    head_dim: gl.constexpr = accumulator.shape[1]
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    offsets_d = gl.arange(0, head_dim, column_layout)
    output = accumulator / (gl.maximum(denominator, 1e-30) * 255.0)[:, None]
    valid_queries = global_query_block * _GL_BLOCK_N + offsets_m < logical_sequence_length
    if block_m == 128:
        valid_queries = output_start_m + offsets_m < output_sequence_length
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
        [_VALUE_MEAN_GROUP_TILES, half_dim],
        gl.NVMMASharedLayout.get_default_for([_VALUE_MEAN_GROUP_TILES, half_dim], gl.bfloat16),
    ).load(rhs_layout)
    accumulator_slice = mma_v2(lhs_high, strip_means, accumulator_slice)
    return mma_v2(lhs_low, strip_means, accumulator_slice)


@gluon.jit
def accumulate_chunk_value_means(
    accumulator,
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
    """Accumulate up to 32 buffered per-tile V means via two D64 BF16 MMA strips."""
    head_dim: gl.constexpr = accumulator.shape[1]
    half_dim: gl.constexpr = head_dim // 2
    gl.static_assert(half_dim == 64)

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
    mean_ptrs_0 = value_mean_ptr + (base + safe_tiles_0.to(gl.int64))[:, None] * head_dim
    mean_ptrs_1 = value_mean_ptr + (base + safe_tiles_1.to(gl.int64))[:, None] * head_dim
    value_mean_shared = key_center_shared.reinterpret(
        gl.bfloat16,
        [2, K_BIAS_GROUP_TILES, half_dim],
        gl.NVMMASharedLayout.get_default_for([K_BIAS_GROUP_TILES, half_dim], gl.bfloat16),
    )
    accumulator_0, accumulator_1 = gl.split(
        gl.permute(gl.reshape(accumulator, [_GL_BLOCK_N, 2, half_dim]), [0, 2, 1])
    )
    accumulator_0 = _accumulate_value_mean_strip(
        gl.convert_layout(accumulator_0, mma_layout),
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
        gl.convert_layout(accumulator_1, mma_layout),
        lhs_high,
        lhs_low,
        value_mean_shared,
        mean_ptrs_0,
        mean_ptrs_1,
        valid_0,
        valid_1,
        half_dim,
        all_valid,
        feature_layout,
        rhs_layout,
    )
    return gl.convert_layout(
        gl.reshape(
            gl.permute(gl.join(accumulator_0, accumulator_1), [0, 2, 1]), [_GL_BLOCK_N, head_dim]
        ),
        mma_layout,
    )


@gluon.jit
def accumulate_value_mean_step(
    accumulator,
    tile_masses_shared,
    pair_max_shared,
    tiles_shared,
    key_center_shared,
    running_max,
    value_mean_ptr,
    mass_0,
    mass_1,
    block_max,
    current_weight,
    kv_batch_head,
    start_n_0,
    start_n_1,
    sequence_tiles,
    pair_index,
    tile_count,
    in_loop: gl.constexpr,
    mma_layout: gl.constexpr,
    use_chunked_value_mean: gl.constexpr,
):
    """Defer D128 V-mean accumulation to 32-tile BF16 tensor-core chunks; keep D64 scalar."""
    if use_chunked_value_mean:
        local_pair = pair_index % _VALUE_MEAN_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
        _store_tile_column(tile_masses_shared, mass_0, local_pair * 2)
        _store_tile_column(tile_masses_shared, mass_1, local_pair * 2 + 1)
        _store_tile_column(pair_max_shared, block_max, local_pair)
        if in_loop:
            if local_pair == _VALUE_MEAN_GROUP_PAIRS - 1:
                next_accumulator = accumulate_chunk_value_means(
                    accumulator,
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
            else:
                next_accumulator = accumulator
        else:
            chunk_offset = (
                pair_index // _VALUE_MEAN_GROUP_PAIRS  # pyright: ignore[reportOperatorIssue]
            ) * _VALUE_MEAN_GROUP_TILES
            remaining_tiles = tile_count - chunk_offset
            next_accumulator = accumulate_chunk_value_means(
                accumulator,
                tile_masses_shared,
                pair_max_shared,
                tiles_shared,
                key_center_shared,
                running_max,
                value_mean_ptr,
                kv_batch_head,
                sequence_tiles,
                remaining_tiles,
                False,
                mma_layout,
            )
    else:
        next_accumulator = accumulate_value_mean_pair(
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
    return next_accumulator
