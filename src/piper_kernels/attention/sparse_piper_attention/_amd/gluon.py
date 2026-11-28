"""Fused RDNA4 sparse Piper over the common quantized attention contract."""

# Gluon device parameters and layouts are not Python runtime values.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels._grid import select_block_grid
from piper_kernels.attention.kernels.piper._amd import quantization as amd_quantization
from piper_kernels.attention.kernels.piper._amd.fragments import (
    ATTENTION_LLVM_FN_ATTRS,
    MMA_LAYOUT,
    accumulate_chunk_value_means,
    compute_chunk_key_biases,
    concat_columns,
    pv_tiles,
    qk_tiles,
    query_fragments,
    requires_64bit_context_offsets,
    requires_64bit_query_offsets,
    rescale_numerator,
    softmax_fragment,
    split_columns,
)
from piper_kernels.attention.kernels.piper.gluon import (
    load_tile_column,
    store_tile_column,
)
from piper_kernels.attention.kernels.sparse_piper.gluon import tile_offset
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS

from .._prepared import (
    _DO_NOT_SPECIALIZE_ARGUMENTS,
    _ContextLauncher,
    _PreparedSparsePiperContext,
    _PreparedSparsePiperOperands,
    _PreparedSparsePiperQuery,
    _ValidatedAttentionLaunch,
)

# Re-exported for offline compiler diagnostics and test verification.
_prepare_query_kernel = amd_quantization._prepare_query_kernel
_LLVM_FN_ATTRS = ATTENTION_LLVM_FN_ATTRS
_requires_64bit_query_offsets = requires_64bit_query_offsets
_requires_64bit_context_offsets = requires_64bit_context_offsets
_PAIRS_PER_CHUNK = gl.constexpr(8)
_TILES_PER_CHUNK = gl.constexpr(16)

# One FP32 record per K64 tile, shared by the table writer and attention reader.
_KEY_SCALE = gl.constexpr(0)
_LOG_MULTIPLIER = gl.constexpr(1)
_LOG_MULTIPLIER_OVER_255 = gl.constexpr(2)
_INVERSE_MULTIPLIER = gl.constexpr(3)
_PARAMETER_COUNT = gl.constexpr(4)


@gluon.jit(do_not_specialize=("count",))
def _prepare_parameters(key_scale_ptr, multiplier_ptr, table_ptr, count):
    """Fold per-K64 K/V scales into one FP32 record table without recompiling per count."""
    index = gl.program_id(0).to(gl.int64) * 256 + gl.arange(
        0, 256, gl.BlockedLayout([1], [32], [4], [0])
    )
    scale = gl.load(key_scale_ptr + index, index < count, 0)
    multiplier = gl.load(multiplier_ptr + index, index < count, 1)
    record = table_ptr + index * _PARAMETER_COUNT
    gl.store(record + _KEY_SCALE, scale, index < count)
    gl.store(record + _LOG_MULTIPLIER, gl.log2(multiplier), index < count)
    gl.store(record + _LOG_MULTIPLIER_OVER_255, gl.log2(multiplier * (1.0 / 255.0)), index < count)
    gl.store(record + _INVERSE_MULTIPLIER, 1.0 / multiplier, index < count)


def prepare_parameters(context: _PreparedSparsePiperContext) -> torch.Tensor:
    """Fold immutable K/V scales once; V already has its final physical format."""
    with device_context(context.key.device):
        parameters = torch.empty(
            (*context.key_scale.shape, _PARAMETER_COUNT.value),
            device=context.key.device,
            dtype=torch.float32,
        )
        count = context.key_scale.numel()
        _prepare_parameters[((count + 255) // 256,)](
            context.key_scale, context.value_scale_multiplier, parameters, count, num_warps=4
        )
    return parameters


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_ARGUMENTS)
def _sparse_piper_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    parameters_ptr,
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
    use_64bit_query_offsets: gl.constexpr,
    use_64bit_context_offsets: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
):
    # Keep global tensor bases in i64. Structural flags select fragment-local
    # address widths without making dynamic sequence lengths compile-time values.
    head = gl.program_id(1).to(gl.int64)
    batch = gl.program_id(2).to(gl.int64)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // query_heads_per_kv_head) + head // query_heads_per_kv_head
    local_query_block = gl.program_id(0)
    query_block = query_block_offset + local_query_block
    global_query_block = global_query_block_offset + local_query_block
    row_layout: gl.constexpr = gl.SliceLayout(2, MMA_LAYOUT)
    block_layout: gl.constexpr = gl.SliceLayout(1, row_layout)
    query_blocks = gl.full([1], query_block, gl.int32, block_layout)
    rows = gl.arange(0, 64, gl.SliceLayout(0, row_layout))
    route_head_offset = gl.load(route_head_offsets_ptr + head)
    route_base = (
        routes_ptr
        + batch * stride_rb
        + query_block.to(gl.int64) * stride_rq
        + route_head_offset * stride_rr
    )
    routed_sparse_tile_count = gl.load(head_keep_blocks_ptr + head)
    use_sparse_routes = global_query_block < sparse_query_blocks if has_dense_query_suffix else True
    selected_sparse_tile_count = gl.where(
        use_sparse_routes,
        routed_sparse_tile_count,
        sparse_key_blocks,
    )
    sequence_tiles = storage_sequence_length // 64
    dense_tile_count = sequence_tiles - sparse_key_blocks
    tile_count = selected_sparse_tile_count + dense_tile_count
    pair_count = gl.cdiv(tile_count, 2)
    query = query_fragments(
        query_ptr + batch_head * query_storage_sequence_length * head_dim,
        query_blocks,
        use_64bit_query_offsets,
        head_dim,
    )
    query_scale = gl.load(
        query_scale_ptr
        + batch_head * (query_storage_sequence_length // 32)
        + query_blocks[:, None] * 2
        + rows[None, :] // 32
    )
    numerator = gl.zeros([1, 64, head_dim], gl.float32, MMA_LAYOUT)
    denominator = gl.zeros([1, 64], gl.float32, row_layout)
    running_max = gl.full([1, 64], -float("inf"), gl.float32, row_layout)
    parameters = parameters_ptr + kv_batch_head * sequence_tiles * _PARAMETER_COUNT
    key_center_head = key_center_ptr + kv_batch_head * sequence_tiles * (2 * head_dim)
    key_center_scale_head = key_center_scale_ptr + kv_batch_head * sequence_tiles * 2
    value_mean_head = value_mean_ptr + kv_batch_head * sequence_tiles * head_dim

    matrix_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0, 1]
    )
    vector_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    # compute_chunk_key_biases fills valid bias columns before the pair loop; each
    # pair reads its two bias columns before writing its two tile-mass columns
    # within each warp's disjoint 16-row slice, cutting LDS below 7 WGs/WGP.
    tile_scratch_shared = gl.allocate_shared_memory(
        gl.float32, [64, _TILES_PER_CHUNK], matrix_shared_layout
    )
    pair_max_shared = gl.allocate_shared_memory(
        gl.float32, [64, _PAIRS_PER_CHUNK], matrix_shared_layout
    )
    tiles_shared = gl.allocate_shared_memory(gl.int32, [_TILES_PER_CHUNK], vector_shared_layout)
    tile_load_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])

    chunk_count = gl.cdiv(pair_count, _PAIRS_PER_CHUNK)
    for chunk in range(chunk_count):
        start_pair = chunk * _PAIRS_PER_CHUNK
        end_pair = gl.minimum(start_pair + _PAIRS_PER_CHUNK, pair_count)
        num_valid_tiles = gl.minimum(tile_count - start_pair * 2, _TILES_PER_CHUNK)
        positions = gl.minimum(
            start_pair * 2 + gl.arange(0, _TILES_PER_CHUNK, tile_load_layout),
            tile_count - 1,
        )
        tiles_shared.store(
            tile_offset(
                route_base,
                positions,
                routed_sparse_tile_count,
                selected_sparse_tile_count,
                sparse_key_blocks,
                stride_rr,
                use_sparse_routes,
            )
        )
        gl.barrier()
        compute_chunk_key_biases(
            query,
            key_center_head,
            key_center_scale_head,
            tiles_shared,
            tile_scratch_shared,
            query_scale,
            num_valid_tiles,
            use_64bit_context_offsets,
            head_dim,
        )
        gl.barrier()

        for pair in range(start_pair, end_pair):
            pair_in_chunk = pair - start_pair
            tile_0 = tiles_shared.gather(gl.full([1], 2 * pair_in_chunk, gl.int32, block_layout), 0)
            tile_1 = tiles_shared.gather(
                gl.full([1], 2 * pair_in_chunk + 1, gl.int32, block_layout), 0
            )
            scores = qk_tiles(
                query,
                key_ptr + kv_batch_head * storage_sequence_length * head_dim,
                tile_0,
                tile_1,
                use_64bit_context_offsets,
                head_dim,
            )
            k_bias_0 = load_tile_column(
                tile_scratch_shared, 2 * pair_in_chunk, gl.SliceLayout(0, row_layout)
            )[None, :]
            k_bias_1 = load_tile_column(
                tile_scratch_shared, 2 * pair_in_chunk + 1, gl.SliceLayout(0, row_layout)
            )[None, :]
            parameters_0 = parameters + tile_0 * _PARAMETER_COUNT
            parameters_1 = parameters + tile_1 * _PARAMETER_COUNT
            key_scale_0 = gl.load(parameters_0 + _KEY_SCALE)
            key_scale_1 = gl.load(parameters_1 + _KEY_SCALE)
            scale_0 = query_scale * key_scale_0[:, None]
            scale_1 = query_scale * key_scale_1[:, None]
            if mask_block_lengths:
                length_0 = gl.load(block_lengths_ptr + tile_0)
                length_1 = gl.load(block_lengths_ptr + tile_1)
            else:
                length_0 = gl.minimum(64, logical_sequence_length - tile_0 * 64)
                length_1 = gl.minimum(64, logical_sequence_length - tile_1 * 64)
            complete = (
                (gl.sum(length_0, 0) == 64)
                & (gl.sum(length_1, 0) == 64)
                & (pair * 2 + 1 < tile_count)
            )
            if not complete:
                # The two K64 tiles always produce 128 score columns, independently of D.
                columns = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
                column_block_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(1, MMA_LAYOUT))
                valid_length_0 = gl.convert_layout(
                    length_0, column_block_layout, assert_trivial=True
                )
                valid_length_1 = gl.convert_layout(
                    length_1, column_block_layout, assert_trivial=True
                )
                valid = gl.where(
                    columns[None, :] < 64,
                    columns[None, :] < valid_length_0[:, None],
                    (columns[None, :] - 64 < valid_length_1[:, None]) & (pair * 2 + 1 < tile_count),
                )
                column_scale_0 = gl.convert_layout(
                    key_scale_0, column_block_layout, assert_trivial=True
                )
                column_scale_1 = gl.convert_layout(
                    key_scale_1, column_block_layout, assert_trivial=True
                )
                scale = (
                    gl.where(
                        columns[None, :] < 64, column_scale_0[:, None], column_scale_1[:, None]
                    )[:, None, :]
                    * query_scale[:, :, None]
                )
                # Invalid scores are -inf. Handle a zero scale before multiplication
                # so padded keys can never turn 0 * -inf into NaN.
                scores = gl.where(
                    valid[:, None, :], gl.where(scale == 0, 0.0, scores), -float("inf")
                )
                scale_0 = gl.where(scale_0 == 0, 1.0, scale_0)
                scale_1 = gl.where(scale_1 == 0, 1.0, scale_1)
            scores_0, scores_1 = split_columns(scores)
            offset_0 = gl.load(parameters_0 + _LOG_MULTIPLIER_OVER_255)
            offset_1 = gl.load(parameters_1 + _LOG_MULTIPLIER_OVER_255)
            maximum_0 = gl.convert_layout(gl.max(scores_0, 2), row_layout)
            maximum_1 = gl.convert_layout(gl.max(scores_1, 2), row_layout)
            block_max = gl.maximum(
                gl.fma(maximum_0, scale_0, offset_0[:, None] + k_bias_0),
                gl.fma(maximum_1, scale_1, offset_1[:, None] + k_bias_1),
            )
            next_max = gl.maximum(running_max, block_max)
            old_weight = gl.exp2(running_max - next_max)
            current_weight = gl.exp2(block_max - next_max)
            numerator = rescale_numerator(numerator, old_weight)
            scores_00, scores_01 = split_columns(scores_0)
            scores_10, scores_11 = split_columns(scores_1)
            chunks = (scores_00, scores_01, scores_10, scores_11)
            inverse_0 = gl.load(parameters_0 + _INVERSE_MULTIPLIER)
            inverse_1 = gl.load(parameters_1 + _INVERSE_MULTIPLIER)
            log_0 = gl.load(parameters_0 + _LOG_MULTIPLIER)
            log_1 = gl.load(parameters_1 + _LOG_MULTIPLIER)
            packed, sums = (), ()
            for fragment in gl.static_range(4):
                words, total = softmax_fragment(
                    chunks[fragment],
                    block_max,
                    scale_0 if fragment < 2 else scale_1,
                    inverse_0 if fragment < 2 else inverse_1,
                    log_0 if fragment < 2 else log_1,
                    k_bias=k_bias_0 if fragment < 2 else k_bias_1,
                )
                packed += (words,)
                sums += (total,)
                with gl.amd.warp_pipeline_stage("softmax", priority=0):
                    pass
            mass_0 = sums[0] + sums[1]
            mass_1 = sums[2] + sums[3]
            denominator = denominator * old_weight + (mass_0 + mass_1) * current_weight
            probabilities = concat_columns(
                concat_columns(packed[0], packed[1]), concat_columns(packed[2], packed[3])
            )
            store_tile_column(tile_scratch_shared, mass_0, 2 * pair_in_chunk)
            store_tile_column(tile_scratch_shared, mass_1, 2 * pair_in_chunk + 1)
            store_tile_column(pair_max_shared, block_max, pair_in_chunk)
            numerator = pv_tiles(
                probabilities,
                value_ptr + kv_batch_head * storage_sequence_length * head_dim,
                tile_0,
                tile_1,
                numerator,
                current_weight,
                use_64bit_context_offsets,
            )
            running_max = next_max

        gl.barrier()
        numerator = accumulate_chunk_value_means(
            numerator,
            tile_scratch_shared,
            pair_max_shared,
            tiles_shared,
            running_max,
            value_mean_head,
            num_valid_tiles,
            use_64bit_context_offsets,
            head_dim,
        )
        gl.barrier()

    features = gl.arange(0, head_dim, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
    inverse_denominator = 1.0 / (gl.maximum(denominator, 1e-30) * 255.0)
    result = numerator * inverse_denominator[:, :, None]
    output_rows = (local_query_block * 64 + rows).to(gl.int64)
    valid_rows = (global_query_block * 64 + rows < logical_sequence_length) | mask_block_lengths
    if apply_coarse_residual:
        coarse = gl.load(
            coarse_output_ptr
            + batch * stride_cb
            + head * stride_ch
            + query_block.to(gl.int64) * stride_cq
            + features
        )
        gate = gl.load(
            coarse_gate_ptr
            + batch * stride_gb
            + head * stride_gh
            + output_rows[None, :, None] * stride_gn
            + features[None, None, :],
            mask=valid_rows[None, :, None],
            other=0.0,
        ).to(gl.float32)
        result = gl.fma(gate, coarse[None, None, :], result)
    gl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + output_rows[None, :, None] * stride_on
        + features[None, None, :],
        result.to(output_ptr.dtype.element_ty),
        mask=valid_rows[None, :, None],
    )


@gluon.jit
def _store_key_centers(
    center,
    key_center_ptr,
    key_center_scale_ptr,
    batch_head_64,
    storage_tiles_64,
    tile_index_64,
    head_base,
    heads,
    head_dim: gl.constexpr,
    mask_heads: gl.constexpr,
):
    """Quantize and store two-stage residual INT8 key centers for D64 or D128."""
    if head_dim == 64:
        center_heads = gl.reshape(center, [2, 64])
        center_scale_0 = amd_quantization.compute_int8_scale(gl.max(gl.abs(center_heads), axis=1))
        center_codes_0 = amd_quantization.round_to_int8(center_heads / center_scale_0[:, None])
        residual = center_heads - center_codes_0.to(gl.float32) * center_scale_0[:, None]
        center_scale_1 = amd_quantization.compute_int8_scale(gl.max(gl.abs(residual), axis=1))
        center_codes_1 = amd_quantization.round_to_int8(residual / center_scale_1[:, None])
        center_layout: gl.constexpr = center_codes_0.type.layout
        center_head_offsets = gl.arange(0, 2, gl.SliceLayout(1, center_layout))
        feature_offsets = gl.arange(0, 64, gl.SliceLayout(0, center_layout))
        center_offsets = (
            (batch_head_64 + center_head_offsets[:, None].to(gl.int64)) * storage_tiles_64
            + tile_index_64
        ) * 128 + feature_offsets[None, :]
        scale_head_offsets = gl.arange(0, 2, center_scale_0.type.layout)
        center_scale_offsets = (
            (batch_head_64 + scale_head_offsets.to(gl.int64)) * storage_tiles_64 + tile_index_64
        ) * 2
        if mask_heads:
            center_mask_2d = ((head_base + center_head_offsets) < heads)[:, None]
            center_mask_1d = (head_base + scale_head_offsets) < heads
            gl.store(key_center_ptr + center_offsets, center_codes_0, mask=center_mask_2d)
            gl.store(key_center_ptr + center_offsets + 64, center_codes_1, mask=center_mask_2d)
            gl.store(
                key_center_scale_ptr + center_scale_offsets, center_scale_0, mask=center_mask_1d
            )
            gl.store(
                key_center_scale_ptr + center_scale_offsets + 1,
                center_scale_1,
                mask=center_mask_1d,
            )
        else:
            gl.store(key_center_ptr + center_offsets, center_codes_0)
            gl.store(key_center_ptr + center_offsets + 64, center_codes_1)
            gl.store(key_center_scale_ptr + center_scale_offsets, center_scale_0)
            gl.store(key_center_scale_ptr + center_scale_offsets + 1, center_scale_1)
        return
    center_scale_0 = amd_quantization.compute_int8_scale(gl.max(gl.abs(center), axis=0))
    center_codes_0 = amd_quantization.round_to_int8(center / center_scale_0)
    residual = center - center_codes_0.to(gl.float32) * center_scale_0
    center_scale_1 = amd_quantization.compute_int8_scale(gl.max(gl.abs(residual), axis=0))
    center_codes_1 = amd_quantization.round_to_int8(residual / center_scale_1)
    feature_offsets = gl.arange(0, 128, center_codes_0.type.layout)
    center_offsets = (batch_head_64 * storage_tiles_64 + tile_index_64) * 256 + feature_offsets
    gl.store(key_center_ptr + center_offsets, center_codes_0)
    gl.store(key_center_ptr + center_offsets + 128, center_codes_1)
    center_scale_offsets = (batch_head_64 * storage_tiles_64 + tile_index_64) * 2
    gl.store(key_center_scale_ptr + center_scale_offsets, center_scale_0)
    gl.store(key_center_scale_ptr + center_scale_offsets + 1, center_scale_1)


@gluon.jit(do_not_specialize=(*amd_quantization.PREP_COMMON_DO_NOT_SPECIALIZE, "sparse_key_blocks"))
def _prepare_key_kernel(
    key_ptr,
    key_int8_ptr,
    key_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    storage_tiles,
    sparse_key_blocks,
    stride_b,
    stride_h,
    stride_n,
    heads,
    heads_inner,
    head_dim: gl.constexpr,
    emit_summaries: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    mask_heads: gl.constexpr,
    aligned_16b: gl.constexpr,
):
    """Quantize K64-centered keys in Gluon with optional min/max routing summaries."""
    batch_index, tile_index, head_base, batch_head_index = amd_quantization.resolve_vec_tile_ids(
        heads, heads_inner, head_dim
    )
    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    tile_index_64 = tile_index.to(gl.int64)

    values, row_valid = amd_quantization.load_tile_vec(
        key_ptr,
        batch_index,
        head_base,
        tile_index,
        stride_b,
        stride_h,
        stride_n,
        block_lengths_ptr,
        logical_sequence_length,
        heads,
        head_dim,
        mask_block_lengths,
        mask_rows,
        mask_heads,
        aligned_16b,
    )

    if emit_summaries and tile_index < sparse_key_blocks:
        col_max, col_min = amd_quantization.reduce_column_minmax_vec(
            values,
            row_valid,
            mask_rows=mask_rows,
            broadcast_to_tile=False,
        )
        amd_quantization.store_summary_vec(
            key_summary_ptr,
            col_max,
            batch_head_index,
            tile_index,
            sparse_key_blocks,
            head_base,
            heads,
            head_dim=head_dim,
            mask_heads=mask_heads,
        )
        amd_quantization.store_summary_vec(
            key_aux_ptr,
            col_min,
            batch_head_index,
            tile_index,
            sparse_key_blocks,
            head_base,
            heads,
            head_dim=head_dim,
            mask_heads=mask_heads,
        )

    rotated = amd_quantization.apply_signed_hadamard_vec(values, head_dim)
    if mask_rows:
        rotated = gl.where(row_valid[:, None], rotated, 0.0)

    rotated_max, rotated_min = amd_quantization.reduce_column_minmax_vec(
        rotated,
        row_valid,
        mask_rows=mask_rows,
        broadcast_to_tile=True,
    )
    if mask_rows:
        center = gl.where(rotated_max >= rotated_min, 0.5 * (rotated_max + rotated_min), 0.0)
    else:
        center = 0.5 * (rotated_max + rotated_min)
    center = gl.convert_layout(center, gl.SliceLayout(0, rotated.type.layout))

    _store_key_centers(
        center,
        key_center_ptr,
        key_center_scale_ptr,
        batch_head_64,
        storage_tiles_64,
        tile_index_64,
        head_base,
        heads,
        head_dim,
        mask_heads,
    )

    centered = rotated - center[None, :]
    if mask_rows:
        centered = gl.where(row_valid[:, None], centered, 0.0)

    if head_dim == 64:
        centered_heads = gl.reshape(centered, [64, 2, 64])
        key_scale = amd_quantization.compute_int8_scale(
            gl.max(gl.max(gl.abs(centered_heads), axis=2), axis=0)
        )
        key_codes = gl.reshape(
            amd_quantization.round_to_int8(centered_heads / key_scale[None, :, None]), [64, 128]
        )
        amd_quantization.store_head_major_vec(
            key_int8_ptr,
            key_codes,
            batch_head_index,
            tile_index,
            storage_tiles,
            head_base,
            heads,
            head_dim=head_dim,
            mask_heads=mask_heads,
        )
        scale_head_offsets = gl.arange(0, 2, key_scale.type.layout)
        key_scale_offsets = (
            batch_head_64 + scale_head_offsets.to(gl.int64)
        ) * storage_tiles_64 + tile_index_64
        if mask_heads:
            gl.store(
                key_scale_ptr + key_scale_offsets,
                key_scale,
                mask=(head_base + scale_head_offsets) < heads,
            )
        else:
            gl.store(key_scale_ptr + key_scale_offsets, key_scale)
    else:
        key_scale = amd_quantization.compute_int8_scale(
            gl.max(gl.max(gl.abs(centered), axis=1), axis=0)
        )
        key_codes = amd_quantization.round_to_int8(centered / key_scale)
        amd_quantization.store_head_major_vec(
            key_int8_ptr,
            key_codes,
            batch_head_index,
            tile_index,
            storage_tiles,
            head_base,
            heads,
            head_dim=head_dim,
            mask_heads=False,
        )
        gl.store(key_scale_ptr + batch_head_64 * storage_tiles_64 + tile_index_64, key_scale)


@gluon.jit(do_not_specialize=amd_quantization.PREP_COMMON_DO_NOT_SPECIALIZE)
def _prepare_value_kernel(
    value_ptr,
    value_int8_ptr,
    value_scale_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    storage_tiles,
    stride_b,
    stride_h,
    stride_n,
    heads,
    heads_inner,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    mask_heads: gl.constexpr,
    aligned_16b: gl.constexpr,
):
    """Center values per K64 tile and store directly in packed ``ValueFormat.AMD_WMMA`` layout."""
    batch_index, tile_index, head_base, batch_head_index = amd_quantization.resolve_vec_tile_ids(
        heads, heads_inner, head_dim
    )
    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    tile_index_64 = tile_index.to(gl.int64)

    values, row_valid = amd_quantization.load_tile_vec(
        value_ptr,
        batch_index,
        head_base,
        tile_index,
        stride_b,
        stride_h,
        stride_n,
        block_lengths_ptr,
        logical_sequence_length,
        heads,
        head_dim,
        mask_block_lengths,
        mask_rows,
        mask_heads,
        aligned_16b,
    )
    if mask_rows:
        valid_count = gl.maximum(gl.sum(row_valid.to(gl.int32), axis=0), 1).to(gl.float32)
        mean_bf16 = amd_quantization.reduce_column_mean_vec(values, valid_count).to(gl.bfloat16)
    else:
        mean_bf16 = amd_quantization.reduce_column_mean_vec(
            values, amd_quantization.GL_TILE_ROWS_F32
        ).to(gl.bfloat16)

    amd_quantization.store_summary_vec(
        value_mean_ptr,
        mean_bf16,
        batch_head_index,
        tile_index,
        storage_tiles,
        head_base,
        heads,
        head_dim=head_dim,
        mask_heads=mask_heads,
    )

    mean_f32 = gl.convert_layout(mean_bf16.to(gl.float32), gl.SliceLayout(0, values.type.layout))
    centered = values - mean_f32[None, :]
    if mask_rows:
        centered = gl.where(row_valid[:, None], centered, 0.0)

    if head_dim == 64:
        centered_heads = gl.reshape(centered, [64, 2, 64])
        residual_scale = amd_quantization.compute_int8_scale(
            gl.max(gl.max(gl.abs(centered_heads), axis=2), axis=0)
        )
        mean_magnitude = gl.convert_layout(
            gl.max(gl.abs(gl.reshape(mean_f32, [2, 64])), axis=1), residual_scale.type.layout
        )
        value_scale = gl.maximum(
            residual_scale, mean_magnitude * amd_quantization.GL_VALUE_MEAN_SCALE_FLOOR
        )
        value_codes = gl.reshape(
            amd_quantization.round_to_int8(centered_heads / value_scale[None, :, None]), [64, 128]
        )
        scale_head_offsets = gl.arange(0, 2, value_scale.type.layout)
        value_scale_offsets = (
            batch_head_64 + scale_head_offsets.to(gl.int64)
        ) * storage_tiles_64 + tile_index_64
        if mask_heads:
            gl.store(
                value_scale_ptr + value_scale_offsets,
                value_scale * amd_quantization.GL_P_UINT8_RANGE,
                mask=(head_base + scale_head_offsets) < heads,
            )
        else:
            gl.store(
                value_scale_ptr + value_scale_offsets,
                value_scale * amd_quantization.GL_P_UINT8_RANGE,
            )
    else:
        residual_scale = amd_quantization.compute_int8_scale(
            gl.max(gl.max(gl.abs(centered), axis=1), axis=0)
        )
        mean_magnitude = gl.max(gl.abs(mean_f32), axis=0)
        value_scale = gl.maximum(
            residual_scale, mean_magnitude * amd_quantization.GL_VALUE_MEAN_SCALE_FLOOR
        )
        value_codes = amd_quantization.round_to_int8(centered / value_scale)
        gl.store(
            value_scale_ptr + batch_head_64 * storage_tiles_64 + tile_index_64,
            value_scale * amd_quantization.GL_P_UINT8_RANGE,
        )

    # Fold AMD-WMMA token bit3<->bit4 swizzle into the transpose so stores vectorize cleanly.
    value_store_layout: gl.constexpr = amd_quantization.VALUE_STORE_LAYOUT
    value_swizzled = gl.reshape(
        gl.permute(gl.reshape(value_codes, [2, 2, 2, 8, 128]), [4, 0, 2, 1, 3]),
        [128, 64],
    )
    value_transposed = gl.convert_layout(value_swizzled, value_store_layout)
    feature_offsets = gl.arange(0, 128, gl.SliceLayout(1, value_store_layout))
    token_offsets = gl.arange(0, 64, gl.SliceLayout(0, value_store_layout))
    if head_dim == 64:
        local_head = feature_offsets // 64
        value_offsets = (
            (
                (
                    (batch_head_64 + local_head[:, None].to(gl.int64)) * storage_tiles_64
                    + tile_index_64
                )
                * 64
                + (feature_offsets % 64)[:, None]
            )
            * amd_quantization.GL_TILE_ROWS
        ) + token_offsets[None, :]
        if mask_heads:
            gl.store(
                value_int8_ptr + value_offsets,
                value_transposed,
                mask=((head_base + local_head) < heads)[:, None],
            )
        else:
            gl.store(value_int8_ptr + value_offsets, value_transposed)
    else:
        value_offsets = (
            (batch_head_64 * storage_tiles_64 + tile_index_64) * 128 + feature_offsets[:, None]
        ) * amd_quantization.GL_TILE_ROWS + token_offsets[None, :]
        gl.store(value_int8_ptr + value_offsets, value_transposed)


def launch_operand_preparation(
    query: torch.Tensor,
    combined_key: torch.Tensor,
    combined_value: torch.Tensor,
    scale: float,
    operands: _PreparedSparsePiperOperands,
) -> None:
    """Populate preallocated sparse-Piper operands using vectorized 8-warp RDNA4 Gluon kernels."""
    _batch, query_heads, logical_sequence_length, head_dim = query.shape
    kv_heads = combined_key.shape[1]
    storage_sequence_length = operands.key.shape[2]
    storage_tiles = storage_sequence_length // TILE_ROWS
    if storage_tiles == 0:
        return
    block_lengths = operands.block_lengths
    emit_summaries = operands.query_summary is not None
    mask_block_lengths = block_lengths is not None
    mask_rows = (logical_sequence_length != storage_sequence_length) or mask_block_lengths
    heads_per_program = 128 // head_dim
    lengths_buffer = block_lengths if block_lengths is not None else operands.key_scale

    with device_context(combined_key.device):
        if query_heads > 0:
            query_grid, query_heads_inner = select_block_grid(
                query, storage_tiles, heads_per_program=heads_per_program
            )
            _prepare_query_kernel[query_grid](
                query,
                operands.query,
                operands.query_scale,
                (
                    operands.query_summary
                    if operands.query_summary is not None
                    else operands.query_scale
                ),
                lengths_buffer,
                logical_sequence_length,
                storage_tiles,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query_heads,
                query_heads_inner,
                float(scale),
                head_dim,
                emit_summaries,
                mask_block_lengths,
                mask_rows,
                head_dim == 64 and (query_heads % 2 != 0),
                amd_quantization.is_aligned_16b(query),
                num_warps=8,
                num_stages=1,
                llvm_fn_attrs=amd_quantization.PREP_LLVM_FN_ATTRS,
            )
        if kv_heads > 0:
            mask_kv_heads = head_dim == 64 and (kv_heads % 2 != 0)
            key_grid, key_heads_inner = select_block_grid(
                combined_key, storage_tiles, heads_per_program=heads_per_program
            )
            _prepare_key_kernel[key_grid](
                combined_key,
                operands.key,
                operands.key_scale,
                operands.key_center,
                operands.key_center_scale,
                operands.key_summary if operands.key_summary is not None else operands.key_scale,
                operands.key_aux if operands.key_aux is not None else operands.key_scale,
                lengths_buffer,
                logical_sequence_length,
                storage_tiles,
                operands.sparse_key_blocks,
                combined_key.stride(0),
                combined_key.stride(1),
                combined_key.stride(2),
                kv_heads,
                key_heads_inner,
                head_dim,
                emit_summaries,
                mask_block_lengths,
                mask_rows,
                mask_kv_heads,
                amd_quantization.is_aligned_16b(combined_key),
                num_warps=8,
                num_stages=1,
                llvm_fn_attrs=amd_quantization.PREP_LLVM_FN_ATTRS,
            )
            value_grid, value_heads_inner = select_block_grid(
                combined_value, storage_tiles, heads_per_program=heads_per_program
            )
            _prepare_value_kernel[value_grid](
                combined_value,
                operands.value,
                operands.value_scale_multiplier,
                operands.value_mean,
                lengths_buffer,
                logical_sequence_length,
                storage_tiles,
                combined_value.stride(0),
                combined_value.stride(1),
                combined_value.stride(2),
                kv_heads,
                value_heads_inner,
                head_dim,
                mask_block_lengths,
                mask_rows,
                mask_kv_heads,
                amd_quantization.is_aligned_16b(combined_value),
                num_warps=8,
                num_stages=1,
                llvm_fn_attrs=amd_quantization.PREP_LLVM_FN_ATTRS,
            )


def bind_context(context: _PreparedSparsePiperContext) -> _ContextLauncher:
    """Own final K/V storage and folded scales for any number of local Q launches."""
    if context.routes_per_query == 0:
        raise ValueError("AMD sparse Piper does not implement skip_dense_routing")
    parameters = prepare_parameters(context)

    def _launch(
        bound_context: _PreparedSparsePiperContext,
        query_state: _PreparedSparsePiperQuery,
        output: torch.Tensor,
        launch: _ValidatedAttentionLaunch,
    ) -> None:
        use_64bit_query_offsets = _requires_64bit_query_offsets(
            launch.query_storage_sequence_length,
            launch.head_dim,
        )
        use_64bit_context_offsets = _requires_64bit_context_offsets(
            launch.storage_sequence_length,
            launch.head_dim,
        )
        with device_context(output.device):
            grid = (launch.query_block_count, launch.heads, launch.batch)
            _sparse_piper_attention_kernel[grid](
                query_state.data,
                bound_context.key,
                bound_context.value,
                query_state.scale,
                bound_context.key_center,
                bound_context.key_center_scale,
                parameters,
                bound_context.value_mean,
                launch.coarse_output,
                launch.coarse_gate,
                launch.block_lengths,
                query_state.routes,
                bound_context.head_keep_blocks,
                bound_context.route_head_offsets,
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
                launch.heads // bound_context.key.shape[1],
                launch.head_dim,
                use_64bit_query_offsets,
                use_64bit_context_offsets,
                launch.mask_block_lengths,
                launch.has_dense_query_suffix,
                launch.apply_coarse_residual,
                num_warps=4,
                num_stages=1,
                llvm_fn_attrs=_LLVM_FN_ATTRS,
            )

    return _ContextLauncher(context, _launch)
