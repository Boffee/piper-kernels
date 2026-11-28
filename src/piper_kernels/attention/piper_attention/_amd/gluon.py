"""RDNA4 dense Piper: per-token V scaling, K1024+V1024 centering, and Q64/K64 recurrence.

Architecture and codegen behavior:
- Uses four wave32 warps and Q64/K64 tiles with Q32/K64 scale groups. Supports
  causal attention (traversing longest query prefixes first), rectangular
  non-causal attention, ragged tails, outer strides, and GQA/MQA. V preparation
  writes initialized padding directly into packed WMMA storage, avoiding a
  full-size transpose/repacking allocation.
- Softmax processes 16-column fragments, combining score scaling with FP32 maxima
  and exponential arguments while warp-pipeline stages prioritize QK/PV matrix
  work over softmax. Probability rounding uses the logical K64 tile's maximum and
  each token's V scale with round-to-nearest-even packing; the epilogue multiplies
  by the per-row reciprocal of ``max(denominator, 1e-30) * 255.0``. Interior tiles
  omit elementwise masks; key tails and causal diagonals retain them.
- Launches in WGP mode (``-cumode``) with ``amd_compiler_workarounds.LDS_ONLY_FENCE_ATTRS``
  so workgroup fences synchronize without invalidating shared L0/L1 caches across
  CUs. On affected Triton 3.9 builds, ``amd_compiler_workarounds`` also enables
  scalar ``v_fma_f32`` score FMAs (avoiding LLVM SLP vectorization register bloat)
  and unpinned warp-pipeline loads (preserving K/V load hoisting and pairing).
- ``gfx1201`` has hardware coverage on RX 9070 XT; ``gfx1200`` has offline
  compilation coverage only.
"""

# Gluon device types and layouts are not Python runtime values.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportAttributeAccessIssue=false

import math

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.piper._amd import quantization as amd_quantization
from piper_kernels.attention.kernels.piper._amd.fragments import (
    MMA_LAYOUT,
    concat_columns,
    four_fragments,
    pack_probabilities,
    pv_tiles,
    qk_fp32_center_bias,
    qk_tiles,
    query_fragments,
    rescale_numerator,
    scalar_fma,
)
from piper_kernels.attention.kernels.piper.gluon import (
    advance_tile_max,
    advance_tile_max_safe,
    max_propagate_nan,
)

from .. import _quantization
from .._validation import (
    PrefillKernelOperands,
    PreparedAttention,
    PreparedContext,
    PreparedQuery,
    WindowMode,
    normalize_window_bounds,
    resolve_query_window,
    resolve_window_mode,
    validate_lse_buffer,
    validate_output_buffer,
    validate_query_offset,
)

# Launch attributes shared by every dense RDNA4 kernel compile.
_LLVM_FN_ATTRS = (
    ("target-features", "-cumode"),
    *amd_compiler_workarounds.UNPINNED_WARP_PIPELINE_ATTRS,
    *amd_compiler_workarounds.LDS_ONLY_FENCE_ATTRS,
)
_GL_TILES_PER_CHUNK = gl.constexpr(_quantization._TILES_PER_CHUNK)
_GL_LN2 = gl.constexpr(math.log(2.0))
_GL_LOG2_SCALE_EPSILON = gl.constexpr(math.log2(1e-7))
# Retained for compiler/provider diagnostics.
_prepare_query_kernel = amd_quantization._prepare_query_kernel
_prepare_value_kernel = _quantization._quantize_value_k1024_kernel


@gluon.jit
def _unfused_mul(a, b):
    """Multiply FP32 values via opaque ``v_mul_f32`` so LLVM cannot contract into FMA."""
    return gl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        constraints="=v,v,v",
        args=(a, b),
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _softmax_fragment(scores, maximum, scale, multiplier, row_active=None):
    """Pack one 16-column fragment while retaining unrounded denominator mass."""
    row_layout: gl.constexpr = maximum.type.layout
    maximum = gl.convert_layout(maximum, gl.SliceLayout(2, scores.type.layout))
    multiplier = gl.convert_layout(multiplier, gl.SliceLayout(1, scores.type.layout))
    if scale is None:
        probabilities = gl.exp2(scores - maximum[:, :, None])
    else:
        scale = gl.convert_layout(scale, maximum.type.layout)
        probabilities = gl.exp2(gl.fma(scores, scale[:, :, None], -maximum[:, :, None]))
    total = gl.convert_layout(gl.sum(probabilities, 2), row_layout)
    scaled_probs = probabilities * multiplier[:, None, :]
    if row_active is not None:
        active = gl.convert_layout(row_active, maximum.type.layout)[:, :, None]
        scaled_probs = gl.where(active, scaled_probs, 0.0)
    return pack_probabilities(scaled_probs), total


@gluon.jit
def _softmax_tile(scores, block_max, scale, multiplier, row_active=None):
    """Keep K64 normalization, but consume FP32 probabilities in 16-column fragments."""
    fragments = four_fragments(scores)
    multiplier_0, multiplier_1 = gl.split(multiplier.reshape([1, 2, 32]).permute([0, 2, 1]))
    multiplier_00, multiplier_01 = gl.split(multiplier_0.reshape([1, 2, 16]).permute([0, 2, 1]))
    multiplier_10, multiplier_11 = gl.split(multiplier_1.reshape([1, 2, 16]).permute([0, 2, 1]))
    multipliers = (multiplier_00, multiplier_01, multiplier_10, multiplier_11)
    packed_fragments, sums = (), ()
    for fragment in gl.static_range(4):
        words, total = _softmax_fragment(
            fragments[fragment], block_max, scale, multipliers[fragment], row_active
        )
        packed_fragments += (words,)
        sums += (total,)
        with gl.amd.warp_pipeline_stage("softmax", priority=0):
            pass
    packed = concat_columns(
        concat_columns(packed_fragments[0], packed_fragments[1]),
        concat_columns(packed_fragments[2], packed_fragments[3]),
    )
    return packed, (sums[0] + sums[1]) + (sums[2] + sums[3])


@gluon.jit
def _safe_softmax_pv_step(
    scores,
    block_max,
    tile_k_bias,
    multiplier,
    value_base_ptr,
    key_blocks,
    running_max,
    running_bias,
    denominator,
    chunk_denom,
    numerator,
    wide_context_offsets: gl.constexpr,
):
    """Run safe online softmax, probability packing, and PV update for one K64 tile."""
    next_max, next_bias, old_weight, current_weight, has_block, safe_block_max = (
        advance_tile_max_safe(running_max, running_bias, block_max, tile_k_bias)
    )
    packed, total = _softmax_tile(scores, safe_block_max, None, multiplier, has_block)
    row_sum = gl.where(has_block, total * current_weight, 0.0)
    denominator = denominator * old_weight + row_sum
    chunk_denom = chunk_denom * old_weight + row_sum
    numerator = pv_tiles(
        packed,
        value_base_ptr,
        key_blocks,
        key_blocks,
        rescale_numerator(numerator, old_weight),
        current_weight,
        wide_context_offsets,
        two_tiles=False,
    )
    return next_max, next_bias, denominator, chunk_denom, numerator


@gluon.jit
def _fold_safe_value_mean(numerator, chunk_denom, value_mean, chunk_has_block=None):
    """Fold non-empty K1024 value means into the FP32 numerator."""
    active = (chunk_denom > 0.0) if chunk_has_block is None else chunk_has_block
    return numerator + gl.where(
        active[:, :, None],
        (chunk_denom * 255.0)[:, :, None] * value_mean[:, None, :],
        0.0,
    )


@gluon.jit
def _normalize_safe_output(numerator, denominator, running_max):
    """Normalize FP32 numerator with empty-row guards and return output/mask/denom."""
    has_any = running_max != -float("inf")
    safe_denom = gl.where(has_any & (denominator != 0.0), denominator, 1.0)
    inverse_denominator = 1.0 / (safe_denom * 255.0)
    output = gl.where(has_any[:, :, None], numerator * inverse_denominator[:, :, None], 0.0)
    return output, has_any, safe_denom


@gluon.jit
def _finalize_safe_output(numerator, denominator, running_max, running_bias):
    """Normalize FP32 numerator and natural-log LSE with empty-row guards."""
    output, has_any, safe_denom = _normalize_safe_output(numerator, denominator, running_max)
    lse = gl.where(
        has_any,
        ((running_max + running_bias) + gl.log2(safe_denom)) * _GL_LN2,
        -float("inf"),
    )
    return output, lse


@gluon.jit(
    do_not_specialize=[
        "query_start",
        "query_rows",
        "global_query_start",
        "key_length",
        "query_storage",
        "key_storage",
        "heads",
        "key_chunks",
        "stride_ob",
        "stride_oh",
        "stride_om",
        "stride_lb",
        "stride_lh",
        "window_left",
        "window_right",
    ]
)
def _dense_piper_kernel(  # noqa: PLR0912, PLR0915
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    key_center_ptr,
    multiplier_ptr,
    log_scale_ptr,
    value_mean_ptr,
    output_ptr,
    query_start,
    query_rows,
    global_query_start,
    key_length,
    query_storage,
    key_storage,
    heads,
    key_chunks,
    stride_ob,
    stride_oh,
    stride_om,
    query_heads_per_kv_head: gl.constexpr,
    head_dim: gl.constexpr,
    window_mode: gl.constexpr,
    wide_query_offsets: gl.constexpr,
    wide_context_offsets: gl.constexpr,
    lse_ptr=None,
    key_lengths_ptr=None,
    stride_lb=0,
    stride_lh=0,
    store_lse: gl.constexpr = False,
    has_key_lengths: gl.constexpr = False,
    offset_causal_from_key_length: gl.constexpr = False,
    window_left=-1,
    window_right=-1,
):
    is_causal_mode: gl.constexpr = window_mode == "causal"
    is_windowed_mode: gl.constexpr = window_mode == "windowed"
    safe_softmax: gl.constexpr = store_lse or has_key_lengths or is_windowed_mode
    head = gl.program_id(1).to(gl.int64)
    batch = gl.program_id(2).to(gl.int64)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // query_heads_per_kv_head) + head // query_heads_per_kv_head
    if has_key_lengths:
        key_length = gl.minimum(gl.maximum(gl.load(key_lengths_ptr + batch), 0), key_length)
    if offset_causal_from_key_length and window_mode != "none":
        global_query_start = key_length - query_rows
    # Launch causal query blocks longest-prefix-first to prevent tail-wave imbalance.
    query_block = (
        (gl.num_programs(0) - 1 - gl.program_id(0)) if is_causal_mode else gl.program_id(0)
    )
    stored_query_block = query_start // 64 + query_block
    row_layout: gl.constexpr = gl.SliceLayout(2, MMA_LAYOUT)
    block_layout: gl.constexpr = gl.SliceLayout(1, row_layout)
    column_layout: gl.constexpr = gl.SliceLayout(1, MMA_LAYOUT)
    blocks = gl.full([1], stored_query_block, gl.int32, block_layout)
    rows = gl.arange(0, 64, gl.SliceLayout(0, row_layout))
    columns = gl.arange(0, 64, gl.SliceLayout(0, column_layout))
    features = gl.arange(0, head_dim, gl.SliceLayout(0, column_layout))
    output_rows = (
        query_block.to(gl.int64) * 64 + rows if wide_query_offsets else query_block * 64 + rows
    )
    global_query_rows = (
        global_query_start.to(gl.int64) + output_rows
        if wide_context_offsets
        else global_query_start + query_block * 64 + rows
    )
    query = query_fragments(
        query_ptr + batch_head * query_storage * head_dim,
        blocks,
        wide_query_offsets,
        head_dim,
    )
    query_scale = gl.load(
        query_scale_ptr
        + batch_head * (query_storage // 32)
        + stored_query_block * 2
        + rows[None, :] // 32,
    )
    numerator = gl.zeros([1, 64, head_dim], gl.float32, MMA_LAYOUT)
    denominator = gl.zeros([1, 64], gl.float32, row_layout)
    running_max = gl.full([1, 64], -float("inf"), gl.float32, row_layout)
    running_bias = gl.zeros([1, 64], gl.float32, row_layout)
    first_tile = 0
    tile_count = gl.cdiv(key_length, 64)
    first_causal_tile = 0
    global_query_block = 0
    boundary_tile_lo = 0
    boundary_tile_hi = tile_count
    eff_wl = gl.zeros((), dtype=gl.int64)
    eff_wr = gl.zeros((), dtype=gl.int64)
    if is_causal_mode:
        global_row_start = global_query_start.to(gl.int64) + query_block.to(gl.int64) * 64
        if safe_softmax:
            causal_end = gl.maximum(global_row_start + 64, 0)
            tile_count = gl.minimum(tile_count, gl.cdiv(causal_end, 64).to(gl.int32))
            first_causal_tile = (gl.maximum(global_row_start + 1, 0) // 64).to(gl.int32)
        else:
            global_query_block = global_query_start // 64 + query_block
            tile_count = gl.minimum(tile_count, global_query_block + 1)
    elif is_windowed_mode:
        eff_wl = gl.minimum(
            window_left.to(gl.int64),
            gl.maximum(0, global_query_start.to(gl.int64) + query_rows.to(gl.int64)),
        )
        eff_wr = gl.minimum(
            window_right.to(gl.int64), key_length.to(gl.int64) + query_rows.to(gl.int64)
        )
        global_row_start = global_query_start.to(gl.int64) + query_block.to(gl.int64) * 64
        global_row_end = (
            gl.minimum(
                global_row_start + 64, global_query_start.to(gl.int64) + query_rows.to(gl.int64)
            )
            - 1
        )
        vis_k_start = (
            gl.maximum(0, global_row_start - eff_wl)
            if window_left >= 0
            else gl.zeros((), dtype=gl.int64)
        )
        vis_k_end = (
            gl.minimum(key_length.to(gl.int64), global_row_end + eff_wr + 1)
            if window_right >= 0
            else key_length.to(gl.int64)
        )
        if vis_k_start >= vis_k_end:
            first_tile = 0
            tile_count = 0
        else:
            first_tile = (vis_k_start // 64).to(gl.int32)
            tile_count = gl.minimum(tile_count, gl.cdiv(vis_k_end, 64).to(gl.int32))
        boundary_tile_lo = (
            gl.cdiv(gl.maximum(0, global_row_end - eff_wl), 64).to(gl.int32)
            if window_left >= 0
            else 0
        )
        boundary_tile_hi = (
            gl.minimum(tile_count, ((global_row_start + eff_wr) // 64).to(gl.int32))
            if window_right >= 0
            else tile_count
        )
    first_chunk = first_tile // _GL_TILES_PER_CHUNK
    chunk_count = gl.cdiv(tile_count, _GL_TILES_PER_CHUNK)
    for chunk in range(first_chunk, chunk_count):
        k_bias = qk_fp32_center_bias(
            query_ptr + batch_head * query_storage * head_dim,
            stored_query_block,
            key_center_ptr + (kv_batch_head * key_chunks + chunk) * head_dim,
            query_scale,
            rows,
            wide_query_offsets,
            head_dim,
            row_layout,
        )
        chunk_denom = gl.zeros([1, 64], gl.float32, row_layout)
        chunk_tile_base = chunk * _GL_TILES_PER_CHUNK
        tile_start = gl.maximum(chunk_tile_base, first_tile)
        tile_stop = gl.minimum(chunk_tile_base + _GL_TILES_PER_CHUNK, tile_count)
        for tile in range(tile_start, tile_stop):
            key_blocks = gl.full([1], tile, gl.int32, block_layout)
            scores = qk_tiles(
                query,
                key_ptr + kv_batch_head * key_storage * head_dim,
                key_blocks,
                key_blocks,
                wide_context_offsets,
                head_dim,
                two_tiles=False,
            )
            scale = query_scale * gl.load(
                key_scale_ptr + kv_batch_head * (key_storage // 64) + tile
            )
            key_rows = (
                tile.to(gl.int64) * 64 + columns if wide_context_offsets else tile * 64 + columns
            )
            boundary = (tile + 1) * 64 > key_length
            if is_causal_mode:
                if safe_softmax:
                    boundary = boundary | (tile >= first_causal_tile)
                else:
                    boundary = boundary | (tile == global_query_block)
            elif is_windowed_mode:
                boundary = boundary | (tile < boundary_tile_lo) | (tile >= boundary_tile_hi)
            if boundary:
                valid = (key_rows[None, None, :] < key_length) & (
                    global_query_rows[None, :, None] == global_query_rows[None, :, None]
                )
                if is_causal_mode:
                    valid = valid & (key_rows[None, None, :] <= global_query_rows[None, :, None])
                elif is_windowed_mode:
                    if window_left >= 0:
                        valid = valid & (
                            key_rows[None, None, :].to(gl.int64)
                            >= (global_query_rows[None, :, None].to(gl.int64) - eff_wl)
                        )
                    if window_right >= 0:
                        valid = valid & (
                            key_rows[None, None, :].to(gl.int64)
                            <= (global_query_rows[None, :, None].to(gl.int64) + eff_wr)
                        )
                # Fused scaling must not turn a masked zero-scale score into 0 * -inf.
                scores = gl.where(
                    valid, gl.where(scale[:, :, None] == 0, 0.0, scores), -float("inf")
                )
                scale = gl.where(scale == 0, 1.0, scale)
            multiplier = gl.load(multiplier_ptr + kv_batch_head * key_storage + key_rows[None, :])
            log_scale = gl.load(log_scale_ptr + kv_batch_head * key_storage + key_rows[None, :])
            if safe_softmax:
                # Pre-scale FP32 scores via unfused v_mul_f32 so LLVM cannot contract
                # scores * scale - maximum back into FMA and mismatch block_max rounding.
                scores = _unfused_mul(scores, scale[:, :, None])
                shifted_scores = gl.where(
                    scores != -float("inf"), scores + log_scale[:, None, :], -float("inf")
                )
                block_max = gl.reduce(
                    shifted_scores,
                    2,
                    max_propagate_nan,
                )
                if is_causal_mode:
                    tile_start_n = tile.to(gl.int64) * 64 if wide_context_offsets else tile * 64
                    tile_has_keys = tile_start_n <= global_query_rows[None, :]
                    block_max = gl.where(tile_has_keys, block_max, -float("inf"))
                    tile_k_bias = gl.where(tile_has_keys, k_bias, 0.0)
                elif is_windowed_mode:
                    tile_k_bias = gl.where(block_max != -float("inf"), k_bias, 0.0)
                else:
                    tile_k_bias = k_bias
                running_max, running_bias, denominator, chunk_denom, numerator = (
                    _safe_softmax_pv_step(
                        scores,
                        block_max,
                        tile_k_bias,
                        multiplier,
                        value_ptr + kv_batch_head * key_storage * head_dim,
                        key_blocks,
                        running_max,
                        running_bias,
                        denominator,
                        chunk_denom,
                        numerator,
                        wide_context_offsets,
                    )
                )
            else:
                block_max = gl.max(
                    scalar_fma(scores, scale[:, :, None], log_scale[:, None, :]),
                    2,
                )
                next_max, next_bias, old_weight, current_weight = advance_tile_max(
                    running_max,
                    running_bias,
                    block_max,
                    k_bias,
                )
                packed, total = _softmax_tile(scores, block_max, scale, multiplier)
                row_sum = total * current_weight
                denominator = denominator * old_weight + row_sum
                chunk_denom = chunk_denom * old_weight + row_sum
                numerator = pv_tiles(
                    packed,
                    value_ptr + kv_batch_head * key_storage * head_dim,
                    key_blocks,
                    key_blocks,
                    rescale_numerator(numerator, old_weight),
                    current_weight,
                    wide_context_offsets,
                    two_tiles=False,
                )
                running_max = next_max
                running_bias = next_bias
        value_mean = gl.load(
            value_mean_ptr + (kv_batch_head * key_chunks + chunk) * head_dim + features[None, :]
        )
        if safe_softmax:
            numerator = _fold_safe_value_mean(numerator, chunk_denom, value_mean)
        else:
            numerator += (chunk_denom * 255.0)[:, :, None] * value_mean[:, None, :]
    if safe_softmax:
        output, lse = _finalize_safe_output(numerator, denominator, running_max, running_bias)
        if store_lse:
            gl.store(
                lse_ptr + batch * stride_lb + head * stride_lh + output_rows[None, :].to(gl.int64),
                lse,
                output_rows[None, :] < query_rows,
            )
    else:
        inverse_denominator = 1.0 / (gl.maximum(denominator, 1e-30) * 255.0)
        output = numerator * inverse_denominator[:, :, None]
    offsets = (
        batch * stride_ob + head * stride_oh + output_rows[None, :, None].to(gl.int64) * stride_om
    )
    gl.store(
        output_ptr + offsets + features[None, None, :],
        output,
        output_rows[None, :, None] < query_rows,
    )


@gluon.jit(
    do_not_specialize=[
        "seq_q",
        "query_storage",
        "heads",
        "num_slots",
        "max_chunks",
        "stride_ki_s",
        "stride_ks_s",
        "stride_kc_s",
        "stride_vi_s",
        "stride_vs_s",
        "stride_vm_s",
        "stride_cs_b",
        "stride_cs_c",
        "stride_ob",
        "stride_oh",
        "stride_om",
        "stride_lb",
        "stride_lh",
        "split_chunks",
        "total_splits",
        "tail_cap",
        "window_left",
        "window_right",
        "logical_seq_q",
    ]
)
def _frozen_pool_piper_kernel(  # noqa: PLR0912, PLR0915
    query_ptr,
    query_scale_ptr,
    key_int8_ptr,
    key_scale_ptr,
    key_center_ptr,
    value_int8_ptr,
    value_scale_ptr,
    value_mean_ptr,
    chunk_to_slot_ptr,
    num_chunks_ptr,
    output_ptr,
    lse_ptr,
    seq_q,
    query_storage,
    heads,
    num_slots,
    max_chunks,
    stride_ki_s,
    stride_ks_s,
    stride_kc_s,
    stride_vi_s,
    stride_vs_s,
    stride_vm_s,
    stride_cs_b,
    stride_cs_c,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    split_chunks,
    partial_bias_ptr,
    partial_log_denom_ptr,
    total_splits,
    query_heads_per_kv_head: gl.constexpr,
    head_dim: gl.constexpr,
    has_chunk_map: gl.constexpr,
    wide_query_offsets: gl.constexpr,
    has_chunk_splits: gl.constexpr = False,
    per_row_query_scale: gl.constexpr = False,
    tail_len_ptr=None,
    tail_cap=0,
    window_left=-1,
    window_right=-1,
    logical_seq_q=0,
    has_tail: gl.constexpr = False,
    window_mode: gl.constexpr = "none",
):
    """Prefill Q blocks directly against finalized 1024-token slots in PiperKVChunkPool."""
    is_causal_mode: gl.constexpr = window_mode == "causal"
    is_windowed_mode: gl.constexpr = window_mode == "windowed"
    has_window: gl.constexpr = is_causal_mode or is_windowed_mode

    head = gl.program_id(1).to(gl.int64)
    if has_chunk_splits:
        batch_split = gl.program_id(2)
        split_id = batch_split % split_chunks
        batch = (batch_split // split_chunks).to(gl.int64)
        out_batch64 = batch * total_splits.to(gl.int64) + split_id.to(gl.int64)
    else:
        batch = gl.program_id(2).to(gl.int64)
        out_batch64 = batch
        split_id = 0
    batch_head = batch * heads + head
    kv_head = head // query_heads_per_kv_head
    query_block = gl.program_id(0)

    row_layout: gl.constexpr = gl.SliceLayout(2, MMA_LAYOUT)
    block_layout: gl.constexpr = gl.SliceLayout(1, row_layout)
    column_layout: gl.constexpr = gl.SliceLayout(1, MMA_LAYOUT)
    blocks = gl.full([1], query_block, gl.int32, block_layout)
    rows = gl.arange(0, 64, gl.SliceLayout(0, row_layout))
    columns = gl.arange(0, 64, gl.SliceLayout(0, column_layout))
    features = gl.arange(0, head_dim, gl.SliceLayout(0, column_layout))
    output_rows = query_block.to(gl.int64) * 64 + rows
    eff_logical_seq_q = gl.maximum(logical_seq_q, 1)
    eff_seq_q = eff_logical_seq_q if per_row_query_scale else seq_q
    row_q_idx = (output_rows % eff_logical_seq_q) if per_row_query_scale else output_rows

    query = query_fragments(
        query_ptr + batch_head * query_storage * head_dim,
        blocks,
        wide_query_offsets,
        head_dim,
    )
    if per_row_query_scale:
        query_scale = gl.load(query_scale_ptr + batch_head * query_storage + output_rows[None, :])
    else:
        query_scale = gl.load(
            query_scale_ptr
            + batch_head * (query_storage // 32)
            + query_block * 2
            + rows[None, :] // 32
        )
    numerator = gl.zeros([1, 64, head_dim], gl.float32, MMA_LAYOUT)
    denominator = gl.zeros([1, 64], gl.float32, row_layout)
    running_max = gl.full([1, 64], -float("inf"), gl.float32, row_layout)
    running_bias = gl.zeros([1, 64], gl.float32, row_layout)

    active_chunks = gl.minimum(gl.maximum(gl.load(num_chunks_ptr + batch), 0), max_chunks)
    if has_window:
        active_tail = (
            gl.minimum(gl.maximum(gl.load(tail_len_ptr + batch), 0), tail_cap).to(gl.int64)
            if has_tail
            else gl.zeros((), dtype=gl.int64)
        )
        frozen_tokens = active_chunks.to(gl.int64) * 1024
        total_kv = frozen_tokens + active_tail
        eff_wl = gl.minimum(window_left.to(gl.int64), total_kv)
        eff_wr = gl.minimum(window_right.to(gl.int64), total_kv + eff_seq_q.to(gl.int64))
        global_query_start = total_kv - eff_seq_q.to(gl.int64)
        if per_row_query_scale:
            q_tile_first = query_block.to(gl.int64) * 64
            q_tile_last = gl.minimum(q_tile_first + 64, seq_q.to(gl.int64)) - 1
            same_head = (q_tile_first // eff_logical_seq_q) == (q_tile_last // eff_logical_seq_q)
            q_idx_min = gl.where(same_head, q_tile_first % eff_logical_seq_q, 0)
            q_idx_max = gl.where(same_head, q_tile_last % eff_logical_seq_q, eff_logical_seq_q - 1)
        else:
            q_idx_min = query_block.to(gl.int64) * 64
            q_idx_max = gl.minimum(q_idx_min + 64, seq_q.to(gl.int64)) - 1
        q_row_min = global_query_start + q_idx_min
        q_row_max = global_query_start + q_idx_max
        k_min = (
            gl.maximum(0, q_row_min - eff_wl)
            if (is_windowed_mode and window_left >= 0)
            else gl.zeros((), dtype=gl.int64)
        )
        if is_causal_mode:
            k_max_excl = gl.minimum(frozen_tokens, gl.maximum(0, q_row_max + 1))
        elif window_right >= 0:
            k_max_excl = gl.minimum(frozen_tokens, gl.maximum(0, q_row_max + eff_wr + 1))
        else:
            k_max_excl = frozen_tokens
        total_tiles = active_chunks * _GL_TILES_PER_CHUNK
        vis_t_start = gl.minimum(total_tiles.to(gl.int64), k_min // 64).to(gl.int32)
        vis_t_end = gl.maximum(
            vis_t_start,
            gl.minimum(total_tiles.to(gl.int64), gl.cdiv(k_max_excl, 64)).to(gl.int32),
        )
        win_lo_base = (
            global_query_start - eff_wl
            if (is_windowed_mode and window_left >= 0)
            else -eff_seq_q.to(gl.int64) - 1
        )
        win_hi_base = (
            global_query_start + eff_wr
            if (is_windowed_mode and window_right >= 0)
            else (global_query_start if is_causal_mode else frozen_tokens + 1024)
        )
    else:
        vis_t_start = gl.zeros((), dtype=gl.int32)
        vis_t_end = active_chunks * _GL_TILES_PER_CHUNK
        win_lo_base = gl.zeros((), dtype=gl.int64)
        win_hi_base = gl.zeros((), dtype=gl.int64)

    if has_chunk_splits:
        vis_tiles = vis_t_end - vis_t_start
        tiles_per_split = (vis_tiles + split_chunks - 1) // split_chunks
        tile_start = gl.minimum(vis_t_end, vis_t_start + split_id * tiles_per_split)
        tile_end = gl.minimum(vis_t_end, tile_start + tiles_per_split)
    else:
        tile_start = vis_t_start
        tile_end = vis_t_end
    has_work = tile_start < tile_end
    chunk_start = gl.where(has_work, tile_start // _GL_TILES_PER_CHUNK, 0)
    chunk_end = gl.where(has_work, (tile_end - 1) // _GL_TILES_PER_CHUNK + 1, 0)
    first_tile = gl.where(has_work, tile_start % _GL_TILES_PER_CHUNK, 0)
    last_tile = gl.where(has_work, (tile_end - 1) % _GL_TILES_PER_CHUNK + 1, 0)

    kv_stride_ki: gl.constexpr = 1024 * head_dim
    kv_stride_ks: gl.constexpr = 16
    kv_stride_vs: gl.constexpr = 1024
    smem_log = gl.allocate_shared_memory(gl.float32, [1024], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    smem_vs = gl.allocate_shared_memory(gl.float32, [1024], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    for chunk in range(chunk_start, chunk_end):
        if has_chunk_map:
            slot = gl.load(
                chunk_to_slot_ptr + batch * stride_cs_b + chunk.to(gl.int64) * stride_cs_c
            )
        else:
            slot = chunk
        if (slot >= 0) & (slot < num_slots):
            slot64 = slot.to(gl.int64)
            k_slot_ptr = key_int8_ptr + slot64 * stride_ki_s + kv_head * kv_stride_ki
            ks_slot_ptr = key_scale_ptr + slot64 * stride_ks_s + kv_head * kv_stride_ks
            kc_slot_ptr = key_center_ptr + slot64 * stride_kc_s + kv_head * head_dim
            v_slot_ptr = value_int8_ptr + slot64 * stride_vi_s + kv_head * kv_stride_ki
            vs_slot_ptr = value_scale_ptr + slot64 * stride_vs_s + kv_head * kv_stride_vs
            vm_slot_ptr = value_mean_ptr + slot64 * stride_vm_s + kv_head * head_dim

            lin_layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
            chunk_vs_lin = gl.load(vs_slot_ptr + gl.arange(0, 1024, lin_layout))
            smem_log.store(
                gl.where(chunk_vs_lin == 0.0, _GL_LOG2_SCALE_EPSILON, gl.log2(chunk_vs_lin))
            )
            smem_vs.store(chunk_vs_lin * 255.0)

            k_bias = qk_fp32_center_bias(
                query_ptr + batch_head * query_storage * head_dim,
                query_block,
                kc_slot_ptr,
                query_scale,
                rows,
                wide_query_offsets,
                head_dim,
                row_layout,
            )
            chunk_active = k_bias != -float("inf")
            chunk_denom = gl.zeros([1, 64], gl.float32, row_layout)
            if has_chunk_splits or has_window:
                t_begin = gl.where(chunk == chunk_start, first_tile, 0)
                t_end = gl.where(chunk == chunk_end - 1, last_tile, _GL_TILES_PER_CHUNK)
            else:
                t_begin = 0
                t_end = _GL_TILES_PER_CHUNK
            if has_window:
                chunk_base64 = chunk.to(gl.int64) * 1024
                chunk_win_lo = gl.maximum(
                    0,
                    gl.minimum(1024, (win_lo_base + row_q_idx) - chunk_base64),
                ).to(gl.int32)
                chunk_win_hi = gl.maximum(
                    -1,
                    gl.minimum(1024, (win_hi_base + row_q_idx) - chunk_base64),
                ).to(gl.int32)
            else:
                chunk_win_lo = gl.zeros([64], gl.int32, gl.SliceLayout(0, row_layout))
                chunk_win_hi = gl.full([64], 1024, gl.int32, gl.SliceLayout(0, row_layout))
            for tile in range(t_begin, t_end):
                key_blocks = gl.full([1], tile, gl.int32, block_layout)
                scores = qk_tiles(
                    query,
                    k_slot_ptr,
                    key_blocks,
                    key_blocks,
                    False,
                    head_dim,
                    two_tiles=False,
                )
                scale = query_scale * gl.load(ks_slot_ptr + tile)
                vs_idx = tile * 64 + columns
                multiplier = smem_vs.gather(vs_idx, 0)[None, :]
                log_scale = smem_log.gather(vs_idx, 0)[None, :]
                scaled_scores = gl.where(
                    scale[:, :, None] == 0, 0.0, _unfused_mul(scores, scale[:, :, None])
                )
                if has_window:
                    key_rows_local = tile * 64 + columns
                    valid = chunk_active[:, :, None] & (
                        columns[None, None, :] == columns[None, None, :]
                    )
                    if is_causal_mode:
                        valid = valid & (
                            key_rows_local[None, None, :] <= chunk_win_hi[None, :, None]
                        )
                    else:
                        valid = valid & (
                            (key_rows_local[None, None, :] >= chunk_win_lo[None, :, None])
                            & (key_rows_local[None, None, :] <= chunk_win_hi[None, :, None])
                        )
                    scores = gl.where(valid, scaled_scores, -float("inf"))
                else:
                    scores = gl.where(chunk_active[:, :, None], scaled_scores, -float("inf"))
                shifted_scores = gl.where(
                    scores != -float("inf"), scores + log_scale[:, None, :], -float("inf")
                )
                block_max = gl.reduce(shifted_scores, 2, max_propagate_nan)
                tile_k_bias = (
                    gl.where(block_max != -float("inf"), k_bias, 0.0) if has_window else k_bias
                )
                running_max, running_bias, denominator, chunk_denom, numerator = (
                    _safe_softmax_pv_step(
                        scores,
                        block_max,
                        tile_k_bias,
                        multiplier,
                        v_slot_ptr,
                        key_blocks,
                        running_max,
                        running_bias,
                        denominator,
                        chunk_denom,
                        numerator,
                        False,
                    )
                )
            value_mean = gl.load(vm_slot_ptr + features[None, :])
            numerator = _fold_safe_value_mean(
                numerator, chunk_denom, value_mean, None if has_window else chunk_active
            )

    valid_out = output_rows[None, :] < seq_q
    offsets = out_batch64 * stride_ob + head * stride_oh + output_rows[None, :, None] * stride_om
    lse_offsets = out_batch64 * stride_lb + head * stride_lh + output_rows[None, :]
    if has_chunk_splits:
        output, has_any, safe_denom = _normalize_safe_output(numerator, denominator, running_max)
        gl.store(output_ptr + offsets + features[None, None, :], output, valid_out[:, :, None])
        gl.store(lse_ptr + lse_offsets, gl.where(has_any, running_max, -float("inf")), valid_out)
        gl.store(partial_bias_ptr + lse_offsets, gl.where(has_any, running_bias, 0.0), valid_out)
        gl.store(
            partial_log_denom_ptr + lse_offsets,
            gl.where(has_any, gl.log2(safe_denom), 0.0),
            valid_out,
        )
    else:
        output, lse = _finalize_safe_output(numerator, denominator, running_max, running_bias)
        gl.store(output_ptr + offsets + features[None, None, :], output, valid_out[:, :, None])
        gl.store(lse_ptr + lse_offsets, lse, valid_out)


def prepare_value(
    value: torch.Tensor,
    *,
    storage_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center V per 1024-token chunk, quantize per token, and pack RDNA4 WMMA tiles."""
    return _quantization.prepare_chunk_centered_value(
        value,
        storage_key_length=storage_length,
        store_log_scale=True,
        scale_storage_length=storage_length,
        pack_wmma=True,
        log_scale_dtype=torch.float32,
    )


def prepare_context(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    window_left: int | None = None,
    window_right: int | None = None,
) -> PreparedContext:
    """Prepare validated K/V once, independently of Q and output storage."""
    eff_left, eff_right, _ = normalize_window_bounds(is_causal, window_left, window_right)
    key_storage = triton.cdiv(key.shape[2], 64) * 64
    with device_context(key.device):
        key_int8, key_scale, key_center = _quantization.prepare_chunk_centered_key(
            key,
            grouped=True,
            storage_key_length=key_storage,
        )
        packed_value, multiplier, log_scale, value_mean = prepare_value(
            value,
            storage_length=key_storage,
        )
    return PreparedContext(
        key=key_int8,
        value=packed_value,
        key_scale=key_scale,
        key_center=key_center,
        value_scale_multiplier=multiplier,
        value_log_scale=log_scale,
        value_mean=value_mean,
        key_length=key.shape[2],
        is_causal=is_causal,
        window_left=eff_left,
        window_right=eff_right,
    )


def prepare_query(
    query: torch.Tensor,
    scale: float,
    *,
    global_row_offset: int = 0,
) -> PreparedQuery:
    """Prepare Q storage at an aligned global origin for full or ranged launches."""
    validate_query_offset(global_row_offset, block_rows=64, name="global_row_offset")
    batch, heads, query_length, head_dim = query.shape
    query_storage = triton.cdiv(query_length, 64) * 64
    query_int8, query_scale = amd_quantization.prepare_query(
        query,
        scale,
        storage_sequence_length=query_storage,
    )
    return PreparedQuery(
        data=query_int8,
        scale=query_scale,
        shape=(batch, heads, query_length, head_dim),
        dtype=query.dtype,
        global_row_offset=global_row_offset,
    )


def prepare_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    window_left: int | None = None,
    window_right: int | None = None,
) -> PreparedAttention:
    """Prepare operands and allocate the reusable output before timed launches."""
    context = prepare_context(
        key,
        value,
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
    )
    prepared_query = prepare_query(query, scale)
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    return PreparedAttention(context=context, query=prepared_query, output=output)


def _launch_window_into(
    context: PreparedContext,
    query: PreparedQuery,
    output: torch.Tensor,
    *,
    query_start: int,
    rows: int,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, heads, _, head_dim = query.shape
    query_storage = query.data.shape[2]
    key_storage = context.key.shape[2]
    key_chunks = context.key_center.shape[2]
    eff_left, eff_right, window_mode = normalize_window_bounds(
        context.is_causal, context.window_left, context.window_right
    )
    with device_context(query.data.device):
        _dense_piper_kernel[(triton.cdiv(rows, 64), heads, batch)](
            query.data,
            context.key,
            context.value,
            query.scale,
            context.key_scale,
            context.key_center,
            context.value_scale_multiplier,
            context.value_log_scale,
            context.value_mean,
            output,
            query_start,
            rows,
            query.global_row_offset + query_start,
            context.key_length,
            query_storage,
            key_storage,
            heads,
            key_chunks,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            heads // context.key.shape[1],
            head_dim,
            window_mode,
            query_storage * (head_dim // 8) > (1 << 31),
            key_storage * head_dim > (1 << 32),
            lse_ptr=lse,
            stride_lb=0 if lse is None else lse.stride(0),
            stride_lh=0 if lse is None else lse.stride(1),
            store_lse=lse is not None,
            window_left=-1 if eff_left is None else eff_left,
            window_right=-1 if eff_right is None else eff_right,
            num_warps=4,
            num_stages=1,
            llvm_fn_attrs=_LLVM_FN_ATTRS,
        )
    return output


def launch_attention_into(
    context: PreparedContext,
    query: PreparedQuery,
    output: torch.Tensor,
    *,
    query_start: int = 0,
    query_rows: int | None = None,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill a caller-owned output for an aligned window of prepared Q."""
    batch, heads, query_length, head_dim = query.shape
    rows = resolve_query_window(
        query_length,
        query_start=query_start,
        query_rows=query_rows,
        global_row_offset=query.global_row_offset,
        block_rows=64,
        key_length=context.key_length,
        is_causal=context.is_causal,
    )
    validate_output_buffer(
        output,
        shape=(batch, heads, rows, head_dim),
        dtype=query.dtype,
        device=query.data.device,
    )
    if lse is not None:
        validate_lse_buffer(
            lse,
            shape=(batch, heads, rows),
            device=query.data.device,
        )
    if batch == 0:
        return output
    return _launch_window_into(
        context,
        query,
        output,
        query_start=query_start,
        rows=rows,
        lse=lse,
    )


def launch_attention(prepared: PreparedAttention) -> torch.Tensor:
    """Reuse the prepared output without allocating or preparing operands."""
    return launch_attention_into(prepared.context, prepared.query, prepared.output)


def run_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    window_left: int | None = None,
    window_right: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if query.shape[0] == 0:
        empty_out = torch.empty_like(query, memory_format=torch.contiguous_format)
        if not return_lse:
            return empty_out
        return empty_out, torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
    prepared = prepare_attention(
        query,
        key,
        value,
        scale,
        is_causal,
        window_left=window_left,
        window_right=window_right,
    )
    lse = (
        torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        if return_lse
        else None
    )
    output = launch_attention_into(prepared.context, prepared.query, prepared.output, lse=lse)
    if return_lse:
        assert lse is not None
        return output, lse
    return output


def launch_frozen_pool_kernel(
    operands: PrefillKernelOperands,
    *,
    batch: int,
    gqa_group: int,
    head_dim: int,
    has_chunk_map: bool,
    split_chunks: int = 1,
    total_splits: int | None = None,
    has_chunk_splits: bool | None = None,
    per_row_query_scale: bool = False,
    partial_bias: torch.Tensor | None = None,
    partial_log_denom: torch.Tensor | None = None,
    tail_len: torch.Tensor | None = None,
    tail_cap: int = 0,
    window_left: int = -1,
    window_right: int = -1,
    logical_seq_q: int = 0,
    window_mode: WindowMode = "none",
) -> None:
    """Launch RDNA4 frozen-pool prefill kernel with typed pre-assembled operands."""
    resolved_total = split_chunks if total_splits is None else total_splits
    resolved_chunk_splits = (resolved_total > 1) if has_chunk_splits is None else has_chunk_splits
    if resolved_chunk_splits and (partial_bias is None or partial_log_denom is None):
        raise ValueError("partial_bias and partial_log_denom are required when has_chunk_splits")
    bias_target = partial_bias if partial_bias is not None else operands.lse
    log_denom_target = partial_log_denom if partial_log_denom is not None else operands.lse
    with device_context(operands.query_int8.device):
        _frozen_pool_piper_kernel[
            (triton.cdiv(operands.seq_q, 64), operands.q_heads, batch * split_chunks)
        ](
            *operands.kernel_args(),
            split_chunks,
            bias_target,
            log_denom_target,
            resolved_total,
            gqa_group,
            head_dim,
            has_chunk_map,
            operands.query_storage * (head_dim // 8) > (1 << 31),
            has_chunk_splits=resolved_chunk_splits,
            per_row_query_scale=per_row_query_scale,
            tail_len_ptr=tail_len,
            tail_cap=tail_cap,
            window_left=window_left,
            window_right=window_right,
            logical_seq_q=logical_seq_q,
            has_tail=tail_len is not None,
            window_mode=window_mode,
            num_warps=4,
            num_stages=1,
            llvm_fn_attrs=_LLVM_FN_ATTRS,
        )


def launch_tail_prefill(
    query_int8: torch.Tensor,
    query_scale: torch.Tensor,
    tail_key_int8: torch.Tensor,
    tail_key_scale: torch.Tensor,
    tail_key_center: torch.Tensor,
    tail_value_int8: torch.Tensor,
    tail_multiplier: torch.Tensor,
    tail_log_scale: torch.Tensor,
    tail_value_mean: torch.Tensor,
    tail_len: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    seq_q: int,
    tail_cap: int,
    is_causal: bool,
    window_mode: WindowMode | None = None,
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill grouped queries over quantized live tail with per-batch tail_len & offset window."""
    resolved_mode = resolve_window_mode(is_causal, window_mode)
    batch, q_heads, _, head_dim = output.shape
    query_storage = query_int8.shape[2]
    key_storage = tail_key_int8.shape[2]
    key_chunks = tail_key_center.shape[2]
    with device_context(query_int8.device):
        _dense_piper_kernel[(triton.cdiv(seq_q, 64), q_heads, batch)](
            query_int8,
            tail_key_int8,
            tail_value_int8,
            query_scale,
            tail_key_scale,
            tail_key_center,
            tail_multiplier,
            tail_log_scale,
            tail_value_mean,
            output,
            0,
            seq_q,
            0,
            tail_cap,
            query_storage,
            key_storage,
            q_heads,
            key_chunks,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            q_heads // tail_key_int8.shape[1],
            head_dim,
            resolved_mode,
            query_storage * (head_dim // 8) > (1 << 31),
            key_storage * head_dim > (1 << 32),
            lse_ptr=lse,
            key_lengths_ptr=tail_len,
            stride_lb=lse.stride(0),
            stride_lh=lse.stride(1),
            store_lse=True,
            has_key_lengths=True,
            offset_causal_from_key_length=resolved_mode != "none",
            window_left=window_left,
            window_right=window_right,
            num_warps=4,
            num_stages=1,
            llvm_fn_attrs=_LLVM_FN_ATTRS,
        )
    return output, lse
