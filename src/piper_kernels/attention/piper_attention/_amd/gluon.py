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
    PreparedAttention,
    PreparedContext,
    PreparedQuery,
    resolve_query_window,
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
def _fold_safe_value_mean(numerator, chunk_denom, value_mean):
    """Fold non-empty K1024 value means into the FP32 numerator."""
    return numerator + gl.where(
        (chunk_denom > 0.0)[:, :, None],
        (chunk_denom * 255.0)[:, :, None] * value_mean[:, None, :],
        0.0,
    )


@gluon.jit
def _finalize_safe_output(numerator, denominator, running_max, running_bias):
    """Normalize FP32 numerator and natural-log LSE with empty-row guards."""
    has_any = running_max != -float("inf")
    safe_denom = gl.where(has_any & (denominator != 0.0), denominator, 1.0)
    inverse_denominator = 1.0 / (safe_denom * 255.0)
    output = gl.where(has_any[:, :, None], numerator * inverse_denominator[:, :, None], 0.0)
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
    is_causal: gl.constexpr,
    wide_query_offsets: gl.constexpr,
    wide_context_offsets: gl.constexpr,
    lse_ptr=None,
    key_lengths_ptr=None,
    stride_lb=0,
    stride_lh=0,
    store_lse: gl.constexpr = False,
    has_key_lengths: gl.constexpr = False,
    offset_causal_from_key_length: gl.constexpr = False,
):
    safe_softmax: gl.constexpr = store_lse or has_key_lengths
    head = gl.program_id(1).to(gl.int64)
    batch = gl.program_id(2).to(gl.int64)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // query_heads_per_kv_head) + head // query_heads_per_kv_head
    if has_key_lengths:
        key_length = gl.minimum(gl.maximum(gl.load(key_lengths_ptr + batch), 0), key_length)
    if offset_causal_from_key_length and is_causal:
        global_query_start = key_length - query_rows
    # Launch causal query blocks longest-prefix-first to prevent tail-wave imbalance.
    query_block = (gl.num_programs(0) - 1 - gl.program_id(0)) if is_causal else gl.program_id(0)
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
    tile_count = gl.cdiv(key_length, 64)
    first_causal_tile = 0
    global_query_block = 0
    if is_causal:
        if safe_softmax:
            global_row_start = global_query_start.to(gl.int64) + query_block.to(gl.int64) * 64
            causal_end = gl.maximum(global_row_start + 64, 0)
            tile_count = gl.minimum(tile_count, gl.cdiv(causal_end, 64).to(gl.int32))
            first_causal_tile = (gl.maximum(global_row_start + 1, 0) // 64).to(gl.int32)
        else:
            global_query_block = global_query_start // 64 + query_block
            tile_count = gl.minimum(tile_count, global_query_block + 1)
    chunk_count = gl.cdiv(tile_count, _GL_TILES_PER_CHUNK)
    for chunk in range(chunk_count):
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
        tile_start = chunk * _GL_TILES_PER_CHUNK
        tile_stop = gl.minimum(tile_start + _GL_TILES_PER_CHUNK, tile_count)
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
            if is_causal:
                if safe_softmax:
                    boundary = boundary | (tile >= first_causal_tile)
                else:
                    boundary = boundary | (tile == global_query_block)
            if boundary:
                valid = key_rows[None, None, :] < key_length
                if is_causal:
                    valid = valid & (key_rows[None, None, :] <= global_query_rows[None, :, None])
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
                if is_causal:
                    tile_start_n = tile.to(gl.int64) * 64 if wide_context_offsets else tile * 64
                    tile_has_keys = tile_start_n <= global_query_rows[None, :]
                    block_max = gl.where(tile_has_keys, block_max, -float("inf"))
                    tile_k_bias = gl.where(tile_has_keys, k_bias, 0.0)
                else:
                    tile_k_bias = k_bias
                next_max, next_bias, old_weight, current_weight, has_block, safe_block_max = (
                    advance_tile_max_safe(
                        running_max,
                        running_bias,
                        block_max,
                        tile_k_bias,
                    )
                )
                packed, total = _softmax_tile(scores, safe_block_max, None, multiplier, has_block)
                row_sum = gl.where(has_block, total * current_weight, 0.0)
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
    ]
)
def _frozen_pool_piper_kernel(  # noqa: PLR0915
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
    query_heads_per_kv_head: gl.constexpr,
    head_dim: gl.constexpr,
    has_chunk_map: gl.constexpr,
    wide_query_offsets: gl.constexpr,
):
    """Prefill Q blocks directly against finalized 1024-token slots in PiperKVChunkPool."""
    head = gl.program_id(1).to(gl.int64)
    batch = gl.program_id(2).to(gl.int64)
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

    query = query_fragments(
        query_ptr + batch_head * query_storage * head_dim,
        blocks,
        wide_query_offsets,
        head_dim,
    )
    query_scale = gl.load(
        query_scale_ptr + batch_head * (query_storage // 32) + query_block * 2 + rows[None, :] // 32
    )
    numerator = gl.zeros([1, 64, head_dim], gl.float32, MMA_LAYOUT)
    denominator = gl.zeros([1, 64], gl.float32, row_layout)
    running_max = gl.full([1, 64], -float("inf"), gl.float32, row_layout)
    running_bias = gl.zeros([1, 64], gl.float32, row_layout)

    active_chunks = gl.minimum(gl.maximum(gl.load(num_chunks_ptr + batch), 0), max_chunks)
    kv_stride_ki: gl.constexpr = 1024 * head_dim
    kv_stride_ks: gl.constexpr = 16
    kv_stride_vs: gl.constexpr = 1024
    for chunk in range(active_chunks):
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
            for tile in range(_GL_TILES_PER_CHUNK):
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
                scores = gl.where(
                    chunk_active[:, :, None],
                    gl.where(scale[:, :, None] == 0, 0.0, _unfused_mul(scores, scale[:, :, None])),
                    -float("inf"),
                )
                raw_vs = gl.load(vs_slot_ptr + tile * 64 + columns[None, :])
                multiplier = raw_vs * 255.0
                log_scale = gl.where(raw_vs == 0.0, _GL_LOG2_SCALE_EPSILON, gl.log2(raw_vs))
                shifted_scores = gl.where(
                    scores != -float("inf"), scores + log_scale[:, None, :], -float("inf")
                )
                block_max = gl.where(
                    chunk_active,
                    gl.reduce(shifted_scores, 2, max_propagate_nan),
                    -float("inf"),
                )
                next_max, next_bias, old_weight, current_weight, has_block, safe_block_max = (
                    advance_tile_max_safe(
                        running_max,
                        running_bias,
                        block_max,
                        k_bias,
                    )
                )
                packed, total = _softmax_tile(scores, safe_block_max, None, multiplier, has_block)
                row_sum = gl.where(has_block, total * current_weight, 0.0)
                denominator = denominator * old_weight + row_sum
                chunk_denom = chunk_denom * old_weight + row_sum
                numerator = pv_tiles(
                    packed,
                    v_slot_ptr,
                    key_blocks,
                    key_blocks,
                    rescale_numerator(numerator, old_weight),
                    current_weight,
                    False,
                    two_tiles=False,
                )
                running_max = next_max
                running_bias = next_bias
            value_mean = gl.load(vm_slot_ptr + features[None, :])
            numerator = _fold_safe_value_mean(numerator, chunk_denom, value_mean)

    output, lse = _finalize_safe_output(numerator, denominator, running_max, running_bias)
    valid_out = output_rows[None, :] < seq_q
    offsets = batch * stride_ob + head * stride_oh + output_rows[None, :, None] * stride_om
    gl.store(output_ptr + offsets + features[None, None, :], output, valid_out[:, :, None])
    gl.store(lse_ptr + batch * stride_lb + head * stride_lh + output_rows[None, :], lse, valid_out)


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
) -> PreparedContext:
    """Prepare validated K/V once, independently of Q and output storage."""
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
) -> PreparedAttention:
    """Prepare operands and allocate the reusable output before timed launches."""
    context = prepare_context(key, value, is_causal=is_causal)
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
) -> torch.Tensor:
    batch, heads, _, head_dim = query.shape
    query_storage = query.data.shape[2]
    key_storage = context.key.shape[2]
    key_chunks = context.key_center.shape[2]
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
            context.is_causal,
            query_storage * (head_dim // 8) > (1 << 31),
            key_storage * head_dim > (1 << 32),
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
    if batch == 0:
        return output
    return _launch_window_into(
        context,
        query,
        output,
        query_start=query_start,
        rows=rows,
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
) -> torch.Tensor:
    if query.shape[0] == 0:
        return torch.empty_like(query, memory_format=torch.contiguous_format)
    return launch_attention(prepare_attention(query, key, value, scale, is_causal))


def launch_frozen_pool_kernel(
    common_args: tuple[object, ...],
    *,
    device: torch.device,
    batch: int,
    q_heads: int,
    seq_q: int,
    query_storage: int,
    gqa_group: int,
    head_dim: int,
    has_chunk_map: bool,
) -> None:
    """Launch RDNA4 frozen-pool prefill kernel with pre-assembled host arguments."""
    with device_context(device):
        _frozen_pool_piper_kernel[(triton.cdiv(seq_q, 64), q_heads, batch)](
            *common_args,
            gqa_group,
            head_dim,
            has_chunk_map,
            query_storage * (head_dim // 8) > (1 << 31),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill grouped queries over quantized live tail with per-batch tail_len & offset causal."""
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
            is_causal,
            query_storage * (head_dim // 8) > (1 << 31),
            key_storage * head_dim > (1 << 32),
            lse_ptr=lse,
            key_lengths_ptr=tail_len,
            stride_lb=lse.stride(0),
            stride_lh=lse.stride(1),
            store_lse=True,
            has_key_lengths=True,
            offset_causal_from_key_length=is_causal,
            num_warps=4,
            num_stages=1,
            llvm_fn_attrs=_LLVM_FN_ATTRS,
        )
    return output, lse


def compile_amd_prefill_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    wide_pool: bool = False,
    compile_fn=None,
) -> tuple[object, object]:
    """Offline-compile RDNA4 frozen-pool and offset-causal tail prefill kernels."""
    triton_dtype = _quantization.validate_kv_compile_args(dtype, head_dim)
    options = {"num_warps": 4, "num_stages": 1, "llvm_fn_attrs": _LLVM_FN_ATTRS}
    compiler = triton.compile if compile_fn is None else compile_fn

    frozen_source = _quantization.build_kernel_source(
        _frozen_pool_piper_kernel,
        constants={
            "query_heads_per_kv_head": 3,
            "head_dim": head_dim,
            "has_chunk_map": has_chunk_map,
            "wide_query_offsets": False,
        },
        overrides=_quantization.frozen_pool_pointer_signature(triton_dtype, wide_pool=wide_pool),
        gluon=True,
        pointer_range_32=not wide_pool,
    )

    tail_overrides: dict[str, str] = dict.fromkeys(
        (
            "query_scale_ptr",
            "key_scale_ptr",
            "key_center_ptr",
            "multiplier_ptr",
            "log_scale_ptr",
            "value_mean_ptr",
            "lse_ptr",
        ),
        "*fp32",
    )
    tail_overrides.update(
        query_ptr="*i8",
        key_ptr="*i8",
        value_ptr="*i8",
        key_lengths_ptr="*i32",
        output_ptr=f"*{triton_dtype}",
    )
    tail_source = _quantization.build_kernel_source(
        _dense_piper_kernel,
        constants={
            "query_heads_per_kv_head": 3,
            "head_dim": head_dim,
            "is_causal": is_causal,
            "wide_query_offsets": False,
            "wide_context_offsets": False,
            "store_lse": True,
            "has_key_lengths": True,
            "offset_causal_from_key_length": is_causal,
        },
        overrides=tail_overrides,
        gluon=True,
    )
    return (
        compiler(frozen_source, target, options),
        compiler(tail_source, target, options),
    )
