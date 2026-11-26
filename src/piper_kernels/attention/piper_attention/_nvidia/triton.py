"""NVIDIA Triton backend for Piper Attention.

The kernel keeps Sage-style INT8 QK and FP32 online softmax, but replaces the
FP8 PV path with one signed-INT8 scale per V row. Each row scale is folded into
the nonnegative probability operand, producing UINT8-by-INT8 tensor-core dots.
Native mixed-sign MMA is used on supported NVIDIA targets; other targets use
the portable PyTorch reference.
"""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import math
from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels._triton.mixed_int8 import (
    install_uint8_int8_dot_hook,
    uint8_int8_dot,
)
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.piper.triton import (
    advance_tile_max_safe,
    conservative_value_log_scale_bound,
    max_propagate_nan,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

from .. import _quantization
from .._validation import (
    PreparedAttention as _PreparedPiperAttention,
)
from .._validation import (
    PreparedContext as _PreparedPiperContext,
)
from .._validation import (
    PreparedQuery as _PreparedPiperQuery,
)
from .._validation import (
    resolve_query_window,
    validate_output_buffer,
    validate_query_offset,
)
from . import gluon_async_copy as _gluon_async_copy
from . import policy as _policy
from .policy import PiperAttentionExecutionPlan

_BLOCK_N = 64
_TL_CENTER_CHUNK = tl.constexpr(_quantization.CENTER_CHUNK)
_TL_TILES_PER_CHUNK = tl.constexpr(_quantization._TILES_PER_CHUNK)
_K_BIAS_SLICE_D = tl.constexpr(16)
_P_UINT8_RANGE = tl.constexpr(255.0)
_LN2 = tl.constexpr(math.log(2.0))


@triton.jit
def _ptx_float32_to_uint8x4(values):
    """Truncate and saturate four probability codes with packed SM72+ PTX."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .s32 a, b, c, d;
            .reg .b32 lo;
            cvt.rzi.s32.f32 a, $1;
            cvt.rzi.s32.f32 b, $2;
            cvt.rzi.s32.f32 c, $3;
            cvt.rzi.s32.f32 d, $4;
            cvt.pack.sat.u8.s32.b32 lo, d, c, 0;
            cvt.pack.sat.u8.s32.b32 $0, b, a, lo;
        }
        """,
        constraints="=r,f,f,f,f",
        args=[values],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _load_key_tile(
    key_ptr,
    batch_head,
    start_n,
    current_n,
    offsets_d,
    key_length,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    padded_kv: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    if use_tensor_descriptors:
        return key_ptr.load([batch_head, start_n, 0]).reshape((block_n, head_dim)).T
    else:
        return tl.load(
            key_ptr
            + (
                batch_head * (tl.cdiv(key_length, 64) * 64 if padded_kv else key_length)
                + current_n[None, :]
            )
            * head_dim
            + offsets_d[:, None],
            mask=current_n[None, :] < key_length,
            other=0,
        )


@triton.jit
def _load_value_tile(
    value_ptr,
    batch_head,
    start_n,
    current_n,
    offsets_d,
    key_length,
    feature_start: tl.constexpr,
    feature_block: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    padded_kv: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    if use_tensor_descriptors:
        return (
            value_ptr.load([batch_head, feature_start, start_n]).reshape((feature_block, block_n)).T
        )
    else:
        return tl.load(
            value_ptr
            + (batch_head * head_dim + feature_start + offsets_d[None, :])
            * (tl.cdiv(key_length, 64) * 64 if padded_kv else key_length)
            + current_n[:, None],
            mask=current_n[:, None] < key_length,
            other=0,
        )


@triton.jit
def _compute_chunk_k_bias(
    query,
    raw_query_ptr,
    query_scale,
    key_center_ptr,
    batch_head,
    query_storage_length,
    query_tile_start,
    kv_batch_head,
    chunk_idx,
    key_chunks,
    offsets_d,
    valid_queries,
    split_pv_head_dim: tl.constexpr,
    unmasked_query_tiles: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Compute one K1024 chunk's Q @ H(K_mean) scalar bias vector per query row."""
    block_m: tl.constexpr = query.shape[0]
    chunk_offset = (kv_batch_head.to(tl.int64) * key_chunks + chunk_idx.to(tl.int64)) * head_dim
    key_center_base = key_center_ptr + chunk_offset
    if split_pv_head_dim and block_m == 128:
        # Dual D64 FP32 accumulators nearly exhaust SM120's 255-reg file on Q128;
        # stream 16-col L1 Q slices instead of widening 128x128 INT8 in registers.
        slice_offsets = tl.arange(0, _K_BIAS_SLICE_D)
        row_offsets = query_tile_start + tl.arange(0, block_m)
        query_row_base = (
            batch_head.to(tl.int64) * query_storage_length + row_offsets[:, None].to(tl.int64)
        ) * head_dim
        acc = tl.zeros((block_m, _K_BIAS_SLICE_D), dtype=tl.float32)
        for feature_start in tl.static_range(  # pyright: ignore[reportGeneralTypeIssues]
            0, head_dim, _K_BIAS_SLICE_D
        ):
            feature_offsets = feature_start + slice_offsets
            if unmasked_query_tiles:
                query_slice = tl.load(raw_query_ptr + query_row_base + feature_offsets[None, :])
            else:
                query_slice = tl.load(
                    raw_query_ptr + query_row_base + feature_offsets[None, :],
                    mask=valid_queries[:, None],
                    other=0,
                )
            center_slice = tl.load(key_center_base + feature_offsets)
            acc += query_slice.to(tl.float32) * center_slice[None, :]
        return tl.sum(acc, axis=1) * query_scale
    key_center = tl.load(key_center_base + offsets_d)
    if split_pv_head_dim:
        query_even, query_odd = tl.split(tl.reshape(query, (block_m, head_dim // 2, 2)))
        center_even, center_odd = tl.split(tl.reshape(key_center, (head_dim // 2, 2)))
        acc = query_even.to(tl.float32) * center_even[None, :]
        acc += query_odd.to(tl.float32) * center_odd[None, :]
        return tl.sum(acc, axis=1) * query_scale
    return tl.sum(query.to(tl.float32) * key_center[None, :], axis=1) * query_scale


@triton.jit
def _restore_chunk_value_mean(
    numerator,
    chunk_denom,
    value_mean_ptr,
    kv_batch_head,
    chunk_idx,
    key_chunks,
    offsets_d,
    offsets_vd,
    split_pv_head_dim: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Fold one K1024 chunk's softmax mass times its V mean into the FP32 numerator."""
    chunk_weight_codes = chunk_denom * _P_UINT8_RANGE
    value_mean_base = (
        value_mean_ptr + (kv_batch_head.to(tl.int64) * key_chunks + chunk_idx) * head_dim
    )
    if split_pv_head_dim:
        half_head_dim: tl.constexpr = head_dim // 2
        accumulator_low, accumulator_high = numerator
        value_mean_low = tl.load(value_mean_base + offsets_vd)
        value_mean_high = tl.load(value_mean_base + half_head_dim + offsets_vd)
        accumulator_low += chunk_weight_codes[:, None] * value_mean_low[None, :]
        accumulator_high += chunk_weight_codes[:, None] * value_mean_high[None, :]
        return (accumulator_low, accumulator_high)
    else:
        accumulator = numerator
        value_mean = tl.load(value_mean_base + offsets_d)
        accumulator += chunk_weight_codes[:, None] * value_mean[None, :]
        return accumulator


@triton.jit
def _advance_tile_max(running_max, running_bias, block_max, k_bias, valid_queries):
    """Update winner-frame running maximum across chunks without round-tripping."""
    has_block = block_max != -float("inf")
    has_running = running_max != -float("inf")
    raw_delta = (block_max - running_max) + (k_bias - running_bias)
    delta = tl.where(
        has_block,
        tl.where(has_running, raw_delta, float("inf")),
        -float("inf"),
    )
    tile_wins = delta > 0.0
    next_max = tl.where(tile_wins, block_max, running_max)
    next_bias = tl.where(tile_wins, k_bias, running_bias)
    decay = tl.exp2(-tl.abs(delta))
    old_weight = tl.where(valid_queries, tl.where(tile_wins, decay, 1.0), 0.0)
    current_weight = tl.where(valid_queries, tl.where(tile_wins, 1.0, decay), 0.0)
    return next_max, next_bias, old_weight, current_weight


@triton.jit
def _attention_tile(  # noqa: PLR0912, PLR0915
    query,
    query_scale,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    numerator,
    denominator,
    chunk_denom,
    running_max,
    running_bias,
    k_bias,
    batch_head,
    start_n,
    offsets_m,
    offsets_n,
    offsets_d,
    valid_queries,
    key_length,
    mask_keys: tl.constexpr,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    derive_value_log_bound: tl.constexpr,
    padded_kv: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    """Advance online-softmax state by one key tile.

    FP32 numerators remain in UINT8 probability-code units for every attention
    mode; each K1024 chunk folds its V mean once outside the inner loop.
    """
    current_n = start_n + offsets_n
    key = _load_key_tile(
        key_ptr,
        batch_head,
        start_n,
        current_n,
        offsets_d,
        key_length,
        head_dim,
        block_n,
        use_tensor_descriptors,
        padded_kv=padded_kv,
    )
    integer_scores = tl.dot(query, key, out_dtype=tl.int32)
    metadata_stride = tl.cdiv(key_length, block_n) * block_n if padded_kv else key_length
    if grouped_qk:
        key_scale = tl.load(
            key_scale_ptr + batch_head * tl.cdiv(key_length, block_n) + start_n // block_n
        )
        scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
    else:
        key_scale = tl.load(
            key_scale_ptr + batch_head * metadata_stride + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
        scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]

    if mask_keys:
        if is_causal:
            valid_keys = current_n[None, :] < key_length
            valid_keys &= current_n[None, :] <= offsets_m[:, None]
        else:
            valid_keys = current_n < key_length
    elif is_causal:
        valid_keys = tl.full((block_m, block_n), True, dtype=tl.int1)
    else:
        valid_keys = tl.full((block_n,), True, dtype=tl.int1)
    if mask_keys and is_causal:
        scores = tl.where(valid_queries[:, None] & valid_keys, scores, -float("inf"))

    if derive_value_log_bound:
        value_scale_multiplier = tl.load(
            value_scale_multiplier_ptr + batch_head * metadata_stride + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
        value_log_scale = conservative_value_log_scale_bound(value_scale_multiplier)
    else:
        value_log_scale = tl.load(
            value_log_scale_ptr + batch_head * metadata_stride + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
    if mask_keys and not is_causal:
        # The loop's final tile always contains at least one real key. Exclude
        # only its padded columns from the shifted maximum and probability sum;
        # masking the full MxN score tile on every iteration is unnecessary.
        value_log_scale = tl.where(valid_keys, value_log_scale, -float("inf"))
    shifted_scores = scores + value_log_scale[None, :]
    block_max = tl.max(shifted_scores, axis=1)
    next_max, next_bias, old_weight, current_weight = _advance_tile_max(
        running_max,
        running_bias,
        block_max,
        k_bias,
        valid_queries,
    )
    probabilities = tl.where(
        valid_queries[:, None] & valid_keys,
        tl.exp2(scores - block_max[:, None]),  # pyright: ignore[reportArgumentType]
        0.0,
    )
    row_sum = tl.sum(probabilities, axis=1) * current_weight
    denominator = denominator * old_weight + row_sum
    chunk_denom = chunk_denom * old_weight + row_sum
    if not derive_value_log_bound:
        value_scale_multiplier = tl.load(
            value_scale_multiplier_ptr + batch_head * metadata_stride + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
    probability_values = probabilities * value_scale_multiplier[None, :] + 0.5  # pyright: ignore[reportPossiblyUnboundVariable]
    if use_packed_probability_conversion:
        probability_uint8 = _ptx_float32_to_uint8x4(probability_values)
    else:
        probability_codes = tl.minimum(
            _P_UINT8_RANGE,
            probability_values,
        ).to(tl.int32)
        probability_uint8 = probability_codes.to(tl.uint8)

    if split_pv_head_dim:
        accumulator_low, accumulator_high = numerator
        half_head_dim: tl.constexpr = head_dim // 2
        offsets_vd = tl.arange(0, half_head_dim)
        value_low = _load_value_tile(
            value_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_vd,
            key_length,
            tl.constexpr(0),
            half_head_dim,
            head_dim,
            block_n,
            use_tensor_descriptors,
            padded_kv=padded_kv,
        )
        value_high = _load_value_tile(
            value_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_vd,
            key_length,
            half_head_dim,
            half_head_dim,
            head_dim,
            block_n,
            use_tensor_descriptors,
            padded_kv=padded_kv,
        )
        partial_low = uint8_int8_dot(probability_uint8, value_low)
        partial_high = uint8_int8_dot(probability_uint8, value_high)
        accumulator_low = (
            accumulator_low * old_weight[:, None]
            + partial_low.to(tl.float32) * current_weight[:, None]
        )
        accumulator_high = (
            accumulator_high * old_weight[:, None]
            + partial_high.to(tl.float32) * current_weight[:, None]
        )
        numerator = (accumulator_low, accumulator_high)
    else:
        accumulator = numerator
        value_tile = _load_value_tile(
            value_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            tl.constexpr(0),
            head_dim,
            head_dim,
            block_n,
            use_tensor_descriptors,
            padded_kv=padded_kv,
        )
        partial = uint8_int8_dot(probability_uint8, value_tile)
        accumulator = (
            accumulator * old_weight[:, None] + partial.to(tl.float32) * current_weight[:, None]
        )
        numerator = accumulator
    return numerator, denominator, next_max, next_bias, chunk_denom


@triton.jit
def _attention_tile_range(
    query,
    query_scale,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    numerator,
    denominator,
    chunk_denom,
    running_max,
    running_bias,
    k_bias,
    kv_batch_head,
    range_start,
    range_stop,
    global_rows,
    offsets_n,
    offsets_d,
    valid_queries,
    key_length,
    mask_keys: tl.constexpr,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    derive_value_log_bound: tl.constexpr,
    loop_num_stages: tl.constexpr,
    loop_licm: tl.constexpr,
    padded_kv: tl.constexpr,
):
    """Advance online softmax and chunk denominator over one K64 tile range."""
    for start_n in tl.range(  # pyright: ignore[reportGeneralTypeIssues]
        range_start,
        range_stop,
        block_n,
        num_stages=loop_num_stages,
        disable_licm=not loop_licm,
    ):
        numerator, denominator, running_max, running_bias, chunk_denom = _attention_tile(
            query,
            query_scale,
            key_ptr,
            value_ptr,
            key_scale_ptr,
            value_scale_multiplier_ptr,
            value_log_scale_ptr,
            numerator,
            denominator,
            chunk_denom,
            running_max,
            running_bias,
            k_bias,
            kv_batch_head,
            start_n,
            global_rows,
            offsets_n,
            offsets_d,
            valid_queries,
            key_length,
            mask_keys=mask_keys,
            is_causal=is_causal,
            grouped_qk=grouped_qk,
            split_pv_head_dim=split_pv_head_dim,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            use_tensor_descriptors=use_tensor_descriptors,
            use_packed_probability_conversion=use_packed_probability_conversion,
            derive_value_log_bound=derive_value_log_bound,
            padded_kv=padded_kv,
        )
    return numerator, denominator, chunk_denom, running_max, running_bias


@triton.jit
def _attention_chunk(
    query,
    raw_query_ptr,
    query_scale,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    key_center_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    value_mean_ptr,
    numerator,
    denominator,
    running_max,
    running_bias,
    batch_head,
    query_storage_length,
    query_tile_start,
    kv_batch_head,
    chunk_idx,
    key_chunks,
    prefix_start,
    prefix_stop,
    boundary_start,
    boundary_stop,
    global_rows,
    offsets_n,
    offsets_d,
    offsets_vd,
    valid_queries,
    key_length,
    prefix_mask_keys: tl.constexpr,
    has_boundary: tl.constexpr,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    unmasked_query_tiles: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    derive_value_log_bound: tl.constexpr,
    loop_num_stages: tl.constexpr,
    loop_licm: tl.constexpr,
    padded_kv: tl.constexpr,
):
    """Evaluate one K1024+V1024 chunk: compute K bias, run K64 tiles, and fold V mean."""
    k_bias = _compute_chunk_k_bias(
        query,
        raw_query_ptr,
        query_scale,
        key_center_ptr,
        batch_head,
        query_storage_length,
        query_tile_start,
        kv_batch_head,
        chunk_idx,
        key_chunks,
        offsets_d,
        valid_queries,
        split_pv_head_dim,
        unmasked_query_tiles,
        head_dim,
    )
    chunk_denom = tl.zeros((block_m,), dtype=tl.float32)
    numerator, denominator, chunk_denom, running_max, running_bias = _attention_tile_range(
        query,
        query_scale,
        key_ptr,
        value_ptr,
        key_scale_ptr,
        value_scale_multiplier_ptr,
        value_log_scale_ptr,
        numerator,
        denominator,
        chunk_denom,
        running_max,
        running_bias,
        k_bias,
        kv_batch_head,
        prefix_start,
        prefix_stop,
        global_rows,
        offsets_n,
        offsets_d,
        valid_queries,
        key_length,
        mask_keys=prefix_mask_keys,
        is_causal=is_causal,
        grouped_qk=grouped_qk,
        split_pv_head_dim=split_pv_head_dim,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        use_tensor_descriptors=use_tensor_descriptors,
        use_packed_probability_conversion=use_packed_probability_conversion,
        derive_value_log_bound=derive_value_log_bound,
        loop_num_stages=loop_num_stages,
        loop_licm=loop_licm,
        padded_kv=padded_kv,
    )
    if has_boundary:
        numerator, denominator, chunk_denom, running_max, running_bias = _attention_tile_range(
            query,
            query_scale,
            key_ptr,
            value_ptr,
            key_scale_ptr,
            value_scale_multiplier_ptr,
            value_log_scale_ptr,
            numerator,
            denominator,
            chunk_denom,
            running_max,
            running_bias,
            k_bias,
            kv_batch_head,
            boundary_start,
            boundary_stop,
            global_rows,
            offsets_n,
            offsets_d,
            valid_queries,
            key_length,
            mask_keys=tl.constexpr(True),
            is_causal=is_causal,
            grouped_qk=grouped_qk,
            split_pv_head_dim=split_pv_head_dim,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            use_tensor_descriptors=use_tensor_descriptors,
            use_packed_probability_conversion=use_packed_probability_conversion,
            derive_value_log_bound=derive_value_log_bound,
            loop_num_stages=loop_num_stages,
            loop_licm=loop_licm,
            padded_kv=padded_kv,
        )
    numerator = _restore_chunk_value_mean(
        numerator,
        chunk_denom,
        value_mean_ptr,
        kv_batch_head,
        chunk_idx,
        key_chunks,
        offsets_d,
        offsets_vd,
        split_pv_head_dim=split_pv_head_dim,
        head_dim=head_dim,
    )
    return numerator, denominator, running_max, running_bias


@triton.jit
def _piper_attention_query_tile(  # noqa: PLR0912, PLR0915
    query_ptr,
    raw_query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    key_center_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    value_mean_ptr,
    output_ptr,
    query_block,
    head,
    query_storage_length,
    key_length,
    query_start,
    query_rows,
    global_query_start,
    key_chunks,
    stride_ob,
    stride_oh,
    stride_om,
    heads,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    unmasked_key_tiles: tl.constexpr,
    query_heads_per_kv_head: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    optimize_causal_traversal: tl.constexpr,
    loop_num_stages: tl.constexpr,
    loop_licm: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    derive_value_log_bound: tl.constexpr,
    contiguous_output: tl.constexpr,
    unmasked_query_tiles: tl.constexpr,
    use_query_tensor_descriptor: tl.constexpr,
    padded_kv: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    """Evaluate one complete or masked query tile with K1024+V1024 outer chunking."""
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // query_heads_per_kv_head) + head // query_heads_per_kv_head
    output_rows = query_block * block_m + tl.arange(0, block_m)
    offsets_m = query_start + output_rows
    global_rows = global_query_start + output_rows
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    if unmasked_query_tiles:
        valid_queries = tl.full((block_m,), True, dtype=tl.int1)
    else:
        valid_queries = output_rows < query_rows

    if use_query_tensor_descriptor:
        query = query_ptr.load([batch_head, query_start + query_block * block_m, 0]).reshape(
            (block_m, head_dim)
        )
    else:
        query = tl.load(
            query_ptr
            + ((batch_head * query_storage_length + offsets_m[:, None]) * head_dim)
            + offsets_d[None, :],
            mask=valid_queries[:, None],
            other=0,
        )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr
            + batch_head * tl.cdiv(query_storage_length, 32)
            + query_start // 32
            + output_rows // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + batch_head * query_storage_length + offsets_m,
            mask=valid_queries,
            other=0.0,
        )

    half_head_dim: tl.constexpr = head_dim // 2
    offsets_vd = tl.arange(0, half_head_dim)
    if split_pv_head_dim:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        numerator = (accumulator_low, accumulator_high)
    else:
        numerator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    running_bias = tl.zeros((block_m,), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, global_query_start + (query_block + 1) * block_m)

    query_tile_start = query_start + query_block * block_m

    if is_causal and optimize_causal_traversal:
        # Chunks strictly before the diagonal run fully unmasked; chunks from
        # split_chunk_start handle prefix + diagonal boundary tiles cleanly.
        full_key_end = key_length // block_n * block_n
        causal_prefix_end = (global_query_start + query_block * block_m) // block_n * block_n
        prefix_end = tl.minimum(causal_prefix_end, full_key_end)
        split_chunk_start = (prefix_end // _TL_CENTER_CHUNK) * _TL_CENTER_CHUNK
        mask_free_stop = tl.minimum(split_chunk_start, end_n)
        for chunk_start in tl.range(0, mask_free_stop, _TL_CENTER_CHUNK, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            chunk_idx = chunk_start // _TL_CENTER_CHUNK
            chunk_end = chunk_start + _TL_CENTER_CHUNK
            numerator, denominator, running_max, running_bias = _attention_chunk(
                query,
                raw_query_ptr,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                key_center_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                value_mean_ptr,
                numerator,
                denominator,
                running_max,
                running_bias,
                batch_head,
                query_storage_length,
                query_tile_start,
                kv_batch_head,
                chunk_idx,
                key_chunks,
                chunk_start,
                chunk_end,
                chunk_end,
                chunk_end,
                global_rows,
                offsets_n,
                offsets_d,
                offsets_vd,
                valid_queries,
                key_length,
                prefix_mask_keys=tl.constexpr(False),
                has_boundary=tl.constexpr(False),
                is_causal=is_causal,
                grouped_qk=grouped_qk,
                split_pv_head_dim=split_pv_head_dim,
                unmasked_query_tiles=unmasked_query_tiles,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
                use_packed_probability_conversion=use_packed_probability_conversion,
                derive_value_log_bound=derive_value_log_bound,
                loop_num_stages=loop_num_stages,
                loop_licm=loop_licm,
                padded_kv=padded_kv,
            )
        for chunk_start in tl.range(mask_free_stop, end_n, _TL_CENTER_CHUNK, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            chunk_idx = chunk_start // _TL_CENTER_CHUNK
            chunk_end = tl.minimum(chunk_start + _TL_CENTER_CHUNK, end_n)
            prefix_stop = tl.minimum(chunk_end, prefix_end)
            boundary_start = tl.maximum(chunk_start, prefix_end)
            numerator, denominator, running_max, running_bias = _attention_chunk(
                query,
                raw_query_ptr,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                key_center_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                value_mean_ptr,
                numerator,
                denominator,
                running_max,
                running_bias,
                batch_head,
                query_storage_length,
                query_tile_start,
                kv_batch_head,
                chunk_idx,
                key_chunks,
                chunk_start,
                prefix_stop,
                boundary_start,
                chunk_end,
                global_rows,
                offsets_n,
                offsets_d,
                offsets_vd,
                valid_queries,
                key_length,
                prefix_mask_keys=tl.constexpr(False),
                has_boundary=tl.constexpr(True),
                is_causal=is_causal,
                grouped_qk=grouped_qk,
                split_pv_head_dim=split_pv_head_dim,
                unmasked_query_tiles=unmasked_query_tiles,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
                use_packed_probability_conversion=use_packed_probability_conversion,
                derive_value_log_bound=derive_value_log_bound,
                loop_num_stages=loop_num_stages,
                loop_licm=loop_licm,
                padded_kv=padded_kv,
            )
    else:
        for chunk_start in tl.range(0, end_n, _TL_CENTER_CHUNK, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            chunk_idx = chunk_start // _TL_CENTER_CHUNK
            chunk_end = tl.minimum(chunk_start + _TL_CENTER_CHUNK, end_n)
            numerator, denominator, running_max, running_bias = _attention_chunk(
                query,
                raw_query_ptr,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                key_center_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                value_mean_ptr,
                numerator,
                denominator,
                running_max,
                running_bias,
                batch_head,
                query_storage_length,
                query_tile_start,
                kv_batch_head,
                chunk_idx,
                key_chunks,
                chunk_start,
                chunk_end,
                chunk_end,
                chunk_end,
                global_rows,
                offsets_n,
                offsets_d,
                offsets_vd,
                valid_queries,
                key_length,
                prefix_mask_keys=tl.constexpr(not unmasked_key_tiles),
                has_boundary=tl.constexpr(False),
                is_causal=is_causal,
                grouped_qk=grouped_qk,
                split_pv_head_dim=split_pv_head_dim,
                unmasked_query_tiles=unmasked_query_tiles,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
                use_packed_probability_conversion=use_packed_probability_conversion,
                derive_value_log_bound=derive_value_log_bound,
                loop_num_stages=loop_num_stages,
                loop_licm=loop_licm,
                padded_kv=padded_kv,
            )

    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    denominator_code_units = denominator_safe * _P_UINT8_RANGE
    if contiguous_output:
        output_base = (
            output_ptr + (batch_head.to(tl.int64) * query_rows + output_rows[:, None]) * head_dim
        )
    else:
        output_base = (
            output_ptr
            + batch.to(tl.int64) * stride_ob
            + head.to(tl.int64) * stride_oh
            + output_rows[:, None].to(tl.int64) * stride_om
        )
    if split_pv_head_dim:
        accumulator_low, accumulator_high = numerator
        output_low = accumulator_low / denominator_code_units  # pyright: ignore[reportPossiblyUnboundVariable]
        output_high = accumulator_high / denominator_code_units  # pyright: ignore[reportPossiblyUnboundVariable]
        tl.store(
            output_base + offsets_vd[None, :],  # pyright: ignore[reportPossiblyUnboundVariable]
            output_low,
            mask=valid_queries[:, None],
        )
        tl.store(
            output_base + half_head_dim + offsets_vd[None, :],  # pyright: ignore[reportPossiblyUnboundVariable]
            output_high,
            mask=valid_queries[:, None],
        )
    else:
        output = numerator / denominator_code_units  # pyright: ignore[reportPossiblyUnboundVariable]
        tl.store(
            output_base + offsets_d[None, :],
            output,
            mask=valid_queries[:, None],
        )


@triton.jit(
    do_not_specialize=[
        "query_storage_length",
        "key_length",
        "heads",
        "query_start",
        "query_rows",
        "global_query_start",
        "key_chunks",
        "stride_ob",
        "stride_oh",
        "stride_om",
    ]
)
def _piper_attention_kernel(
    query_ptr,
    query_descriptor,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    key_center_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    value_mean_ptr,
    output_ptr,
    query_storage_length,
    key_length,
    query_start,
    query_rows,
    global_query_start,
    key_chunks,
    stride_ob,
    stride_oh,
    stride_om,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    aligned_queries: tl.constexpr,
    unmasked_key_tiles: tl.constexpr,
    heads,
    query_heads_per_kv_head: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_query_tensor_descriptor: tl.constexpr,
    optimize_causal_traversal: tl.constexpr,
    loop_num_stages: tl.constexpr,
    loop_licm: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    derive_value_log_bound: tl.constexpr,
    full_query: tl.constexpr,
    contiguous_output: tl.constexpr,
    padded_kv: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
    query_group_size: tl.constexpr = 0,  # pyright: ignore[reportArgumentType]
):
    """Cover full query tiles and their ragged tail in one grid."""
    if full_query:
        query_start = 0
        global_query_start = 0
        query_rows = query_storage_length
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    if query_group_size > 0:
        query_tiles = tl.num_programs(0)
        linear = head * query_tiles + query_block
        group = linear // (query_group_size * heads)
        group_rows = tl.minimum(query_tiles - group * query_group_size, query_group_size)
        group_offset = linear % (query_group_size * heads)
        head = group_offset // group_rows
        query_block = group * query_group_size + group_offset % group_rows
    if is_causal and optimize_causal_traversal:
        query_block = tl.num_programs(0) - 1 - query_block
    tile_args = (
        key_ptr,
        value_ptr,
        query_scale_ptr,
        key_scale_ptr,
        key_center_ptr,
        value_scale_multiplier_ptr,
        value_log_scale_ptr,
        value_mean_ptr,
        output_ptr,
        query_block,
        head,
        query_storage_length,
        key_length,
        query_start,
        query_rows,
        global_query_start,
        key_chunks,
        stride_ob,
        stride_oh,
        stride_om,
        heads,
    )
    tile_options = tl.constexpr(
        (
            is_causal,
            grouped_qk,
            split_pv_head_dim,
            unmasked_key_tiles,
            query_heads_per_kv_head,
            head_dim,
            block_m,
            block_n,
            use_tensor_descriptors,
            optimize_causal_traversal,
            loop_num_stages,
            loop_licm,
            use_packed_probability_conversion,
            derive_value_log_bound,
            contiguous_output,
        )
    )
    # This CTA-uniform branch folds away for aligned queries. Only the tail
    # needs masked pointer loads; full tiles retain the Q descriptor if present.
    if aligned_queries or query_block < query_rows // block_m:
        _piper_attention_query_tile(
            query_descriptor if use_query_tensor_descriptor else query_ptr,
            query_ptr,
            *tile_args,
            *tile_options,
            unmasked_query_tiles=tl.constexpr(True),
            use_query_tensor_descriptor=use_query_tensor_descriptor,
            padded_kv=padded_kv,
        )
    else:
        _piper_attention_query_tile(
            query_ptr,
            query_ptr,
            *tile_args,
            *tile_options,
            unmasked_query_tiles=tl.constexpr(False),
            use_query_tensor_descriptor=tl.constexpr(False),
            padded_kv=padded_kv,
        )


def _make_key_value_descriptors(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    split_pv_head_dim: bool,
) -> tuple[TensorDescriptor, TensorDescriptor]:
    batch, heads, storage_key_length, head_dim = key.shape
    with device_context(key.device):
        key_descriptor = TensorDescriptor(
            base=key,
            shape=[batch * heads, storage_key_length, head_dim],
            strides=[storage_key_length * head_dim, head_dim, 1],
            block_shape=[1, _BLOCK_N, head_dim],
        )
        value_descriptor = TensorDescriptor(
            base=value,
            shape=[batch * heads, head_dim, storage_key_length],
            strides=[head_dim * storage_key_length, storage_key_length, 1],
            block_shape=[
                1,
                head_dim // 2 if split_pv_head_dim else head_dim,
                _BLOCK_N,
            ],
        )
        return key_descriptor, value_descriptor


def _make_query_descriptor(
    query: torch.Tensor,
    block_m: int,
) -> TensorDescriptor:
    """Describe flattened-BH Q for complete query-block loads."""
    batch, heads, query_length, head_dim = query.shape
    with device_context(query.device):
        return TensorDescriptor(
            base=query,
            shape=[batch * heads, query_length, head_dim],
            strides=[query_length * head_dim, head_dim, 1],
            block_shape=[1, block_m, head_dim],
        )


def default_execution_plan(
    query: torch.Tensor,
    is_causal: bool,
    *,
    target: AcceleratorTarget | None = None,
    key_length: int | None = None,
) -> PiperAttentionExecutionPlan:
    """Resolve production policy for preparation, benchmarks, and tuning."""
    head_dim = query.shape[3]
    target = AcceleratorTarget.from_device(query.device) if target is None else target
    return _policy.select_execution_plan(
        target,
        head_dim=head_dim,
        is_causal=is_causal,
        query_length=query.shape[2],
        key_length=key_length,
    )


def _prepare_piper_context(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    execution_plan: PiperAttentionExecutionPlan,
) -> _PreparedPiperContext:
    """Prepare validated K/V once, independently of Q and output storage."""
    _batch, _kv_heads, key_length, head_dim = key.shape
    plan = execution_plan
    if plan.split_pv_head_dim and head_dim != 128:
        raise ValueError("split-PV Piper Attention requires head_dim=128")
    if plan.optimize_causal_traversal and not is_causal:
        raise ValueError("optimized causal traversal requires causal attention")
    plan.validate_kernel()
    with device_context(key.device):
        install_uint8_int8_dot_hook()
        padded_key_length = int(triton.cdiv(key_length, _BLOCK_N)) * _BLOCK_N
        # Descriptors and the Gluon kernel's cp.async copies read whole K64 tiles.
        pad_storage = plan.use_tensor_descriptors or plan.attention_kernel == "gluon_async_copy"
        storage_key_length = padded_key_length if pad_storage else key_length

        key_int8, key_scale, key_center = _quantization.prepare_chunk_centered_key(
            key,
            grouped=plan.grouped_qk,
            storage_key_length=storage_key_length,
        )
        value_int8, value_scale_multiplier, value_log_scale, value_mean = (
            _quantization.prepare_chunk_centered_value(
                value,
                storage_key_length=storage_key_length,
                store_log_scale=not plan.derive_value_log_bound,
            )
        )

        key_argument: torch.Tensor | TensorDescriptor = key_int8
        value_argument: torch.Tensor | TensorDescriptor = value_int8
        if plan.use_tensor_descriptors:
            key_argument, value_argument = _make_key_value_descriptors(
                key_int8,
                value_int8,
                split_pv_head_dim=plan.split_pv_head_dim,
            )
        return _PreparedPiperContext(
            key=key_argument,
            value=value_argument,
            key_scale=key_scale,
            key_center=key_center,
            value_scale_multiplier=value_scale_multiplier,
            value_log_scale=value_log_scale,
            value_mean=value_mean,
            key_length=key_length,
            is_causal=is_causal,
            execution_plan=plan,
        )


def _prepare_piper_query(
    query: torch.Tensor,
    scale: float,
    *,
    execution_plan: PiperAttentionExecutionPlan,
    global_row_offset: int = 0,
) -> _PreparedPiperQuery:
    """Prepare Q using its K/V plan and a tile-aligned global origin.

    To match full-sequence quantization, chunks must contain complete Q32 scale
    groups except at the sequence tail. Launch windows may end within a group.
    """
    batch, heads, query_length, head_dim = query.shape
    plan = execution_plan
    validate_query_offset(global_row_offset, block_rows=plan.block_m, name="global_row_offset")
    plan.validate_kernel()
    if plan.fuse_query_quantization:
        # The Gluon kernel reads floating-point Q and quantizes each tile itself.
        return _PreparedPiperQuery(
            data=query,
            scale=None,
            descriptor=None,
            shape=(batch, heads, query_length, head_dim),
            dtype=query.dtype,
            global_row_offset=global_row_offset,
            softmax_scale=scale,
        )
    # The Gluon kernel copies whole query tiles; padded rows and scales are zero.
    storage_query_length = (
        int(triton.cdiv(query_length, plan.block_m)) * plan.block_m
        if plan.attention_kernel == "gluon_async_copy"
        else None
    )
    with device_context(query.device):
        query_int8, query_scale = qk_quantization.prepare_query(
            query,
            scale,
            grouped=plan.grouped_qk,
            storage_query_length=storage_query_length,
        )
        descriptor = (
            _make_query_descriptor(query_int8, plan.block_m)
            if plan.use_tensor_descriptors and plan.block_m == 128
            else None
        )
    return _PreparedPiperQuery(
        data=query_int8,
        scale=query_scale,
        descriptor=descriptor,
        shape=(batch, heads, query_length, head_dim),
        dtype=query.dtype,
        global_row_offset=global_row_offset,
    )


def _prepare_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    execution_plan: PiperAttentionExecutionPlan,
) -> _PreparedPiperAttention:
    """Prepare all operands and allocate output before timed or captured launches."""
    context = _prepare_piper_context(key, value, is_causal=is_causal, execution_plan=execution_plan)
    prepared_query = _prepare_piper_query(query, scale, execution_plan=execution_plan)
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    return _PreparedPiperAttention(context=context, query=prepared_query, output=output)


def _launch_triton_window_into(
    context: _PreparedPiperContext,
    query: _PreparedPiperQuery,
    output: torch.Tensor,
    *,
    query_start: int,
    rows: int,
) -> torch.Tensor:
    batch, heads, _, head_dim = query.shape
    plan = context.execution_plan
    assert plan is not None
    attention_kernel = cast(Any, _piper_attention_kernel)
    use_query_tensor_descriptor = query.descriptor is not None
    contiguous_output = output.is_contiguous()
    retain_query_tail = (
        plan.retain_query_tail_for_strided_output
        and not contiguous_output
        and context.key_length % _BLOCK_N != 0
    )
    aligned_queries = rows % plan.block_m == 0 and not retain_query_tail
    query_tiles = triton.cdiv(rows, plan.block_m)
    query_group_size = plan.strided_output_query_group if not contiguous_output else 0
    if heads == 1 or query_tiles <= query_group_size:
        query_group_size = 0  # The grouped mapping would be the identity.
    with device_context(query.data.device):
        attention_kernel[(query_tiles, heads, batch)](
            query.data,
            query.descriptor if use_query_tensor_descriptor else query.data,
            context.key,
            context.value,
            query.scale,
            context.key_scale,
            context.key_center,
            context.value_scale_multiplier,
            context.value_log_scale,
            context.value_mean,
            output,
            query.data.shape[2],
            context.key_length,
            query_start,
            rows,
            query.global_row_offset + query_start,
            context.key_center.shape[2],
            output.stride(0),
            output.stride(1),
            output.stride(2),
            is_causal=context.is_causal,
            grouped_qk=plan.grouped_qk,
            split_pv_head_dim=plan.split_pv_head_dim,
            aligned_queries=aligned_queries,
            unmasked_key_tiles=(not context.is_causal and context.key_length % _BLOCK_N == 0),
            heads=heads,
            query_heads_per_kv_head=heads // context.key_scale.shape[1],
            head_dim=head_dim,
            block_m=plan.block_m,
            block_n=_BLOCK_N,
            use_tensor_descriptors=plan.use_tensor_descriptors,
            use_query_tensor_descriptor=use_query_tensor_descriptor,
            optimize_causal_traversal=plan.optimize_causal_traversal,
            loop_num_stages=plan.loop_num_stages,
            loop_licm=plan.loop_licm,
            use_packed_probability_conversion=plan.use_packed_probability_conversion,
            derive_value_log_bound=plan.derive_value_log_bound,
            full_query=(
                query_start == 0 and query.global_row_offset == 0 and rows == query.data.shape[2]
            ),
            contiguous_output=contiguous_output,
            padded_kv=context.padded_kv,
            query_group_size=query_group_size,
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
            maxnreg=(
                plan.ragged_strided_output_maxnreg
                if not contiguous_output and not aligned_queries
                else None
            ),
        )
    return output


def _launch_piper_attention_into(
    context: _PreparedPiperContext,
    query: _PreparedPiperQuery,
    output: torch.Tensor,
    *,
    query_start: int = 0,
    query_rows: int | None = None,
) -> torch.Tensor:
    """Launch a local Q window into caller-owned BHSD output storage."""
    batch, heads, query_length, head_dim = query.shape
    plan = context.execution_plan
    assert plan is not None
    rows = resolve_query_window(
        query_length,
        query_start=query_start,
        query_rows=query_rows,
        global_row_offset=query.global_row_offset,
        block_rows=plan.block_m,
        key_length=context.key_length,
        is_causal=context.is_causal,
    )
    validate_output_buffer(
        output,
        shape=(batch, heads, rows, head_dim),
        dtype=query.dtype,
        device=query.data.device,
    )
    if plan.attention_kernel == "gluon_async_copy":
        assert isinstance(context.key, torch.Tensor)
        assert isinstance(context.value, torch.Tensor)
        return _gluon_async_copy.launch_attention(
            query.data,
            context.key,
            context.value,
            query.scale,
            context.key_scale,
            context.key_center,
            context.value_scale_multiplier,
            context.value_mean,
            output,
            softmax_scale=query.softmax_scale,
            query_length=query.shape[2],
            key_length=context.key_length,
            is_causal=context.is_causal,
            block_q=plan.block_m,
            quantize_query=plan.fuse_query_quantization,
            max_registers=plan.max_registers,
            query_start=query_start,
            global_query_start=query.global_row_offset + query_start,
            padded_kv=context.padded_kv,
        )
    return _launch_triton_window_into(
        context,
        query,
        output,
        query_start=query_start,
        rows=rows,
    )


def _launch_piper_attention(prepared: _PreparedPiperAttention) -> torch.Tensor:
    """Reuse the prepared output without allocating or preparing operands."""
    return _launch_piper_attention_into(prepared.context, prepared.query, prepared.output)


def _run_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    execution_plan: PiperAttentionExecutionPlan | None = None,
) -> torch.Tensor:
    """Run Piper Attention preprocessing and its fused recurrence."""
    plan = (
        execution_plan
        if execution_plan is not None
        else default_execution_plan(
            query,
            is_causal,
            key_length=key.shape[2],
        )
    )
    prepared = _prepare_piper_attention(
        query,
        key,
        value,
        scale,
        is_causal,
        execution_plan=plan,
    )
    return _launch_piper_attention(prepared)


_LOG2_SCALE_EPSILON = tl.constexpr(-23.253496664211536)


@triton.jit(
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
def _piper_chunked_prefill_kernel(  # noqa: PLR0912, PLR0915
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
    query_heads_per_kv_head: tl.constexpr,
    head_dim: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    has_chunk_map: tl.constexpr,
    is_tail: tl.constexpr,
    is_causal: tl.constexpr,
    block_m: tl.constexpr,
    loop_num_stages: tl.constexpr,
):
    """Zero-spill NVIDIA prefill over PiperKVChunkPool (is_tail=False) or quantized live tail."""
    query_block = tl.program_id(0)
    if is_tail and is_causal:
        query_block = tl.num_programs(0) - 1 - query_block
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    kv_head = head // query_heads_per_kv_head
    kv_heads = heads // query_heads_per_kv_head
    kv_batch_head = (batch * kv_heads + kv_head).to(tl.int64)
    output_rows = query_block * block_m + tl.arange(0, block_m)
    valid_queries = output_rows < seq_q
    offsets_n = tl.arange(0, 64)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    offsets_vd = tl.arange(0, half_head_dim)

    query = tl.load(
        query_ptr
        + ((batch_head.to(tl.int64) * query_storage + output_rows[:, None]) * head_dim)
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr
            + batch_head.to(tl.int64) * tl.cdiv(query_storage, 32)
            + output_rows // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + batch_head.to(tl.int64) * query_storage + output_rows,
            mask=valid_queries,
            other=0.0,
        )

    if split_pv_head_dim:
        numerator = (
            tl.zeros((block_m, half_head_dim), dtype=tl.float32),
            tl.zeros((block_m, half_head_dim), dtype=tl.float32),
        )
    else:
        numerator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    running_bias = tl.zeros((block_m,), dtype=tl.float32)

    active_count = tl.minimum(tl.maximum(tl.load(num_chunks_ptr + batch), 0), max_chunks)
    if is_tail:
        # For tail prefill: num_chunks_ptr holds tail_len, max_chunks is tail_cap,
        # and num_slots holds key_storage.
        active_len = active_count
        global_query_start = active_len - seq_q
        tile_count = tl.cdiv(active_len, 64)
        if is_causal:
            causal_end = tl.maximum(global_query_start + query_block * block_m + block_m, 0)
            tile_count = tl.minimum(tile_count, tl.cdiv(causal_end, 64))
        chunk_count = tl.cdiv(tile_count, _TL_TILES_PER_CHUNK)
        key_storage64 = num_slots.to(tl.int64)
        key_chunks64 = tl.cdiv(max_chunks, _TL_CENTER_CHUNK).to(tl.int64)
    else:
        global_query_start = 0
        tile_count = 0
        chunk_count = active_count
        key_storage64 = tl.zeros((), dtype=tl.int64)
        key_chunks64 = tl.zeros((), dtype=tl.int64)

    kv_stride_ki: tl.constexpr = 1024 * head_dim
    kv_stride_ks = tl.constexpr(16 if grouped_qk else 1024)
    for chunk in tl.range(0, chunk_count, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        if is_tail:
            slot = tl.zeros((), dtype=tl.int32)
            chunk_valid = True
        elif has_chunk_map:
            slot = tl.load(
                chunk_to_slot_ptr
                + batch.to(tl.int64) * stride_cs_b
                + chunk.to(tl.int64) * stride_cs_c
            )
            chunk_valid = (slot >= 0) & (slot < num_slots)
        else:
            slot = chunk
            chunk_valid = (slot >= 0) & (slot < num_slots)

        if chunk_valid:
            if is_tail:
                chunk64 = chunk.to(tl.int64)
                chunk_row_offset = chunk64 * 1024
                k_chunk_ptr = (
                    key_int8_ptr + (kv_batch_head * key_storage64 + chunk_row_offset) * head_dim
                )
                ks_chunk_ptr = (
                    key_scale_ptr
                    + kv_batch_head * (key_storage64 // 64 if grouped_qk else key_storage64)
                    + (chunk64 * 16 if grouped_qk else chunk_row_offset)
                )
                kc_chunk_ptr = key_center_ptr + (kv_batch_head * key_chunks64 + chunk64) * head_dim
                v_chunk_ptr = (
                    value_int8_ptr + (kv_batch_head * key_storage64 + chunk_row_offset) * head_dim
                )
                vs_chunk_ptr = value_scale_ptr + kv_batch_head * key_storage64 + chunk_row_offset
                vm_chunk_ptr = value_mean_ptr + (kv_batch_head * key_chunks64 + chunk64) * head_dim
                inner_tiles = tl.minimum(
                    _TL_TILES_PER_CHUNK, tile_count - chunk * _TL_TILES_PER_CHUNK
                )
            else:
                slot64 = slot.to(tl.int64)
                kv_head64 = kv_head.to(tl.int64)
                k_chunk_ptr = key_int8_ptr + slot64 * stride_ki_s + kv_head64 * kv_stride_ki
                ks_chunk_ptr = key_scale_ptr + slot64 * stride_ks_s + kv_head64 * kv_stride_ks
                kc_chunk_ptr = key_center_ptr + slot64 * stride_kc_s + kv_head64 * head_dim
                v_chunk_ptr = value_int8_ptr + slot64 * stride_vi_s + kv_head64 * kv_stride_ki
                vs_chunk_ptr = value_scale_ptr + slot64 * stride_vs_s + kv_head64 * 1024
                vm_chunk_ptr = value_mean_ptr + slot64 * stride_vm_s + kv_head64 * head_dim
                inner_tiles = _TL_TILES_PER_CHUNK

            if split_pv_head_dim:
                query_even, query_odd = tl.split(tl.reshape(query, (block_m, half_head_dim, 2)))
                kc = tl.load(kc_chunk_ptr + offsets_d)
                kc_even, kc_odd = tl.split(tl.reshape(kc, (half_head_dim, 2)))
                q_scale = query_scale[:, None]
                k_acc = (query_even.to(tl.float32) * q_scale) * kc_even[None, :]
                k_acc += (query_odd.to(tl.float32) * q_scale) * kc_odd[None, :]
                k_bias = tl.sum(k_acc, axis=1)
            else:
                kc = tl.load(kc_chunk_ptr + offsets_d)
                k_bias = tl.sum((query.to(tl.float32) * query_scale[:, None]) * kc[None, :], axis=1)
            chunk_active = valid_queries & (k_bias != -float("inf"))
            chunk_denom = tl.zeros((block_m,), dtype=tl.float32)

            for tile in tl.range(0, inner_tiles, num_stages=loop_num_stages, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
                start_n = tile * 64
                current_n = start_n + offsets_n
                key = tl.trans(
                    tl.load(k_chunk_ptr + current_n[:, None] * head_dim + offsets_d[None, :])
                )
                integer_scores = tl.dot(query, key, out_dtype=tl.int32)
                if grouped_qk:
                    key_scale = tl.load(ks_chunk_ptr + tile)
                    scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
                else:
                    key_scale = tl.load(ks_chunk_ptr + current_n)
                    scores = (
                        integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
                    )
                if is_tail:
                    global_n = chunk * 1024 + current_n
                    valid_keys = global_n[None, :] < active_count
                    if is_causal:
                        valid_keys &= (
                            global_n[None, :] <= (global_query_start + output_rows)[:, None]
                        )
                    scores = tl.where(chunk_active[:, None] & valid_keys, scores, -float("inf"))
                else:
                    scores = tl.where(chunk_active[:, None], scores, -float("inf"))
                raw_vs = tl.load(vs_chunk_ptr + current_n)
                value_scale_multiplier = raw_vs * _P_UINT8_RANGE
                value_log_scale = tl.where(raw_vs == 0.0, _LOG2_SCALE_EPSILON, tl.math.log2(raw_vs))
                shifted_scores = tl.where(
                    scores != -float("inf"), scores + value_log_scale[None, :], -float("inf")
                )
                block_max = tl.where(
                    chunk_active, tl.reduce(shifted_scores, 1, max_propagate_nan), -float("inf")
                )
                next_max, next_bias, old_weight, current_weight, has_block, safe_block_max = (
                    advance_tile_max_safe(
                        running_max,
                        running_bias,
                        block_max,
                        k_bias,
                        valid_queries,
                    )
                )
                probabilities = tl.exp2(scores - safe_block_max[:, None])
                row_sum = tl.where(has_block, tl.sum(probabilities, axis=1) * current_weight, 0.0)
                denominator = denominator * old_weight + row_sum
                chunk_denom = chunk_denom * old_weight + row_sum
                probability_values = (
                    tl.where(
                        (chunk_active & has_block)[:, None],
                        probabilities * value_scale_multiplier[None, :],
                        0.0,
                    )
                    + 0.5
                )
                probability_uint8 = _ptx_float32_to_uint8x4(probability_values)

                v_tile_base = v_chunk_ptr + tile * (head_dim * 64)
                if split_pv_head_dim:
                    acc_lo, acc_hi = numerator
                    v_lo = tl.trans(
                        tl.load(v_tile_base + offsets_vd[:, None] * 64 + offsets_n[None, :])
                    )
                    v_hi = tl.trans(
                        tl.load(
                            v_tile_base
                            + (half_head_dim + offsets_vd[:, None]) * 64
                            + offsets_n[None, :]
                        )
                    )
                    p_lo = uint8_int8_dot(probability_uint8, v_lo)
                    p_hi = uint8_int8_dot(probability_uint8, v_hi)
                    acc_lo = (
                        acc_lo * old_weight[:, None] + p_lo.to(tl.float32) * current_weight[:, None]
                    )
                    acc_hi = (
                        acc_hi * old_weight[:, None] + p_hi.to(tl.float32) * current_weight[:, None]
                    )
                    numerator = (acc_lo, acc_hi)
                else:
                    v_tile = tl.trans(
                        tl.load(v_tile_base + offsets_d[:, None] * 64 + offsets_n[None, :])
                    )
                    partial = uint8_int8_dot(probability_uint8, v_tile)
                    numerator = (
                        numerator * old_weight[:, None]
                        + partial.to(tl.float32) * current_weight[:, None]
                    )
                running_max = next_max
                running_bias = next_bias

            chunk_active_mean = (chunk_denom > 0.0)[:, None]
            scale_eff = (chunk_denom * _P_UINT8_RANGE)[:, None]
            if split_pv_head_dim:
                acc_lo, acc_hi = numerator
                vm_lo = tl.load(vm_chunk_ptr + offsets_vd)
                vm_hi = tl.load(vm_chunk_ptr + half_head_dim + offsets_vd)
                acc_lo += tl.where(chunk_active_mean, scale_eff * vm_lo[None, :], 0.0)
                acc_hi += tl.where(chunk_active_mean, scale_eff * vm_hi[None, :], 0.0)
                numerator = (acc_lo, acc_hi)
            else:
                vm = tl.load(vm_chunk_ptr + offsets_d)
                numerator += tl.where(chunk_active_mean, scale_eff * vm[None, :], 0.0)

    has_any = running_max != -float("inf")
    safe_denom = tl.where(has_any & (denominator != 0.0), denominator, 1.0)
    denominator_code_units = safe_denom[:, None] * _P_UINT8_RANGE
    lse = tl.where(
        has_any,
        ((running_max + running_bias) + tl.math.log2(safe_denom)) * _LN2,
        -float("inf"),
    )
    tl.store(
        lse_ptr + batch.to(tl.int64) * stride_lb + head.to(tl.int64) * stride_lh + output_rows,
        lse,
        mask=valid_queries,
    )
    output_base = (
        output_ptr
        + batch.to(tl.int64) * stride_ob
        + head.to(tl.int64) * stride_oh
        + output_rows[:, None].to(tl.int64) * stride_om
    )
    if split_pv_head_dim:
        acc_lo, acc_hi = numerator
        out_lo = tl.where(has_any[:, None], acc_lo / denominator_code_units, 0.0)
        out_hi = tl.where(has_any[:, None], acc_hi / denominator_code_units, 0.0)
        tl.store(output_base + offsets_vd[None, :], out_lo, mask=valid_queries[:, None])
        tl.store(
            output_base + half_head_dim + offsets_vd[None, :], out_hi, mask=valid_queries[:, None]
        )
    else:
        output = tl.where(has_any[:, None], numerator / denominator_code_units, 0.0)
        tl.store(output_base + offsets_d[None, :], output, mask=valid_queries[:, None])


def launch_prefill_kernel(
    common_args: tuple[object, ...],
    *,
    device: torch.device,
    batch: int,
    q_heads: int,
    seq_q: int,
    gqa_group: int,
    head_dim: int,
    grouped_qk: bool,
    has_chunk_map: bool,
    is_tail: bool,
    is_causal: bool,
    target: AcceleratorTarget | None = None,
) -> None:
    """Launch NVIDIA unified chunked prefill kernel with pre-assembled host arguments."""
    resolved_target = AcceleratorTarget.from_device(device) if target is None else target
    block_m, num_warps, split_pv = _policy.select_prefill_config(
        resolved_target, head_dim=head_dim, is_tail=is_tail
    )
    with device_context(device):
        install_uint8_int8_dot_hook()
        _piper_chunked_prefill_kernel[(triton.cdiv(seq_q, block_m), q_heads, batch)](
            *common_args,
            query_heads_per_kv_head=gqa_group,
            head_dim=head_dim,
            grouped_qk=grouped_qk,
            split_pv_head_dim=split_pv,
            has_chunk_map=has_chunk_map,
            is_tail=is_tail,
            is_causal=is_causal,
            block_m=block_m,
            loop_num_stages=1,
            num_warps=num_warps,
            num_stages=1,
        )


def launch_tail_prefill(
    query_int8: torch.Tensor,
    query_scale: torch.Tensor,
    tail_key_int8: torch.Tensor,
    tail_key_scale: torch.Tensor,
    tail_key_center: torch.Tensor,
    tail_value_int8: torch.Tensor,
    tail_value_scale: torch.Tensor,
    tail_value_mean: torch.Tensor,
    tail_len: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    seq_q: int,
    tail_cap: int,
    is_causal: bool,
    grouped_qk: bool,
    target: AcceleratorTarget | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch NVIDIA zero-spill offset-causal quantized tail prefill returning output and LSE."""
    batch, q_heads, _, head_dim = output.shape
    tail_args: tuple[object, ...] = (
        query_int8,
        query_scale,
        tail_key_int8,
        tail_key_scale,
        tail_key_center,
        tail_value_int8,
        tail_value_scale,
        tail_value_mean,
        tail_len,
        tail_len,
        output,
        lse,
        seq_q,
        query_int8.shape[2],
        q_heads,
        tail_key_int8.shape[2],
        tail_cap,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
    )
    launch_prefill_kernel(
        tail_args,
        device=query_int8.device,
        batch=batch,
        q_heads=q_heads,
        seq_q=seq_q,
        gqa_group=q_heads // tail_key_int8.shape[1],
        head_dim=head_dim,
        grouped_qk=grouped_qk,
        has_chunk_map=False,
        is_tail=True,
        is_causal=is_causal,
        target=target,
    )
    return output, lse


def compile_nvidia_prefill_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    compile_fn=None,
) -> tuple[object, object]:
    """Offline-compile unified NVIDIA frozen-pool and offset-causal tail prefill kernels."""
    from triton.compiler import ASTSource  # noqa: PLC0415

    accel = AcceleratorTarget.from_compiler_target(target)
    triton_dtype = "bf16" if dtype is torch.bfloat16 else "fp16"
    grouped_qk = _policy.groups_qk_scales(accel)
    compiler = triton.compile if compile_fn is None else compile_fn

    def _compile_variant(*, tail_mode: bool, causal_mode: bool, chunk_map: bool) -> object:
        block_m, num_warps, split_pv = _policy.select_prefill_config(
            accel, head_dim=head_dim, is_tail=tail_mode
        )
        constants: dict[str, object] = {
            "query_heads_per_kv_head": 3,
            "head_dim": head_dim,
            "grouped_qk": grouped_qk,
            "split_pv_head_dim": split_pv,
            "has_chunk_map": chunk_map,
            "is_tail": tail_mode,
            "is_causal": causal_mode,
            "block_m": block_m,
            "loop_num_stages": 1,
        }
        sig = {
            name: "i32" for name in _piper_chunked_prefill_kernel.arg_names if name not in constants
        }
        sig.update(
            query_ptr="*i8",
            query_scale_ptr="*fp32",
            key_int8_ptr="*i8",
            key_scale_ptr="*fp32",
            key_center_ptr="*fp32",
            value_int8_ptr="*i8",
            value_scale_ptr="*fp32",
            value_mean_ptr="*fp32",
            chunk_to_slot_ptr="*i32",
            num_chunks_ptr="*i32",
            output_ptr=f"*{triton_dtype}",
            lse_ptr="*fp32",
        )
        return compiler(
            ASTSource(
                _piper_chunked_prefill_kernel,
                sig,
                constexprs=constants,
                attrs=_quantization.compile_pointer_attrs(
                    _piper_chunked_prefill_kernel.arg_names, sig
                ),
            ),
            target,
            {"num_warps": num_warps, "num_stages": 1},
        )

    return (
        _compile_variant(tail_mode=False, causal_mode=False, chunk_map=has_chunk_map),
        _compile_variant(tail_mode=True, causal_mode=is_causal, chunk_map=False),
    )
