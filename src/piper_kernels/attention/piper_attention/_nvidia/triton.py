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
    normalize_window_bounds,
    resolve_query_window,
    validate_lse_buffer,
    validate_output_buffer,
    validate_query_offset,
)
from . import gluon_async_copy as _gluon_async_copy
from . import policy as _policy
from .policy import PiperAttentionExecutionPlan

_BLOCK_N = 64
_TL_CENTER_CHUNK = tl.constexpr(_quantization.CENTER_CHUNK)
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
def _fold_chunk_value_mean(
    numerator,
    chunk_denom,
    value_mean_base,
    offsets_d,
    offsets_vd,
    split_pv_head_dim: tl.constexpr,
    head_dim: tl.constexpr,
    guard_zero: tl.constexpr,
    active_mask=None,
):
    """Fold one K1024 chunk's softmax mass times its V mean into the FP32 numerator."""
    chunk_weight_codes = (chunk_denom * _P_UINT8_RANGE)[:, None]
    zero_guard = ((chunk_denom > 0.0) if active_mask is None else active_mask)[:, None]
    if split_pv_head_dim:
        half_head_dim: tl.constexpr = head_dim // 2
        accumulator_low, accumulator_high = numerator
        value_mean_low = tl.load(value_mean_base + offsets_vd)
        delta_low = chunk_weight_codes * value_mean_low[None, :]
        if guard_zero:
            accumulator_low = accumulator_low + tl.where(zero_guard, delta_low, 0.0)
        else:
            accumulator_low = accumulator_low + delta_low
        value_mean_high = tl.load(value_mean_base + half_head_dim + offsets_vd)
        delta_high = chunk_weight_codes * value_mean_high[None, :]
        if guard_zero:
            accumulator_high = accumulator_high + tl.where(zero_guard, delta_high, 0.0)
        else:
            accumulator_high = accumulator_high + delta_high
        return (accumulator_low, accumulator_high)
    else:
        value_mean = tl.load(value_mean_base + offsets_d)
        delta = chunk_weight_codes * value_mean[None, :]
        if guard_zero:
            delta = tl.where(zero_guard, delta, 0.0)
        return numerator + delta


@triton.jit
def _safe_softmax_probabilities(
    scores,
    safe_block_max,
    value_log_scale,
    active_mask=None,
):
    """Compute clamped base-2 probabilities without exp2 overflow on tiny V scales."""
    score_delta = scores - safe_block_max[:, None]
    max_delta = tl.minimum(-value_log_scale[None, :], 100.0)
    probabilities = tl.exp2(tl.where(score_delta > max_delta, max_delta, score_delta))
    if active_mask is not None:
        probabilities = tl.where(active_mask, probabilities, 0.0)
    return probabilities


@triton.jit
def _quantize_probabilities_u8(
    probabilities,
    value_scale_multiplier,
    use_packed_probability_conversion: tl.constexpr,
    active_mask=None,
):
    """Scale FP32 probabilities by per-key V multipliers and round to UINT8 codes."""
    scaled = probabilities * value_scale_multiplier[None, :]
    if active_mask is not None:
        scaled = tl.where(active_mask, scaled, 0.0)
    probability_values = scaled + 0.5
    if use_packed_probability_conversion:
        return _ptx_float32_to_uint8x4(probability_values)
    else:
        probability_codes = tl.minimum(_P_UINT8_RANGE, probability_values).to(tl.int32)
        return probability_codes.to(tl.uint8)


@triton.jit
def _accumulate_pv_tile(
    numerator,
    probability_uint8,
    value_tile,
    old_weight,
    current_weight,
    split_pv_head_dim: tl.constexpr,
):
    """Update single or split-head-dim FP32 PV accumulators from UINT8xINT8 MMA."""
    if split_pv_head_dim:
        accumulator_low, accumulator_high = numerator
        value_low, value_high = value_tile
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
        return (accumulator_low, accumulator_high)
    else:
        partial = uint8_int8_dot(probability_uint8, value_tile)
        return numerator * old_weight[:, None] + partial.to(tl.float32) * current_weight[:, None]


@triton.jit
def _store_pv_output(
    output_base,
    numerator,
    denominator_code_units,
    offsets_d,
    offsets_vd,
    valid_queries,
    has_any,
    split_pv_head_dim: tl.constexpr,
    head_dim: tl.constexpr,
    mask_empty: tl.constexpr,
):
    """Normalize single or split-head-dim PV accumulators and store output rows."""
    if split_pv_head_dim:
        half_head_dim: tl.constexpr = head_dim // 2
        accumulator_low, accumulator_high = numerator
        output_low = accumulator_low / denominator_code_units
        output_high = accumulator_high / denominator_code_units
        if mask_empty:
            output_low = tl.where(has_any[:, None], output_low, 0.0)
            output_high = tl.where(has_any[:, None], output_high, 0.0)
        tl.store(output_base + offsets_vd[None, :], output_low, mask=valid_queries[:, None])
        tl.store(
            output_base + half_head_dim + offsets_vd[None, :],
            output_high,
            mask=valid_queries[:, None],
        )
    else:
        output = numerator / denominator_code_units
        if mask_empty:
            output = tl.where(has_any[:, None], output, 0.0)
        tl.store(output_base + offsets_d[None, :], output, mask=valid_queries[:, None])


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
    window_left,
    window_right,
    mask_keys: tl.constexpr,
    window_mode: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    derive_value_log_bound: tl.constexpr,
    safe_softmax: tl.constexpr,
    padded_kv: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    """Advance online-softmax state by one key tile.

    FP32 numerators remain in UINT8 probability-code units for every attention
    mode; each K1024 chunk folds its V mean once outside the inner loop.
    """
    is_causal_mode: tl.constexpr = window_mode == "causal"
    is_windowed_mode: tl.constexpr = window_mode == "windowed"
    mask_row_bounds: tl.constexpr = is_causal_mode or is_windowed_mode

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
        scores = integer_scores.to(tl.float32) * (query_scale[:, None] * key_scale[None, :])

    if mask_keys:
        if is_causal_mode:
            valid_keys = (current_n[None, :] < key_length) & (
                current_n[None, :] <= offsets_m[:, None]
            )
        elif is_windowed_mode:
            valid_keys = tl.broadcast_to(current_n[None, :] < key_length, (block_m, block_n))
            if window_left >= 0:
                valid_keys &= (
                    offsets_m[:, None].to(tl.int64) - current_n[None, :].to(tl.int64)
                ) <= window_left.to(tl.int64)
            if window_right >= 0:
                valid_keys &= (
                    current_n[None, :].to(tl.int64) - offsets_m[:, None].to(tl.int64)
                ) <= window_right.to(tl.int64)
        else:
            valid_keys = current_n < key_length
    elif mask_row_bounds:
        valid_keys = tl.full((block_m, block_n), True, dtype=tl.int1)
    else:
        valid_keys = tl.full((block_n,), True, dtype=tl.int1)
    if mask_keys and mask_row_bounds:
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
    if mask_keys and not mask_row_bounds:
        # The loop's final tile always contains at least one real key. Exclude
        # only its padded columns from the shifted maximum and probability sum;
        # masking the full MxN score tile on every iteration is unnecessary.
        value_log_scale = tl.where(valid_keys, value_log_scale, -float("inf"))
    if safe_softmax:
        shifted_scores = tl.where(
            scores != -float("inf"),
            scores + value_log_scale[None, :],
            -float("inf"),
        )
        block_max = tl.where(
            valid_queries,
            tl.reduce(shifted_scores, 1, max_propagate_nan),
            -float("inf"),
        )
        tile_k_bias = tl.where(block_max != -float("inf"), k_bias, 0.0)
        next_max, next_bias, old_weight, current_weight, has_block, safe_block_max = (
            advance_tile_max_safe(
                running_max,
                running_bias,
                block_max,
                tile_k_bias,
                valid_queries,
            )
        )
        probabilities = _safe_softmax_probabilities(
            scores,
            safe_block_max,
            value_log_scale,
            active_mask=(valid_queries & has_block)[:, None] & valid_keys,
        )
        row_sum = tl.where(has_block, tl.sum(probabilities, axis=1) * current_weight, 0.0)
    else:
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
    probability_uint8 = _quantize_probabilities_u8(
        probabilities,
        value_scale_multiplier,  # pyright: ignore[reportPossiblyUnboundVariable]
        use_packed_probability_conversion=use_packed_probability_conversion,
    )

    if split_pv_head_dim:
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
        value_tile = (value_low, value_high)
    else:
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
    numerator = _accumulate_pv_tile(
        numerator,
        probability_uint8,
        value_tile,
        old_weight,
        current_weight,
        split_pv_head_dim=split_pv_head_dim,
    )
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
    window_left,
    window_right,
    mask_keys: tl.constexpr,
    window_mode: tl.constexpr,
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
    safe_softmax: tl.constexpr,
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
            window_left,
            window_right,
            mask_keys=mask_keys,
            window_mode=window_mode,
            grouped_qk=grouped_qk,
            split_pv_head_dim=split_pv_head_dim,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            use_tensor_descriptors=use_tensor_descriptors,
            use_packed_probability_conversion=use_packed_probability_conversion,
            derive_value_log_bound=derive_value_log_bound,
            safe_softmax=safe_softmax,
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
    key_chunks,
    global_rows,
    offsets_n,
    offsets_d,
    offsets_vd,
    valid_queries,
    key_length,
    window_left,
    window_right,
    chunk_idx,
    prefix_start,
    prefix_stop,
    boundary_start,
    boundary_stop,
    window_mode: tl.constexpr,
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
    safe_softmax: tl.constexpr,
    padded_kv: tl.constexpr,
    prefix_mask_keys: tl.constexpr,
    has_boundary: tl.constexpr,
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
        window_left,
        window_right,
        mask_keys=prefix_mask_keys,
        window_mode=window_mode,
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
        safe_softmax=safe_softmax,
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
            window_left,
            window_right,
            mask_keys=tl.constexpr(True),
            window_mode=window_mode,
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
            safe_softmax=safe_softmax,
            padded_kv=padded_kv,
        )
    value_mean_base = (
        value_mean_ptr + (kv_batch_head.to(tl.int64) * key_chunks + chunk_idx) * head_dim
    )
    numerator = _fold_chunk_value_mean(
        numerator,
        chunk_denom,
        value_mean_base,
        offsets_d,
        offsets_vd,
        split_pv_head_dim=split_pv_head_dim,
        head_dim=head_dim,
        guard_zero=safe_softmax,
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
    lse_ptr,
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
    stride_lb,
    stride_lh,
    window_left,
    window_right,
    heads,
    window_mode: tl.constexpr,
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
    store_lse: tl.constexpr,
    unmasked_query_tiles: tl.constexpr,
    use_query_tensor_descriptor: tl.constexpr,
    padded_kv: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    """Evaluate one complete or masked query tile with K1024+V1024 outer chunking."""
    is_causal_mode: tl.constexpr = window_mode == "causal"
    is_windowed_mode: tl.constexpr = window_mode == "windowed"
    safe_softmax: tl.constexpr = store_lse or is_windowed_mode

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
    if is_causal_mode:
        end_n = tl.minimum(key_length, global_query_start + (query_block + 1) * block_m)

    query_tile_start = query_start + query_block * block_m
    chunk_inputs = (
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
    )
    chunk_meta = (
        batch_head,
        query_storage_length,
        query_tile_start,
        kv_batch_head,
        key_chunks,
        global_rows,
        offsets_n,
        offsets_d,
        offsets_vd,
        valid_queries,
        key_length,
        window_left,
        window_right,
    )
    chunk_options = tl.constexpr(
        (
            window_mode,
            grouped_qk,
            split_pv_head_dim,
            unmasked_query_tiles,
            head_dim,
            block_m,
            block_n,
            use_tensor_descriptors,
            use_packed_probability_conversion,
            derive_value_log_bound,
            loop_num_stages,
            loop_licm,
            safe_softmax,
            padded_kv,
        )
    )

    if is_causal_mode and optimize_causal_traversal:
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
                *chunk_inputs,
                numerator,
                denominator,
                running_max,
                running_bias,
                *chunk_meta,
                chunk_idx,
                chunk_start,
                chunk_end,
                chunk_end,
                chunk_end,
                *chunk_options,
                prefix_mask_keys=tl.constexpr(False),
                has_boundary=tl.constexpr(False),
            )
        for chunk_start in tl.range(mask_free_stop, end_n, _TL_CENTER_CHUNK, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            chunk_idx = chunk_start // _TL_CENTER_CHUNK
            chunk_end = tl.minimum(chunk_start + _TL_CENTER_CHUNK, end_n)
            prefix_stop = tl.minimum(chunk_end, prefix_end)
            boundary_start = tl.maximum(chunk_start, prefix_end)
            numerator, denominator, running_max, running_bias = _attention_chunk(
                *chunk_inputs,
                numerator,
                denominator,
                running_max,
                running_bias,
                *chunk_meta,
                chunk_idx,
                chunk_start,
                prefix_stop,
                boundary_start,
                chunk_end,
                *chunk_options,
                prefix_mask_keys=tl.constexpr(False),
                has_boundary=tl.constexpr(True),
            )
    elif is_windowed_mode:
        eff_wl = tl.minimum(
            window_left.to(tl.int64), (global_query_start + query_rows).to(tl.int64)
        )
        eff_wr = tl.minimum(window_right.to(tl.int64), key_length.to(tl.int64))
        q_tile_min = global_query_start + query_block * block_m
        q_tile_max = tl.minimum(q_tile_min + block_m, global_query_start + query_rows) - 1
        vis_k_start = tl.zeros((), dtype=tl.int64)
        if window_left >= 0:
            vis_k_start = tl.maximum(0, q_tile_min.to(tl.int64) - eff_wl)
        vis_k_end = key_length.to(tl.int64)
        if window_right >= 0:
            vis_k_end = tl.minimum(vis_k_end, q_tile_max.to(tl.int64) + eff_wr + 1)
        if vis_k_start < vis_k_end:
            first_tile_start = (vis_k_start.to(tl.int32) // block_n) * block_n
            vis_k_end32 = vis_k_end.to(tl.int32)
            first_chunk_start = (first_tile_start // _TL_CENTER_CHUNK) * _TL_CENTER_CHUNK
            for chunk_start in tl.range(
                first_chunk_start, vis_k_end32, _TL_CENTER_CHUNK, disable_licm=True
            ):  # pyright: ignore[reportGeneralTypeIssues]
                chunk_idx = chunk_start // _TL_CENTER_CHUNK
                chunk_lo = tl.maximum(chunk_start, first_tile_start)
                chunk_hi = tl.minimum(chunk_start + _TL_CENTER_CHUNK, vis_k_end32)
                numerator, denominator, running_max, running_bias = _attention_chunk(
                    *chunk_inputs,
                    numerator,
                    denominator,
                    running_max,
                    running_bias,
                    *chunk_meta,
                    chunk_idx,
                    chunk_lo,
                    chunk_hi,
                    chunk_hi,
                    chunk_hi,
                    *chunk_options,
                    prefix_mask_keys=tl.constexpr(True),
                    has_boundary=tl.constexpr(False),
                )
    else:
        for chunk_start in tl.range(0, end_n, _TL_CENTER_CHUNK, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            chunk_idx = chunk_start // _TL_CENTER_CHUNK
            chunk_end = tl.minimum(chunk_start + _TL_CENTER_CHUNK, end_n)
            numerator, denominator, running_max, running_bias = _attention_chunk(
                *chunk_inputs,
                numerator,
                denominator,
                running_max,
                running_bias,
                *chunk_meta,
                chunk_idx,
                chunk_start,
                chunk_end,
                chunk_end,
                chunk_end,
                *chunk_options,
                prefix_mask_keys=tl.constexpr(not unmasked_key_tiles),
                has_boundary=tl.constexpr(False),
            )

    if safe_softmax:
        has_any = running_max != -float("inf")
        safe_denom = tl.where(has_any & (denominator != 0.0), denominator, 1.0)
        denominator_code_units = safe_denom[:, None] * _P_UINT8_RANGE
        if store_lse:
            lse = tl.where(
                has_any,
                ((running_max + running_bias) + tl.math.log2(safe_denom)) * _LN2,
                -float("inf"),
            )
            tl.store(
                lse_ptr
                + batch.to(tl.int64) * stride_lb
                + head.to(tl.int64) * stride_lh
                + output_rows,
                lse,
                mask=valid_queries,
            )
    else:
        has_any = tl.full((block_m,), True, dtype=tl.int1)
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
    _store_pv_output(
        output_base,
        numerator,
        denominator_code_units,  # pyright: ignore[reportPossiblyUnboundVariable]
        offsets_d,
        offsets_vd,
        valid_queries,
        has_any,
        split_pv_head_dim=split_pv_head_dim,
        head_dim=head_dim,
        mask_empty=safe_softmax,
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
        "stride_lb",
        "stride_lh",
        "window_left",
        "window_right",
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
    window_mode: tl.constexpr,
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
    lse_ptr=None,
    stride_lb=0,
    stride_lh=0,
    window_left=-1,
    window_right=-1,
    store_lse: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
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
    if window_mode == "causal" and optimize_causal_traversal:
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
        lse_ptr,
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
        stride_lb,
        stride_lh,
        window_left,
        window_right,
        heads,
    )
    tile_options = tl.constexpr(
        (
            window_mode,
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
            store_lse,
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
    window_mode: str | None = None,
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
        window_mode=cast(Any, window_mode),
    )


def _prepare_piper_context(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    execution_plan: PiperAttentionExecutionPlan,
    window_left: int | None = None,
    window_right: int | None = None,
) -> _PreparedPiperContext:
    """Prepare validated K/V once, independently of Q and output storage."""
    _batch, _kv_heads, key_length, head_dim = key.shape
    plan = execution_plan
    eff_left, eff_right, window_mode = normalize_window_bounds(is_causal, window_left, window_right)
    if plan.split_pv_head_dim and head_dim != 128:
        raise ValueError("split-PV Piper Attention requires head_dim=128")
    if plan.optimize_causal_traversal and window_mode != "causal":
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
            window_left=eff_left,
            window_right=eff_right,
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
    window_left: int | None = None,
    window_right: int | None = None,
) -> _PreparedPiperAttention:
    """Prepare all operands and allocate output before timed or captured launches."""
    context = _prepare_piper_context(
        key,
        value,
        is_causal=is_causal,
        execution_plan=execution_plan,
        window_left=window_left,
        window_right=window_right,
    )
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
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, heads, _, head_dim = query.shape
    plan = context.execution_plan
    assert plan is not None
    attention_kernel = cast(Any, _piper_attention_kernel)
    eff_left, eff_right, window_mode = normalize_window_bounds(
        context.is_causal, context.window_left, context.window_right
    )
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
            window_mode=window_mode,
            grouped_qk=plan.grouped_qk,
            split_pv_head_dim=plan.split_pv_head_dim,
            aligned_queries=aligned_queries,
            unmasked_key_tiles=(window_mode == "none" and context.key_length % _BLOCK_N == 0),
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
            lse_ptr=lse,
            stride_lb=0 if lse is None else lse.stride(0),
            stride_lh=0 if lse is None else lse.stride(1),
            window_left=-1 if eff_left is None else eff_left,
            window_right=-1 if eff_right is None else eff_right,
            store_lse=lse is not None,
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
    lse: torch.Tensor | None = None,
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
    if lse is not None:
        validate_lse_buffer(
            lse,
            shape=(batch, heads, rows),
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
            window_left=context.window_left,
            window_right=context.window_right,
            lse=lse,
        )
    return _launch_triton_window_into(
        context,
        query,
        output,
        query_start=query_start,
        rows=rows,
        lse=lse,
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
    window_left: int | None = None,
    window_right: int | None = None,
    return_lse: bool = False,
    execution_plan: PiperAttentionExecutionPlan | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run Piper Attention preprocessing and its fused recurrence."""
    if query.shape[0] == 0:
        empty_out = torch.empty_like(query, memory_format=torch.contiguous_format)
        if not return_lse:
            return empty_out
        return empty_out, torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)

    eff_left, eff_right, window_mode = normalize_window_bounds(is_causal, window_left, window_right)
    plan = (
        execution_plan
        if execution_plan is not None
        else default_execution_plan(
            query,
            is_causal,
            window_mode=window_mode,
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
        window_left=eff_left,
        window_right=eff_right,
    )
    lse = (
        torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        if return_lse
        else None
    )
    output = _launch_piper_attention_into(
        prepared.context,
        prepared.query,
        prepared.output,
        lse=lse,
    )
    if return_lse:
        assert lse is not None
        return output, lse
    return output
