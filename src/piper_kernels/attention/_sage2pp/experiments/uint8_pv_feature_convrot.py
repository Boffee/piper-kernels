"""UINT8-equivalent P with feature-rotated signed-INT8 V.

This experiment avoids rotating the quadratic probability matrix.  Instead it
rotates V along the head-feature dimension, evaluates PV in the rotated basis,
and applies the inverse rotation to the output.

The primary path chooses one V scale per key row after rotation. Because that
scale lies on the PV contraction dimension, it must affect each key's
probability contribution:

    V_rot = diag(s_v) @ V_q
    P @ V_rot = (P * s_v) @ V_q.

The default evaluates the same product in log-softmax coordinates. For
``y = score + log(s_v)``:

    sum(exp(score) * s_v * V_q)   sum(exp(y) * V_q)
    -------------------------------- = ---------------------.
           sum(exp(score))           sum(exp(y) / s_v)

This produces the same normalized UINT8 probability operand without an extra
``max(P * s_v)`` reduction for every query/K tile.

The older per-feature scaling mode remains available as a control.

NVIDIA's integer MMA consumes signed INT8 operands through Triton's public dot
interface.  Nonnegative probabilities use all 256 UINT8 codes through the
affine identity

    u @ v = (u - 128) @ v + 128 * sum(v),

where ``u`` is an integer in ``[0, 255]`` and ``u - 128`` is representable as
signed INT8.  The K=64 ``sum(v)`` term is produced during V quantization and
stored in INT16 (its exact range is only ``[-8128, 8128]``). Attention
reconstructs ``128 * sum(v)`` with a left shift and supplies it as the integer
MMA accumulator. This halves correction metadata traffic without changing the
represented product.
"""

# ruff: noqa: ANN001, ANN202, ARG001, PLR0912, PLR0913, PLR0915, PLR0917

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from typing import Any, Literal, cast

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels.attention._convrot_triton import rotate_rows_in_registers
from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.uint4_pv_convrot import (
    _inverse_rotate_output_kernel,
    _prepare_qk,
)

_PV_BLOCK = 64
_P_UINT8_RANGE = tl.constexpr(255.0)
_P_ZERO_POINT = tl.constexpr(128)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)
_RECURRENCE_SCALE_BITS = tl.constexpr(8)
_RECURRENCE_SCALE = tl.constexpr(256.0)
_RECURRENCE_ROUNDING = tl.constexpr(128)
_RECURRENCE_ACCUMULATOR_LIMIT = tl.constexpr(8388607)
_PAIR_SCALE_BITS = tl.constexpr(10)
_PAIR_SCALE = tl.constexpr(1024.0)
_PAIR_ROUNDING = tl.constexpr(512)
_VALUE_MEAN_CHUNK = 1024
_VALUE_MEAN_BLOCK_N = 64
_VALUE_MEAN_BLOCK_D = 64


@triton.jit
def _rescale_int32_recurrence(accumulator, weight):
    """Multiply an INT32 tile by a guarded Q8 row scale."""
    weight_q8 = tl.minimum(
        _RECURRENCE_SCALE,
        weight * _RECURRENCE_SCALE + 0.5,
    ).to(tl.int32)
    accumulator = tl.maximum(
        -_RECURRENCE_ACCUMULATOR_LIMIT,
        tl.minimum(_RECURRENCE_ACCUMULATOR_LIMIT, accumulator),
    )
    return (accumulator * weight_q8[:, None] + _RECURRENCE_ROUNDING) >> _RECURRENCE_SCALE_BITS


@triton.jit
def _rounded_shift_int32_rows(values, shifts):
    """Align row-wise block-floating INT32 values with rounded right shifts."""
    clamped_shifts = tl.minimum(tl.maximum(shifts, 0), 31)
    safe_shifts = tl.maximum(clamped_shifts, 1)
    one = tl.full(shifts.shape, 1, dtype=tl.int32)
    rounding = one << (safe_shifts - 1)
    shifted = (values + rounding[:, None]) >> clamped_shifts[:, None]
    return tl.where(clamped_shifts[:, None] == 0, values, shifted)


@triton.jit
def _merge_int32_exponent_tile(
    accumulator,
    partial,
    running_exponent,
    block_exponent,
    single_shift: tl.constexpr,
):
    """Merge one locally normalized INT32 tile in a shared power-of-two coordinate."""
    next_exponent = tl.maximum(running_exponent, block_exponent)
    if single_shift:
        accumulator_is_dominant = running_exponent >= block_exponent
        dominant = tl.where(
            accumulator_is_dominant[:, None],
            accumulator,
            partial,
        )
        lower = tl.where(
            accumulator_is_dominant[:, None],
            partial,
            accumulator,
        )
        exponent_gap = next_exponent - tl.minimum(running_exponent, block_exponent)
        return dominant + _rounded_shift_int32_rows(lower, exponent_gap)
    accumulator = _rounded_shift_int32_rows(
        accumulator,
        next_exponent - running_exponent,
    )
    partial = _rounded_shift_int32_rows(
        partial,
        next_exponent - block_exponent,
    )
    return accumulator + partial


@triton.jit
def _rescale_int32_pair(values, weight):
    """Apply a row-wise Q10 weight to one bounded K64 INT32 partial."""
    weight_q10 = tl.minimum(_PAIR_SCALE, weight * _PAIR_SCALE + 0.5).to(tl.int32)
    product = values * weight_q10[:, None]
    return tl.where(
        product >= 0,
        (product + _PAIR_ROUNDING) >> _PAIR_SCALE_BITS,
        -((-product + _PAIR_ROUNDING) >> _PAIR_SCALE_BITS),
    )


@triton.jit
def _apply_delayed_fp16_correction(
    value_correction_ptr,
    correction_group_weights,
    correction_group_blocks,
    offsets_vd,
    accumulator_low,
    accumulator_high,
    active_corrections: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Merge one padded K16 correction group into both D64 accumulators."""
    correction_group_slots = tl.arange(0, 16)
    half_head_dim: tl.constexpr = head_dim // 2
    correction_group_low = tl.load(
        value_correction_ptr
        + correction_group_blocks[:, None] * head_dim
        + offsets_vd[None, :],
        mask=correction_group_slots[:, None] < active_corrections,
        other=0.0,
    )
    correction_group_high = tl.load(
        value_correction_ptr
        + correction_group_blocks[:, None] * head_dim
        + half_head_dim
        + offsets_vd[None, :],
        mask=correction_group_slots[:, None] < active_corrections,
        other=0.0,
    )
    accumulator_low = tl.dot(
        correction_group_weights,
        correction_group_low,
        accumulator_low,
        out_dtype=tl.float16,
    )
    accumulator_high = tl.dot(
        correction_group_weights,
        correction_group_high,
        accumulator_high,
        out_dtype=tl.float16,
    )
    return accumulator_low, accumulator_high


@triton.jit
def _value_mean_partial_kernel(
    value_ptr,
    partial_ptr,
    key_length,
    num_chunks,
    stride_vb,
    stride_vh,
    stride_vn,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_n: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """Reduce one sequence chunk into a per-head/per-feature FP32 partial."""
    chunk = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_d = feature_block * block_d + tl.arange(0, block_d)
    offsets_n = tl.arange(0, block_n)
    accumulator = tl.zeros((block_d,), dtype=tl.float32)
    chunk_start = chunk * chunk_n
    for offset in tl.range(0, chunk_n, block_n, disable_licm=True):
        current_n = chunk_start + offset + offsets_n
        values = tl.load(
            value_ptr
            + batch * stride_vb
            + head * stride_vh
            + current_n[:, None] * stride_vn
            + offsets_d[None, :],
            mask=(current_n[:, None] < key_length) & (offsets_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(values, axis=0)
    tl.store(
        partial_ptr + (batch_head * num_chunks + chunk) * head_dim + offsets_d,
        accumulator,
        mask=offsets_d < head_dim,
    )


@triton.jit
def _value_mean_finalize_kernel(
    partial_ptr,
    mean_ptr,
    key_length,
    num_chunks,
    head_dim: tl.constexpr,
    block_chunks: tl.constexpr,
    block_d: tl.constexpr,
):
    """Merge chunk partials into the compact FP32 V mean."""
    batch_head = tl.program_id(0)
    feature_block = tl.program_id(1)
    offsets_c = tl.arange(0, block_chunks)
    offsets_d = feature_block * block_d + tl.arange(0, block_d)
    partials = tl.load(
        partial_ptr
        + (batch_head * num_chunks + offsets_c[:, None]) * head_dim
        + offsets_d[None, :],
        mask=(offsets_c[:, None] < num_chunks) & (offsets_d[None, :] < head_dim),
        other=0.0,
    )
    mean = tl.sum(partials, axis=0) / key_length
    tl.store(
        mean_ptr + batch_head * head_dim + offsets_d,
        mean,
        mask=offsets_d < head_dim,
    )


@triton.jit
def _quantize_value_feature_convrot_int8_kernel(
    value_ptr,
    value_mean_ptr,
    scale_ptr,
    log_scale_ptr,
    inverse_scale_ptr,
    correction_ptr,
    output_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    rotation_group: tl.constexpr,
    value_scale_floor: tl.constexpr,
    store_log_scale: tl.constexpr,
    store_value_scale: tl.constexpr,
    store_probability_multiplier: tl.constexpr,
    store_value_correction: tl.constexpr,
    store_scaled_fp16_correction: tl.constexpr,
    probability_range: tl.constexpr,
    tile_common_log_denominator: tl.constexpr,
    narrow_int8_log_denominator: tl.constexpr,
    scale_forward_log_recurrence: tl.constexpr,
    output_transposed: tl.constexpr,
    center_value: tl.constexpr,
):
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if center_value:
        value_mean = tl.load(value_mean_ptr + (batch * heads + head) * head_dim + offsets_d)
        value = tl.where(mask, value - value_mean[None, :], 0.0)
    value = rotate_rows_in_registers(value, offsets_d, block_n, rotation_group)
    scale = tl.max(tl.abs(value), axis=0) / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = _sage_backend._round_to_int8(
        value / scale[None, :],
        _V_INT8_RANGE,
    )
    scale_block = (batch * heads + head) * tl.cdiv(key_length, block_n) + key_block
    metadata_offsets = scale_block * head_dim + offsets_d
    if store_value_scale:
        stored_scale = tl.where(store_probability_multiplier, scale * probability_range, scale)
        tl.store(scale_ptr + metadata_offsets, stored_scale)
    if store_value_correction:
        value_sum = tl.sum(quantized.to(tl.int32), axis=0)
        if store_scaled_fp16_correction:
            tl.store(correction_ptr + metadata_offsets, value_sum.to(tl.float32) * (1.0 / 512.0))
        else:
            tl.store(correction_ptr + metadata_offsets, value_sum)
    _sage_backend._store_value_tile(
        output_ptr,
        quantized,
        batch,
        head,
        offsets_n,
        offsets_d,
        mask,
        stride_ob,
        stride_oh,
        stride_on,
        output_transposed,
    )


@triton.jit
def _quantize_value_feature_convrot_per_key_int8_kernel(
    value_ptr,
    value_mean_ptr,
    scale_ptr,
    log_scale_ptr,
    inverse_scale_ptr,
    correction_ptr,
    output_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    rotation_group: tl.constexpr,
    value_scale_floor: tl.constexpr,
    store_log_scale: tl.constexpr,
    store_value_scale: tl.constexpr,
    store_probability_multiplier: tl.constexpr,
    store_value_correction: tl.constexpr,
    store_scaled_fp16_correction: tl.constexpr,
    probability_range: tl.constexpr,
    tile_common_log_denominator: tl.constexpr,
    narrow_int8_log_denominator: tl.constexpr,
    scale_forward_log_recurrence: tl.constexpr,
    output_transposed: tl.constexpr,
    center_value: tl.constexpr,
):
    """Rotate features, then use one symmetric INT8 scale per key row."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid_keys = offsets_n < key_length
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid_keys[:, None],
        other=0.0,
    ).to(tl.float32)
    if center_value:
        value_mean = tl.load(value_mean_ptr + (batch * heads + head) * head_dim + offsets_d)
        value = tl.where(
            valid_keys[:, None],
            value - value_mean[None, :],
            0.0,
        )
    value = rotate_rows_in_registers(value, offsets_d, block_n, rotation_group)
    scale = tl.max(tl.abs(value), axis=1) / _V_INT8_RANGE + _SCALE_EPSILON
    if value_scale_floor > 0.0:
        scale = tl.maximum(scale, tl.max(scale, axis=0) * value_scale_floor)
    quantized = _sage_backend._round_to_int8(
        value / scale[:, None],
        _V_INT8_RANGE,
    )
    batch_head = batch * heads + head
    if store_value_scale:
        stored_scale = tl.where(store_probability_multiplier, scale * probability_range, scale)
        tl.store(
            scale_ptr + batch_head * key_length + offsets_n,
            stored_scale,
            mask=valid_keys,
        )
    if store_log_scale:
        log_scale = tl.log2(scale)
        tl.store(
            log_scale_ptr + batch_head * key_length + offsets_n,
            log_scale,
            mask=valid_keys,
        )
        if scale_forward_log_recurrence:
            pass
        elif narrow_int8_log_denominator:
            inverse_scale = tl.where(valid_keys, 1.0 / scale, 0.0)
            inverse_quant_scale = tl.max(inverse_scale, axis=0) / _V_INT8_RANGE + _SCALE_EPSILON
            inverse_int8 = _sage_backend._round_to_int8(
                inverse_scale / inverse_quant_scale,
                _V_INT8_RANGE,
            )
            tl.store(
                inverse_scale_ptr + batch_head * key_length + offsets_n,
                inverse_int8,
                mask=valid_keys,
            )
            tl.store(
                scale_ptr + batch_head * tl.cdiv(key_length, block_n) + key_block,
                inverse_quant_scale,
            )
        elif tile_common_log_denominator:
            valid_count = tl.sum(valid_keys.to(tl.float32), axis=0)
            tile_log_center = tl.sum(
                tl.where(valid_keys, log_scale, 0.0),
                axis=0,
            ) / tl.maximum(valid_count, 1.0)
            tl.store(
                inverse_scale_ptr + batch_head * key_length + key_block * block_n,
                tl.exp2(-tile_log_center),
            )
        else:
            tl.store(
                inverse_scale_ptr + batch_head * key_length + offsets_n,
                1.0 / scale,
                mask=valid_keys,
            )
    if store_value_correction:
        value_sum = tl.sum(quantized.to(tl.int32), axis=0)
        correction_offsets = (
            batch_head * tl.cdiv(key_length, block_n) + key_block
        ) * head_dim + offsets_d
        if store_scaled_fp16_correction:
            tl.store(correction_ptr + correction_offsets, value_sum.to(tl.float32) * (1.0 / 512.0))
        else:
            tl.store(correction_ptr + correction_offsets, value_sum)
    _sage_backend._store_value_tile(
        output_ptr,
        quantized,
        batch,
        head,
        offsets_n,
        offsets_d,
        valid_keys[:, None],
        stride_ob,
        stride_oh,
        stride_on,
        output_transposed,
    )


@triton.jit
def _uint8_pv_feature_convrot_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    value_log_scale_ptr,
    value_inverse_scale_ptr,
    value_correction_ptr,
    value_mean_ptr,
    rotated_output_ptr,
    query_length,
    key_length,
    query_block_offset: tl.constexpr,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    value_scale_per_key: tl.constexpr,
    tile_probability_scale: tl.constexpr,
    log_probability_scale: tl.constexpr,
    shift_log_scores: tl.constexpr,
    omit_log_scale_shift: tl.constexpr,
    weighted_log_denominator: tl.constexpr,
    scale_forward_log_recurrence: tl.constexpr,
    running_max_probability_recurrence: tl.constexpr,
    tile_common_log_denominator: tl.constexpr,
    narrow_int8_log_denominator: tl.constexpr,
    affine_probability: tl.constexpr,
    native_uint8_mma: tl.constexpr,
    integer_output_recurrence: tl.constexpr,
    integer_tile_exponent_recurrence: tl.constexpr,
    single_shift_tile_exponent_recurrence: tl.constexpr,
    predot_exponent_alignment: tl.constexpr,
    dithered_predot_alignment: tl.constexpr,
    immediate_k32_pv_conversion: tl.constexpr,
    lazy_int32_exponent_recurrence: tl.constexpr,
    integer_exponent_headroom: tl.constexpr,
    paired_int32_tiles: tl.constexpr,
    probability_fp16: tl.constexpr,
    fp16_pv_scaling: tl.constexpr,
    factored_pv_scaling: tl.constexpr,
    precomputed_pv_multiplier: tl.constexpr,
    use_pv_scale_descriptor: tl.constexpr,
    omit_pv_scaling: tl.constexpr,
    normalized_fp16_recurrence: tl.constexpr,
    scaled_fp16_numerator: tl.constexpr,
    scaled_fp16_denominator: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    scaled_fp16_correction: tl.constexpr,
    delayed_fp16_correction_group: tl.constexpr,
    unmasked_query_tiles: tl.constexpr,
    unmasked_self_attention: tl.constexpr,
    output_rotation_group: tl.constexpr,
    center_value: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    value_transposed: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
    use_tensor_descriptors: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    query_block = tl.program_id(0) + query_block_offset
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    if affine_probability:
        base_probability_range: tl.constexpr = _P_UINT8_RANGE
    else:
        base_probability_range: tl.constexpr = _V_INT8_RANGE
    if split_pv_head_dim:
        offsets_vd = tl.arange(0, half_head_dim)
    if unmasked_query_tiles:
        valid_queries = tl.full((block_m,), True, dtype=tl.int1)
    else:
        valid_queries = offsets_m < query_length

    query = tl.load(
        query_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr + (batch * heads + head) * tl.cdiv(query_length, 32) + offsets_m // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + (batch * heads + head) * query_length + offsets_m,
            mask=valid_queries,
            other=0.0,
        )

    if split_pv_head_dim:
        if lazy_int32_exponent_recurrence or integer_tile_exponent_recurrence:
            accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.int32)
            accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.int32)
        elif normalized_fp16_recurrence or scaled_fp16_numerator:
            accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
            accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
        else:
            accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
            accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        if paired_int32_tiles:
            pending_low = tl.zeros((block_m, half_head_dim), dtype=tl.int32)
            pending_high = tl.zeros((block_m, half_head_dim), dtype=tl.int32)
    elif (
        integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
    ):
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.int32)
    else:
        if normalized_fp16_recurrence or scaled_fp16_numerator:
            accumulator = tl.zeros((block_m, head_dim), dtype=tl.float16)
        else:
            accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
        if paired_int32_tiles:
            pending = tl.zeros((block_m, head_dim), dtype=tl.int32)
    if scaled_fp16_denominator:
        denominator = tl.zeros((block_m,), dtype=tl.float16)
    else:
        denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    if delayed_fp16_correction_group:
        correction_group_slots = tl.arange(0, 16)
        correction_group_maxima = tl.full(
            (block_m, 16),
            -float("inf"),
            dtype=tl.float32,
        )
    if integer_tile_exponent_recurrence or lazy_int32_exponent_recurrence:
        running_exponent = tl.full((block_m,), -(1 << 30), dtype=tl.int32)
    if paired_int32_tiles:
        pending_block_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, (query_block + 1) * block_m)

    for start_n in tl.range(0, end_n, block_n, disable_licm=True):
        current_n = start_n + offsets_n
        batch_head = batch * heads + head
        key = _sage_backend._load_attention_key_tile(
            key_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores = tl.dot(query, key, out_dtype=tl.int32)
        if grouped_qk:
            key_scale = tl.load(
                key_scale_ptr
                + (batch * heads + head) * tl.cdiv(key_length, block_n)
                + start_n // block_n
            )
            scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
        else:
            key_scale = tl.load(
                key_scale_ptr + (batch * heads + head) * key_length + current_n,
                mask=current_n < key_length,
                other=0.0,
            )
            scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]

        if unmasked_self_attention:
            valid_keys = tl.full((block_m, block_n), True, dtype=tl.int1)
        else:
            valid_keys = current_n[None, :] < key_length
            if is_causal:
                valid_keys &= current_n[None, :] <= offsets_m[:, None]
            scores = tl.where(
                valid_queries[:, None] & valid_keys,
                scores,
                -float("inf"),
            )

        if value_scale_per_key:
            if log_probability_scale:
                # Change softmax coordinates from z to y=z+log2(s_v):
                #   sum(exp(z) s_v Vq) / sum(exp(z))
                # = sum(exp(y) Vq) / sum(exp(y) / s_v).
                # This gives fixed-range UINT8 probabilities without the
                # separate max(P * s_v) reduction used by dynamic mode.
                if shift_log_scores and not omit_log_scale_shift:
                    if unmasked_self_attention:
                        key_value_log_scale = tl.load(
                            value_log_scale_ptr
                            + (batch * heads + head) * key_length
                            + current_n
                        )
                    else:
                        key_value_log_scale = tl.load(
                            value_log_scale_ptr
                            + (batch * heads + head) * key_length
                            + current_n,
                            mask=current_n < key_length,
                            other=0.0,
                        )
                    if scale_forward_log_recurrence:
                        shifted_scores = scores + key_value_log_scale[None, :]
                    else:
                        scores += key_value_log_scale[None, :]
            else:
                key_value_scale = tl.load(
                    value_scale_ptr + (batch * heads + head) * key_length + current_n,
                    mask=current_n < key_length,
                    other=0.0,
                )

        if scale_forward_log_recurrence and not omit_log_scale_shift:
            block_max = tl.max(shifted_scores, axis=1)
        else:
            block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        if paired_int32_tiles:
            first_in_pair = (start_n // block_n) % 2 == 0
            pair_max = tl.maximum(pending_block_max, block_max)
            pending_is_pair_max = pending_block_max >= block_max
            lower_pair_max = tl.minimum(pending_block_max, block_max)
            lower_pair_weight = tl.exp2(lower_pair_max - pair_max)
            pair_output_weight = tl.exp2(pair_max - next_max)
        if unmasked_query_tiles:
            old_weight = tl.exp2(running_max - next_max)
        else:
            old_weight = tl.where(
                valid_queries,
                tl.exp2(running_max - next_max),
                0.0,
            )
        if lazy_int32_exponent_recurrence:
            safe_block_max = tl.where(valid_queries, block_max, 0.0)
            block_exponent = tl.ceil(safe_block_max).to(tl.int32)
            exponent_limit = running_exponent + integer_exponent_headroom
            next_exponent = tl.where(
                block_exponent > exponent_limit,
                block_exponent,
                running_exponent,
            )
            exponent_old_weight = tl.where(
                valid_queries,
                tl.exp2((running_exponent - next_exponent).to(tl.float32)),
                0.0,
            )
            exponent_shift = next_exponent - running_exponent
            if split_pv_head_dim:
                accumulator_low = _rounded_shift_int32_rows(
                    accumulator_low,
                    exponent_shift,
                )
                accumulator_high = _rounded_shift_int32_rows(
                    accumulator_high,
                    exponent_shift,
                )
            else:
                accumulator = _rounded_shift_int32_rows(accumulator, exponent_shift)
            old_weight = exponent_old_weight
            current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - next_exponent.to(tl.float32)[:, None]),
                0.0,
            )
        elif integer_output_recurrence:
            # Keep every PV tile in the running maximum's coordinate system,
            # allowing its integer MMA result to accumulate directly into the
            # persistent INT32 numerator.  Fixed-point rescales the old numerator only
            # when the online-softmax maximum advances.
            current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - next_max[:, None]),
                0.0,
            )
            accumulator = _rescale_int32_recurrence(accumulator, old_weight)
        else:
            if running_max_probability_recurrence:
                current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
                probabilities = tl.where(
                    valid_queries[:, None] & valid_keys,
                    tl.exp2(scores - next_max[:, None]),
                    0.0,
                )
            elif unmasked_query_tiles:
                current_weight = tl.exp2(block_max - next_max)
                probabilities = tl.exp2(scores - block_max[:, None])
            else:
                current_weight = tl.where(
                    valid_queries,
                    tl.exp2(block_max - next_max),
                    0.0,
                )
                probabilities = tl.where(
                    valid_queries[:, None] & valid_keys,
                    tl.exp2(scores - block_max[:, None]),
                    0.0,
                )
            if split_pv_head_dim:
                if (
                    not integer_tile_exponent_recurrence
                    and not normalized_fp16_recurrence
                    and not scaled_fp16_numerator
                ):
                    accumulator_low *= old_weight[:, None]
                    accumulator_high *= old_weight[:, None]
            elif (
                not integer_tile_exponent_recurrence
                and not normalized_fp16_recurrence
                and not scaled_fp16_numerator
            ):
                accumulator *= old_weight[:, None]
        if integer_tile_exponent_recurrence:
            safe_block_max = tl.where(valid_queries, block_max, 0.0)
            block_exponent = tl.ceil(safe_block_max).to(tl.int32)
            if predot_exponent_alignment:
                next_tile_exponent = tl.maximum(running_exponent, block_exponent)
                old_exponent_shift = next_tile_exponent - running_exponent
                if split_pv_head_dim:
                    accumulator_low = _rounded_shift_int32_rows(
                        accumulator_low,
                        old_exponent_shift,
                    )
                    accumulator_high = _rounded_shift_int32_rows(
                        accumulator_high,
                        old_exponent_shift,
                    )
                else:
                    accumulator = _rounded_shift_int32_rows(
                        accumulator,
                        old_exponent_shift,
                    )
        if probability_fp16:
            probabilities = probabilities.to(tl.float16)
        previous_denominator = denominator
        if not narrow_int8_log_denominator:
            if scale_forward_log_recurrence:
                # This is exactly the weighted-denominator log recurrence in its
                # original-score coordinate system:
                #   exp(z + log2(s) - m) / s = exp(z - m).
                # The matching factor s is applied to P immediately before its
                # integer dot below, so P codes and the represented numerator
                # remain unchanged while this reduction becomes unweighted.
                denominator_contribution = tl.sum(probabilities, axis=1)
            elif value_scale_per_key and log_probability_scale and weighted_log_denominator:
                if tile_common_log_denominator:
                    tile_inverse_scale = tl.load(
                        value_inverse_scale_ptr + (batch * heads + head) * key_length + start_n
                    )
                    denominator_contribution = tl.sum(probabilities, axis=1) * tile_inverse_scale
                else:
                    key_value_inverse_scale = tl.load(
                        value_inverse_scale_ptr + (batch * heads + head) * key_length + current_n,
                        mask=current_n < key_length,
                        other=1.0,
                    )
                    denominator_contribution = tl.sum(
                        probabilities * key_value_inverse_scale[None, :],
                        axis=1,
                    )
            else:
                denominator_contribution = tl.sum(probabilities, axis=1)
            if scaled_fp16_denominator:
                # The 2^-4 coordinate bounds a worst-case 131072-token
                # denominator by 8192 while retaining much more absolute
                # resolution than the numerator's 2^-16 coordinate. This is
                # a numerical/performance ablation; long diffuse recurrences
                # still expose FP16 mantissa loss.
                denominator_contribution_scaled = (
                    denominator_contribution * (1.0 / 16.0)
                ).to(tl.float16)
                denominator = (
                    denominator * old_weight.to(tl.float16)
                    + denominator_contribution_scaled
                    * current_weight.to(tl.float16)
                )
            else:
                denominator = (
                    denominator * old_weight
                    + denominator_contribution * current_weight
                )
        if normalized_fp16_recurrence:
            inverse_denominator = 1.0 / tl.maximum(denominator, 1e-30)
            old_output_weight = previous_denominator * old_weight * inverse_denominator
            tile_output_weight = current_weight * inverse_denominator

        metadata_block = (batch * heads + head) * tl.cdiv(key_length, block_n) + start_n // block_n
        metadata_offsets = metadata_block * head_dim + offsets_d
        if affine_probability:
            if lazy_int32_exponent_recurrence:
                probability_range: tl.constexpr = _P_UINT8_RANGE / (
                    1 << integer_exponent_headroom
                )
                probability_code_limit: tl.constexpr = _P_UINT8_RANGE
            else:
                probability_range: tl.constexpr = _P_UINT8_RANGE
                probability_code_limit: tl.constexpr = probability_range
            if not native_uint8_mma and not split_pv_head_dim:
                value_correction = (
                    tl.load(value_correction_ptr + metadata_offsets).to(tl.int32) << 7
                )
        else:
            probability_range: tl.constexpr = _V_INT8_RANGE
            probability_code_limit: tl.constexpr = probability_range
        if value_scale_per_key:
            if log_probability_scale:
                if scale_forward_log_recurrence and not omit_pv_scaling:
                    # Scale-forward paths consume s_v only while forming the
                    # PV operand, after the FP32 denominator is complete.
                    if use_pv_scale_descriptor:
                        key_value_scale_for_pv = value_scale_ptr.load(
                            [batch * heads + head, start_n]
                        ).reshape((block_n,))
                    elif unmasked_self_attention:
                        key_value_scale_for_pv = tl.load(
                            value_scale_ptr
                            + (batch * heads + head) * key_length
                            + current_n
                        )
                    else:
                        key_value_scale_for_pv = tl.load(
                            value_scale_ptr
                            + (batch * heads + head) * key_length
                            + current_n,
                            mask=current_n < key_length,
                            other=0.0,
                        )
                if integer_tile_exponent_recurrence:
                    probability_for_dot = (
                        probabilities * tl.exp2(block_max - block_exponent.to(tl.float32))[:, None]
                    )
                    if scale_forward_log_recurrence and not omit_pv_scaling:
                        if fp16_pv_scaling:
                            probability_for_dot = (
                                probability_for_dot.to(tl.float16)
                                * key_value_scale_for_pv[None, :]
                            )
                        elif factored_pv_scaling:
                            if precomputed_pv_multiplier:
                                pv_probability_multiplier = key_value_scale_for_pv
                            else:
                                pv_probability_multiplier = (
                                    key_value_scale_for_pv * probability_range
                                )
                        else:
                            probability_for_dot *= key_value_scale_for_pv[None, :]
                elif scale_forward_log_recurrence:
                    if omit_pv_scaling:
                        probability_for_dot = probabilities
                    elif fp16_pv_scaling:
                        probability_for_dot = (
                            probabilities.to(tl.float16) * key_value_scale_for_pv[None, :]
                        )
                    elif factored_pv_scaling:
                        probability_for_dot = probabilities
                        if precomputed_pv_multiplier:
                            pv_probability_multiplier = key_value_scale_for_pv
                        else:
                            pv_probability_multiplier = key_value_scale_for_pv * probability_range
                    else:
                        probability_for_dot = probabilities * key_value_scale_for_pv[None, :]
                else:
                    probability_for_dot = probabilities
                probability_quant_scale = tl.full(
                    (block_m,),
                    1.0 / probability_range,
                    dtype=tl.float32,
                )
                probability_output_scale = probability_quant_scale
            elif tile_probability_scale:
                # A single scale for the whole K tile avoids a separate
                # block_m-wide max reduction.  Dividing V's per-key scales by
                # their tile maximum keeps the UINT8 operand in [0, 255].
                tile_value_scale = tl.max(key_value_scale, axis=0) + 1e-30
                probability_for_dot = probabilities * (key_value_scale / tile_value_scale)[None, :]
                probability_quant_scale = tl.full(
                    (block_m,),
                    1.0 / probability_range,
                    dtype=tl.float32,
                )
                probability_output_scale = tl.full(
                    (block_m,),
                    tile_value_scale / probability_range,
                    dtype=tl.float32,
                )
            else:
                probability_for_dot = probabilities * key_value_scale[None, :]
                probability_quant_scale = (
                    tl.max(probability_for_dot, axis=1) / probability_range + 1e-30
                )
                probability_output_scale = probability_quant_scale
        else:
            probability_for_dot = probabilities
            probability_quant_scale = tl.full(
                (block_m,),
                1.0 / probability_range,
                dtype=tl.float32,
            )
            probability_output_scale = probability_quant_scale

        # Encode UINT8 probabilities as signed INT8 for IMMA.  With per-key V
        # scales, the quantity represented here is P[k] * scale_V[k], which
        # folds the scale on the contraction dimension into the left operand.
        # Invalid and
        # masked probabilities are zero, hence encode as -128 and are exactly
        # cancelled by the tile-wide +128 * sum(Vq) correction below.
        if factored_pv_scaling:
            # Factor the fixed UINT8 range out of the MxK product: compute
            # s_v[k] * 255 once per K entry rather than multiplying every
            # probability by s_v[k] and then by 255.
            probability_code_values = probability_for_dot * pv_probability_multiplier[None, :]
        elif fp16_pv_scaling:
            # Isolate reduced precision to the PV-side scale and UINT8 code
            # formation. The online softmax and denominator above remain FP32.
            probability_code_values = probability_for_dot * probability_range
        else:
            probability_code_values = probability_for_dot / probability_quant_scale[:, None]
        probability_codes = tl.minimum(
            probability_code_limit,
            probability_code_values + 0.5,
        ).to(tl.int32)
        if predot_exponent_alignment:
            aligned_probability_codes = probability_codes.to(tl.float32) * tl.exp2(
                (block_exponent - next_tile_exponent).to(tl.float32)
            )[:, None]
            if dithered_predot_alignment:
                # One stratified threshold per key is enough: real-model quality
                # matches per-(query,key) phases while this stays a K-vector.
                phase = (current_n * 151 + batch_head * 199) & 1023
                probability_rounding = (phase.to(tl.float32) + 0.5) * (1.0 / 1024.0)
            else:
                probability_rounding = tl.full(
                    (block_m, block_n),
                    0.5,
                    dtype=tl.float32,
                )
            probability_codes = (
                aligned_probability_codes + probability_rounding
            ).to(tl.int32)
        if native_uint8_mma:
            probability_operand = probability_codes.to(tl.uint8)
        elif affine_probability:
            probability_int8 = (probability_codes - _P_ZERO_POINT).to(tl.int8)
        else:
            probability_int8 = probability_codes.to(tl.int8)
        if immediate_k32_pv_conversion:
            probability_pairs = probability_operand.reshape(
                (block_m, 2, block_n // 2)
            ).permute((0, 2, 1))
            probability_operand0, probability_operand1 = probability_pairs.split()
        if narrow_int8_log_denominator:
            inverse_scale_int8 = tl.load(
                value_inverse_scale_ptr + batch_head * key_length + current_n,
                mask=current_n < key_length,
                other=0,
            )
            inverse_scale_matrix = tl.broadcast_to(
                inverse_scale_int8[:, None],
                (block_n, 16),
            )
            if native_uint8_mma:
                denominator_partial = tl.dot(
                    probability_operand,
                    inverse_scale_matrix,
                    out_dtype=tl.int32,
                )
            elif affine_probability:
                denominator_correction = _P_ZERO_POINT * tl.sum(
                    inverse_scale_int8.to(tl.int32),
                    axis=0,
                )
                denominator_accumulator = (
                    tl.zeros((block_m, 16), dtype=tl.int32) + denominator_correction
                )
                denominator_partial = tl.dot(
                    probability_int8,
                    inverse_scale_matrix,
                    denominator_accumulator,
                    out_dtype=tl.int32,
                )
            else:
                denominator_partial = tl.dot(
                    probability_int8,
                    inverse_scale_matrix,
                    out_dtype=tl.int32,
                )
            inverse_quant_scale = tl.load(
                value_scale_ptr + batch_head * tl.cdiv(key_length, block_n) + start_n // block_n
            )
            denominator_contribution = (
                tl.sum(denominator_partial, axis=1).to(tl.float32)
                * (1.0 / 16.0)
                * probability_quant_scale
                * inverse_quant_scale
            )
            denominator = denominator * old_weight + denominator_contribution * current_weight
        if split_pv_head_dim:
            if immediate_k32_pv_conversion:
                offsets_n32 = tl.arange(0, block_n // 2)
                current_n32_0 = start_n + offsets_n32
                current_n32_1 = start_n + block_n // 2 + offsets_n32
                value_low0 = _sage_backend._load_attention_value_subtile(
                    value_ptr,
                    batch,
                    head,
                    batch_head,
                    start_n,
                    current_n32_0,
                    offsets_vd,
                    key_length,
                    feature_start=0,
                    feature_block=half_head_dim,
                    value_transposed=value_transposed,
                    use_tensor_descriptors=use_tensor_descriptors,
                    heads=heads,
                    head_dim=head_dim,
                    block_n=block_n // 2,
                )
                value_low1 = _sage_backend._load_attention_value_subtile(
                    value_ptr,
                    batch,
                    head,
                    batch_head,
                    start_n + block_n // 2,
                    current_n32_1,
                    offsets_vd,
                    key_length,
                    feature_start=0,
                    feature_block=half_head_dim,
                    value_transposed=value_transposed,
                    use_tensor_descriptors=use_tensor_descriptors,
                    heads=heads,
                    head_dim=head_dim,
                    block_n=block_n // 2,
                )
            else:
                value_low = _sage_backend._load_attention_value_subtile(
                    value_ptr,
                    batch,
                    head,
                    batch_head,
                    start_n,
                    current_n,
                    offsets_vd,
                    key_length,
                    feature_start=0,
                    feature_block=half_head_dim,
                    value_transposed=value_transposed,
                    use_tensor_descriptors=use_tensor_descriptors,
                    heads=heads,
                    head_dim=head_dim,
                    block_n=block_n,
                )
            if native_uint8_mma:
                if immediate_k32_pv_conversion:
                    partial_low0 = tl.dot(
                        probability_operand0,
                        value_low0,
                        out_dtype=tl.int32,
                    )
                    accumulator_low += (
                        partial_low0.to(tl.float32)
                        * probability_output_scale[:, None]
                        * current_weight[:, None]
                    )
                    partial_low1 = tl.dot(
                        probability_operand1,
                        value_low1,
                        out_dtype=tl.int32,
                    )
                    accumulator_low += (
                        partial_low1.to(tl.float32)
                        * probability_output_scale[:, None]
                        * current_weight[:, None]
                    )
                elif lazy_int32_exponent_recurrence or predot_exponent_alignment:
                    accumulator_low = tl.dot(
                        probability_operand,
                        value_low,
                        accumulator_low,
                        out_dtype=tl.int32,
                    )
                else:
                    partial_low = tl.dot(
                        probability_operand,
                        value_low,
                        out_dtype=tl.int32,
                    )
            elif affine_probability:
                if scaled_fp16_correction:
                    if not delayed_fp16_correction_group:
                        correction_low_scaled = tl.load(
                            value_correction_ptr + metadata_block * head_dim + offsets_vd
                        )
                    partial_low = tl.dot(
                        probability_int8,
                        value_low,
                        out_dtype=tl.int32,
                    )
                else:
                    correction_low = tl.load(
                        value_correction_ptr + metadata_block * head_dim + offsets_vd
                    ).to(tl.int32) << 7
                    correction_accumulator_low = (
                        tl.zeros((block_m, half_head_dim), dtype=tl.int32)
                        + correction_low[None, :]
                    )
                    partial_low = tl.dot(
                        probability_int8,
                        value_low,
                        correction_accumulator_low,
                        out_dtype=tl.int32,
                    )
            else:
                partial_low = tl.dot(
                    probability_int8,
                    value_low,
                    out_dtype=tl.int32,
                )
            if (
                immediate_k32_pv_conversion
                or lazy_int32_exponent_recurrence
                or predot_exponent_alignment
            ):
                pass
            elif integer_tile_exponent_recurrence:
                accumulator_low = _merge_int32_exponent_tile(
                    accumulator_low,
                    partial_low,
                    running_exponent,
                    block_exponent,
                    single_shift_tile_exponent_recurrence,
                )
            elif paired_int32_tiles:
                dominant_low = tl.where(
                    pending_is_pair_max[:, None],
                    pending_low,
                    partial_low,
                )
                lower_low = tl.where(
                    pending_is_pair_max[:, None],
                    partial_low,
                    pending_low,
                )
                paired_low = dominant_low + _rescale_int32_pair(lower_low, lower_pair_weight)
                accumulator_low += tl.where(
                    first_in_pair,
                    0.0,
                    paired_low.to(tl.float32)
                    * probability_output_scale[:, None]
                    * pair_output_weight[:, None],
                )
                pending_low = tl.where(first_in_pair, partial_low, pending_low)
            elif normalized_fp16_recurrence:
                accumulator_low = (
                    accumulator_low.to(tl.float32) * old_output_weight[:, None]
                    + partial_low.to(tl.float32)
                    * probability_output_scale[:, None]
                    * tile_output_weight[:, None]
                ).to(tl.float16)
            elif scaled_fp16_numerator:
                # Keep the unnormalized online-softmax numerator in a fixed
                # 2^-16 code-space coordinate. P's per-key V scale is already
                # folded into the UINT8 operand, so no scale-coordinate
                # transition is needed here. Apply the matching 255 / 2^16
                # factor to the row denominator in the epilogue instead of
                # applying 1 / 255 to every K64 partial.
                partial_low_scaled = (
                    partial_low.to(tl.float32) * (1.0 / 65536.0)
                ).to(tl.float16)
                if scaled_fp16_correction and not delayed_fp16_correction_group:
                    partial_low_scaled += correction_low_scaled[None, :]
                accumulator_low = (
                    accumulator_low * old_weight[:, None].to(tl.float16)
                    + partial_low_scaled
                    * current_weight[:, None].to(tl.float16)
                )
            elif running_max_probability_recurrence:
                accumulator_low += partial_low.to(tl.float32)
            else:
                accumulator_low += (
                    partial_low.to(tl.float32)
                    * probability_output_scale[:, None]
                    * current_weight[:, None]
                )

            if immediate_k32_pv_conversion:
                value_high0 = _sage_backend._load_attention_value_subtile(
                    value_ptr,
                    batch,
                    head,
                    batch_head,
                    start_n,
                    current_n32_0,
                    offsets_vd,
                    key_length,
                    feature_start=half_head_dim,
                    feature_block=half_head_dim,
                    value_transposed=value_transposed,
                    use_tensor_descriptors=use_tensor_descriptors,
                    heads=heads,
                    head_dim=head_dim,
                    block_n=block_n // 2,
                )
                value_high1 = _sage_backend._load_attention_value_subtile(
                    value_ptr,
                    batch,
                    head,
                    batch_head,
                    start_n + block_n // 2,
                    current_n32_1,
                    offsets_vd,
                    key_length,
                    feature_start=half_head_dim,
                    feature_block=half_head_dim,
                    value_transposed=value_transposed,
                    use_tensor_descriptors=use_tensor_descriptors,
                    heads=heads,
                    head_dim=head_dim,
                    block_n=block_n // 2,
                )
            else:
                value_high = _sage_backend._load_attention_value_subtile(
                    value_ptr,
                    batch,
                    head,
                    batch_head,
                    start_n,
                    current_n,
                    offsets_vd,
                    key_length,
                    feature_start=half_head_dim,
                    feature_block=half_head_dim,
                    value_transposed=value_transposed,
                    use_tensor_descriptors=use_tensor_descriptors,
                    heads=heads,
                    head_dim=head_dim,
                    block_n=block_n,
                )
            if native_uint8_mma:
                if immediate_k32_pv_conversion:
                    partial_high0 = tl.dot(
                        probability_operand0,
                        value_high0,
                        out_dtype=tl.int32,
                    )
                    accumulator_high += (
                        partial_high0.to(tl.float32)
                        * probability_output_scale[:, None]
                        * current_weight[:, None]
                    )
                    partial_high1 = tl.dot(
                        probability_operand1,
                        value_high1,
                        out_dtype=tl.int32,
                    )
                    accumulator_high += (
                        partial_high1.to(tl.float32)
                        * probability_output_scale[:, None]
                        * current_weight[:, None]
                    )
                elif lazy_int32_exponent_recurrence or predot_exponent_alignment:
                    accumulator_high = tl.dot(
                        probability_operand,
                        value_high,
                        accumulator_high,
                        out_dtype=tl.int32,
                    )
                else:
                    partial_high = tl.dot(
                        probability_operand,
                        value_high,
                        out_dtype=tl.int32,
                    )
            elif affine_probability:
                if scaled_fp16_correction:
                    if not delayed_fp16_correction_group:
                        correction_high_scaled = tl.load(
                            value_correction_ptr
                            + metadata_block * head_dim
                            + half_head_dim
                            + offsets_vd
                        )
                    partial_high = tl.dot(
                        probability_int8,
                        value_high,
                        out_dtype=tl.int32,
                    )
                else:
                    correction_high = tl.load(
                        value_correction_ptr
                        + metadata_block * head_dim
                        + half_head_dim
                        + offsets_vd
                    ).to(tl.int32) << 7
                    correction_accumulator_high = (
                        tl.zeros((block_m, half_head_dim), dtype=tl.int32)
                        + correction_high[None, :]
                    )
                    partial_high = tl.dot(
                        probability_int8,
                        value_high,
                        correction_accumulator_high,
                        out_dtype=tl.int32,
                    )
            else:
                partial_high = tl.dot(
                    probability_int8,
                    value_high,
                    out_dtype=tl.int32,
                )
            if (
                immediate_k32_pv_conversion
                or lazy_int32_exponent_recurrence
                or predot_exponent_alignment
            ):
                pass
            elif integer_tile_exponent_recurrence:
                accumulator_high = _merge_int32_exponent_tile(
                    accumulator_high,
                    partial_high,
                    running_exponent,
                    block_exponent,
                    single_shift_tile_exponent_recurrence,
                )
            elif paired_int32_tiles:
                dominant_high = tl.where(
                    pending_is_pair_max[:, None],
                    pending_high,
                    partial_high,
                )
                lower_high = tl.where(
                    pending_is_pair_max[:, None],
                    partial_high,
                    pending_high,
                )
                paired_high = dominant_high + _rescale_int32_pair(
                    lower_high,
                    lower_pair_weight,
                )
                accumulator_high += tl.where(
                    first_in_pair,
                    0.0,
                    paired_high.to(tl.float32)
                    * probability_output_scale[:, None]
                    * pair_output_weight[:, None],
                )
                pending_high = tl.where(first_in_pair, partial_high, pending_high)
            elif normalized_fp16_recurrence:
                accumulator_high = (
                    accumulator_high.to(tl.float32) * old_output_weight[:, None]
                    + partial_high.to(tl.float32)
                    * probability_output_scale[:, None]
                    * tile_output_weight[:, None]
                ).to(tl.float16)
            elif scaled_fp16_numerator:
                partial_high_scaled = (
                    partial_high.to(tl.float32) * (1.0 / 65536.0)
                ).to(tl.float16)
                if scaled_fp16_correction and not delayed_fp16_correction_group:
                    partial_high_scaled += correction_high_scaled[None, :]
                accumulator_high = (
                    accumulator_high * old_weight[:, None].to(tl.float16)
                    + partial_high_scaled
                    * current_weight[:, None].to(tl.float16)
                )
            elif running_max_probability_recurrence:
                accumulator_high += partial_high.to(tl.float32)
            else:
                accumulator_high += (
                    partial_high.to(tl.float32)
                    * probability_output_scale[:, None]
                    * current_weight[:, None]
                )
        else:
            value = _sage_backend._load_attention_value_tile(
                value_ptr,
                batch,
                head,
                batch_head,
                start_n,
                current_n,
                offsets_d,
                key_length,
                value_transposed,
                use_tensor_descriptors,
                heads,
                head_dim,
                block_n,
            )
            if native_uint8_mma:
                if (
                    integer_output_recurrence
                    or lazy_int32_exponent_recurrence
                    or predot_exponent_alignment
                ):
                    corrected_int32 = tl.dot(
                        probability_operand,
                        value,
                        accumulator,
                        out_dtype=tl.int32,
                    )
                else:
                    corrected_int32 = tl.dot(
                        probability_operand,
                        value,
                        out_dtype=tl.int32,
                    )
            elif affine_probability:
                correction_accumulator = (
                    tl.zeros(
                        (block_m, head_dim),
                        dtype=tl.int32,
                    )
                    + value_correction[None, :]
                )
                if (
                    integer_output_recurrence
                    or lazy_int32_exponent_recurrence
                    or predot_exponent_alignment
                ):
                    correction_accumulator += accumulator
                corrected_int32 = tl.dot(
                    probability_int8,
                    value,
                    correction_accumulator,
                    out_dtype=tl.int32,
                )
            elif (
                integer_output_recurrence
                or lazy_int32_exponent_recurrence
                or predot_exponent_alignment
            ):
                corrected_int32 = tl.dot(
                    probability_int8,
                    value,
                    accumulator,
                    out_dtype=tl.int32,
                )
            else:
                corrected_int32 = tl.dot(
                    probability_int8,
                    value,
                    out_dtype=tl.int32,
                )
            if (
                integer_output_recurrence
                or lazy_int32_exponent_recurrence
                or predot_exponent_alignment
            ):
                accumulator = corrected_int32
            elif integer_tile_exponent_recurrence:
                accumulator = _merge_int32_exponent_tile(
                    accumulator,
                    corrected_int32,
                    running_exponent,
                    block_exponent,
                    single_shift_tile_exponent_recurrence,
                )
            elif value_scale_per_key:
                if paired_int32_tiles:
                    dominant_partial = tl.where(
                        pending_is_pair_max[:, None],
                        pending,
                        corrected_int32,
                    )
                    lower_partial = tl.where(
                        pending_is_pair_max[:, None],
                        corrected_int32,
                        pending,
                    )
                    paired_partial = dominant_partial + _rescale_int32_pair(
                        lower_partial,
                        lower_pair_weight,
                    )
                    accumulator += tl.where(
                        first_in_pair,
                        0.0,
                        paired_partial.to(tl.float32)
                        * probability_output_scale[:, None]
                        * pair_output_weight[:, None],
                    )
                    pending = tl.where(first_in_pair, corrected_int32, pending)
                elif normalized_fp16_recurrence:
                    accumulator = (
                        accumulator.to(tl.float32) * old_output_weight[:, None]
                        + corrected_int32.to(tl.float32)
                        * probability_output_scale[:, None]
                        * tile_output_weight[:, None]
                    ).to(tl.float16)
                elif scaled_fp16_numerator:
                    partial_scaled = (
                        corrected_int32.to(tl.float32) * (1.0 / 65536.0)
                    ).to(tl.float16)
                    accumulator = (
                        accumulator * old_weight[:, None].to(tl.float16)
                        + partial_scaled
                        * current_weight[:, None].to(tl.float16)
                    )
                elif running_max_probability_recurrence:
                    accumulator += corrected_int32.to(tl.float32)
                else:
                    accumulator += (
                        corrected_int32.to(tl.float32)
                        * probability_output_scale[:, None]
                        * current_weight[:, None]
                    )
            else:
                value_scale = tl.load(value_scale_ptr + metadata_offsets)
                accumulator += (
                    corrected_int32.to(tl.float32)
                    * probability_output_scale[:, None]
                    * value_scale[None, :]
                    * current_weight[:, None]
                )
        if delayed_fp16_correction_group:
            correction_group_slot = (start_n // block_n) % delayed_fp16_correction_group
            correction_group_maxima = tl.where(
                correction_group_slots[None, :] == correction_group_slot,
                block_max[:, None],
                correction_group_maxima,
            )
            if correction_group_slot == delayed_fp16_correction_group - 1:
                correction_group_weights = tl.where(
                    correction_group_slots[None, :] < delayed_fp16_correction_group,
                    tl.exp2(correction_group_maxima - next_max[:, None]),
                    0.0,
                ).to(tl.float16)
                correction_group_first_block = (
                    start_n // block_n - (delayed_fp16_correction_group - 1)
                )
                correction_group_blocks = (
                    (batch * heads + head) * tl.cdiv(key_length, block_n)
                    + correction_group_first_block
                    + correction_group_slots
                )
                accumulator_low, accumulator_high = _apply_delayed_fp16_correction(
                    value_correction_ptr,
                    correction_group_weights,
                    correction_group_blocks,
                    offsets_vd,
                    accumulator_low,
                    accumulator_high,
                    active_corrections=delayed_fp16_correction_group,
                    head_dim=head_dim,
                )
        if paired_int32_tiles:
            pending_block_max = tl.where(first_in_pair, block_max, pending_block_max)
        if lazy_int32_exponent_recurrence:
            running_exponent = next_exponent
        if integer_tile_exponent_recurrence:
            if predot_exponent_alignment:
                running_exponent = next_tile_exponent
            else:
                running_exponent = tl.maximum(running_exponent, block_exponent)
        running_max = next_max

    if split_pv_head_dim:
        if paired_int32_tiles:
            has_pending = (tl.cdiv(end_n, block_n) % 2) == 1
            pending_weight = tl.exp2(pending_block_max - running_max)
            accumulator_low += tl.where(
                has_pending,
                pending_low.to(tl.float32) * (1.0 / _P_UINT8_RANGE) * pending_weight[:, None],
                0.0,
            )
            accumulator_high += tl.where(
                has_pending,
                pending_high.to(tl.float32) * (1.0 / _P_UINT8_RANGE) * pending_weight[:, None],
                0.0,
            )
        if lazy_int32_exponent_recurrence:
            recurrence_scale: tl.constexpr = (1 << integer_exponent_headroom) / _P_UINT8_RANGE
            denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
            output_low = accumulator_low.to(tl.float32) * recurrence_scale / denominator_safe
            output_high = accumulator_high.to(tl.float32) * recurrence_scale / denominator_safe
        elif integer_tile_exponent_recurrence:
            recurrence_scale = (
                (1.0 / _P_UINT8_RANGE)
                * tl.exp2(running_exponent.to(tl.float32) - running_max)
            )
            denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
            output_low = (
                accumulator_low.to(tl.float32)
                * recurrence_scale[:, None]
                / denominator_safe
            )
            output_high = (
                accumulator_high.to(tl.float32)
                * recurrence_scale[:, None]
                / denominator_safe
            )
        elif normalized_fp16_recurrence:
            output_low = accumulator_low.to(tl.float32)
            output_high = accumulator_high.to(tl.float32)
        elif scaled_fp16_numerator:
            denominator_safe = tl.maximum(
                denominator.to(tl.float32), 1e-30
            )[:, None]
            if scaled_fp16_denominator:
                output_low = accumulator_low.to(tl.float32) / (
                    denominator_safe * (base_probability_range / 4096.0)
                )
                output_high = accumulator_high.to(tl.float32) / (
                    denominator_safe * (base_probability_range / 4096.0)
                )
            else:
                denominator_code_scale: tl.constexpr = (
                    base_probability_range / 65536.0
                )
                output_low = accumulator_low.to(tl.float32) / (
                    denominator_safe * denominator_code_scale
                )
                output_high = accumulator_high.to(tl.float32) / (
                    denominator_safe * denominator_code_scale
                )
        else:
            denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
            output_low = accumulator_low / denominator_safe
            output_high = accumulator_high / denominator_safe
        if running_max_probability_recurrence:
            if affine_probability:
                delayed_probability_scale: tl.constexpr = 1.0 / _P_UINT8_RANGE
            else:
                delayed_probability_scale: tl.constexpr = 1.0 / _V_INT8_RANGE
            output_low *= delayed_probability_scale
            output_high *= delayed_probability_scale
        if center_value:
            value_mean_base = value_mean_ptr + (batch * heads + head) * head_dim
            output_low += tl.load(value_mean_base + offsets_vd)[None, :]
            output_high += tl.load(value_mean_base + half_head_dim + offsets_vd)[None, :]
        output_base = (
            rotated_output_ptr
            + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        )
        tl.store(
            output_base + offsets_vd[None, :],
            output_low,
            mask=valid_queries[:, None],
        )
        tl.store(
            output_base + half_head_dim + offsets_vd[None, :],
            output_high,
            mask=valid_queries[:, None],
        )
    elif (
        integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
    ):
        if affine_probability:
            if lazy_int32_exponent_recurrence:
                recurrence_scale: tl.constexpr = (
                    1 << integer_exponent_headroom
                ) / _P_UINT8_RANGE
            else:
                recurrence_scale: tl.constexpr = 1.0 / _P_UINT8_RANGE
        else:
            recurrence_scale: tl.constexpr = 1.0 / _V_INT8_RANGE
        numerator = accumulator.to(tl.float32) * recurrence_scale
        if integer_tile_exponent_recurrence:
            numerator *= tl.exp2(running_exponent.to(tl.float32) - running_max)[:, None]
        rotated_output = numerator / tl.maximum(denominator, 1e-30)[:, None]
    elif not split_pv_head_dim:
        if paired_int32_tiles:
            has_pending = (tl.cdiv(end_n, block_n) % 2) == 1
            pending_weight = tl.exp2(pending_block_max - running_max)
            accumulator += tl.where(
                has_pending,
                pending.to(tl.float32) * (1.0 / _P_UINT8_RANGE) * pending_weight[:, None],
                0.0,
            )
        if normalized_fp16_recurrence:
            rotated_output = accumulator.to(tl.float32)
        elif scaled_fp16_numerator:
            denominator_safe = tl.maximum(
                denominator.to(tl.float32), 1e-30
            )[:, None]
            if scaled_fp16_denominator:
                rotated_output = accumulator.to(tl.float32) / (
                    denominator_safe * (_P_UINT8_RANGE / 4096.0)
                )
            else:
                denominator_code_scale: tl.constexpr = (
                    _P_UINT8_RANGE / 65536.0
                )
                rotated_output = accumulator.to(tl.float32) / (
                    denominator_safe * denominator_code_scale
                )
        else:
            rotated_output = accumulator / tl.maximum(denominator, 1e-30)[:, None]
        if running_max_probability_recurrence:
            if affine_probability:
                delayed_probability_scale: tl.constexpr = 1.0 / _P_UINT8_RANGE
            else:
                delayed_probability_scale: tl.constexpr = 1.0 / _V_INT8_RANGE
            rotated_output *= delayed_probability_scale
    if not split_pv_head_dim:
        rotated_output = rotate_rows_in_registers(
            rotated_output,
            offsets_d,
            block_m,
            output_rotation_group,
        )
        if center_value:
            rotated_output += tl.load(
                value_mean_ptr + (batch * heads + head) * head_dim + offsets_d
            )[None, :]
        tl.store(
            rotated_output_ptr
            + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
            + offsets_d[None, :],
            rotated_output,
            mask=valid_queries[:, None],
        )


@triton.jit
def _uint8_pv_scale_forward_bulk_tail_tile(
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_ptr,
    value_log_scale_ptr,
    query,
    query_scale,
    accumulator_low,
    accumulator_high,
    denominator,
    running_max,
    batch_head,
    start_n,
    key_length,
    tail: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """Update the selected scale-forward recurrence for one full or tail K tile."""
    offsets_n = tl.arange(0, block_n)
    current_n = start_n + offsets_n
    key = key_ptr.load([batch_head, start_n, 0]).reshape((block_n, head_dim)).T
    integer_scores = tl.dot(query, key, out_dtype=tl.int32)
    key_scale = tl.load(
        key_scale_ptr
        + batch_head * tl.cdiv(key_length, block_n)
        + start_n // block_n
    )
    scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
    if tail:
        valid_keys = current_n < key_length
        scores = tl.where(valid_keys[None, :], scores, -float("inf"))
        value_log_scale = tl.load(
            value_log_scale_ptr + batch_head * key_length + current_n,
            mask=valid_keys,
            other=0.0,
        )
    else:
        value_log_scale = tl.load(
            value_log_scale_ptr + batch_head * key_length + current_n
        )
    shifted_scores = scores + value_log_scale[None, :]
    block_max = tl.max(shifted_scores, axis=1)
    next_max = tl.maximum(running_max, block_max)
    old_weight = tl.exp2(running_max - next_max)
    current_weight = tl.exp2(block_max - next_max)
    probabilities = tl.exp2(scores - block_max[:, None])
    if tail:
        probabilities = tl.where(valid_keys[None, :], probabilities, 0.0)

    accumulator_low *= old_weight[:, None]
    accumulator_high *= old_weight[:, None]
    denominator = denominator * old_weight + tl.sum(probabilities, axis=1) * current_weight
    if tail:
        pv_multiplier = tl.load(
            value_scale_ptr + batch_head * key_length + current_n,
            mask=valid_keys,
            other=0.0,
        )
    else:
        pv_multiplier = tl.load(
            value_scale_ptr + batch_head * key_length + current_n
        )
    probability_codes = tl.minimum(
        _P_UINT8_RANGE,
        probabilities * pv_multiplier[None, :] + 0.5,
    ).to(tl.uint8)
    value_low = value_ptr.load([batch_head, 0, start_n]).reshape(
        (head_dim // 2, block_n)
    ).T
    value_high = value_ptr.load([batch_head, head_dim // 2, start_n]).reshape(
        (head_dim // 2, block_n)
    ).T
    partial_low = tl.dot(probability_codes, value_low, out_dtype=tl.int32)
    partial_high = tl.dot(probability_codes, value_high, out_dtype=tl.int32)
    output_scale: tl.constexpr = 1.0 / _P_UINT8_RANGE
    accumulator_low += (
        partial_low.to(tl.float32)
        * output_scale
        * current_weight[:, None]
    )
    accumulator_high += (
        partial_high.to(tl.float32)
        * output_scale
        * current_weight[:, None]
    )
    return accumulator_low, accumulator_high, denominator, next_max


@triton.jit
def _uint8_pv_scale_forward_bulk_tail_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    value_log_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    has_key_tail: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """Selected SM120 recurrence with a predicate-free K bulk and one masked tail."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_d = tl.arange(0, head_dim)
    offsets_vd = tl.arange(0, head_dim // 2)
    query = tl.load(
        query_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :]
    )
    query_scale = tl.load(
        query_scale_ptr
        + batch_head * tl.cdiv(query_length, 32)
        + offsets_m // 32
    )
    accumulator_low = tl.zeros((block_m, head_dim // 2), dtype=tl.float32)
    accumulator_high = tl.zeros((block_m, head_dim // 2), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    full_key_length = key_length - key_length % block_n
    for start_n in tl.range(0, full_key_length, block_n, disable_licm=True):
        accumulator_low, accumulator_high, denominator, running_max = (
            _uint8_pv_scale_forward_bulk_tail_tile(
                key_ptr,
                value_ptr,
                key_scale_ptr,
                value_scale_ptr,
                value_log_scale_ptr,
                query,
                query_scale,
                accumulator_low,
                accumulator_high,
                denominator,
                running_max,
                batch_head,
                start_n,
                key_length,
                tail=False,
                heads=heads,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
            )
        )
    if has_key_tail:
        accumulator_low, accumulator_high, denominator, running_max = (
            _uint8_pv_scale_forward_bulk_tail_tile(
                key_ptr,
                value_ptr,
                key_scale_ptr,
                value_scale_ptr,
                value_log_scale_ptr,
                query,
                query_scale,
                accumulator_low,
                accumulator_high,
                denominator,
                running_max,
                batch_head,
                full_key_length,
                key_length,
                tail=True,
                heads=heads,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
            )
        )

    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    output_base = output_ptr + (batch_head * query_length + offsets_m[:, None]) * head_dim
    tl.store(
        output_base + offsets_vd[None, :],
        accumulator_low / denominator_safe,
    )
    tl.store(
        output_base + head_dim // 2 + offsets_vd[None, :],
        accumulator_high / denominator_safe,
    )


def _launch_uint8_pv_scale_forward_bulk_tail_attention(
    prepared: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    storage_key_length: int,
    *,
    block_m: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    maxnreg: int | None = 168,
) -> torch.Tensor:
    """Launch the profiler-only selected recurrence over complete query blocks."""
    query, key, value, query_scale, key_scale, value_scale, value_log_scale, *_ = prepared
    batch, heads, _, head_dim = query.shape
    if (
        head_dim != 128
        or storage_key_length < key_length
        or storage_key_length % _PV_BLOCK
    ):
        raise ValueError("bulk-tail profiling requires padded D128 K/V storage")
    key_argument, value_argument = _sage_backend._make_attention_arguments(
        key,
        value,
        batch,
        heads,
        storage_key_length,
        head_dim,
        True,
        True,
        head_dim // 2,
    )
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    kernel = cast(Any, _uint8_pv_scale_forward_bulk_tail_kernel)
    kernel[(query_length // block_m, heads, batch)](
        query,
        key_argument,
        value_argument,
        query_scale,
        key_scale,
        value_scale,
        value_log_scale,
        output,
        query_length,
        key_length,
        has_key_tail=key_length % _PV_BLOCK != 0,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=_PV_BLOCK,
        **launch_options,
    )
    return output


def _prepare_uint8_pv_feature_convrot_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    rotation_group: int,
    value_scale_axis: Literal["feature", "key"],
    value_scale_floor: float,
    probability_scale_mode: Literal["dynamic", "tile", "log"],
    value_transposed: bool = True,
    affine_probability: bool = True,
    native_uint8_mma: bool = False,
    scaled_fp16_correction: bool = False,
    tile_common_log_denominator: bool = False,
    narrow_int8_log_denominator: bool = False,
    scale_forward_log_recurrence: bool = False,
    fp32_scale_forward_metadata: bool = False,
    precompute_pv_multiplier: bool = False,
    storage_key_length: int | None = None,
    center_value: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Quantize canonical Q/K and feature-rotated block-scaled INT8 V."""
    if rotation_group not in (0, 16, 64):
        raise ValueError(f"rotation group must be 0, 16, or 64, got {rotation_group}")
    head_dim = value.shape[-1]
    if rotation_group and head_dim % rotation_group:
        raise ValueError(
            f"head dimension {head_dim} must be divisible by rotation group {rotation_group}"
        )
    if center_value and rotation_group:
        raise ValueError("centered V currently requires rotation_group=0")
    if value_scale_axis not in ("feature", "key"):
        raise ValueError(f"value scale axis must be 'feature' or 'key', got {value_scale_axis!r}")
    if not 0.0 <= value_scale_floor <= 1.0:
        raise ValueError(f"value scale floor must be in [0, 1], got {value_scale_floor}")
    if value_scale_floor and value_scale_axis != "key":
        raise ValueError("value scale flooring requires per-key value scales")
    if tile_common_log_denominator and narrow_int8_log_denominator:
        raise ValueError("select only one approximate log denominator")
    if scale_forward_log_recurrence and (
        value_scale_axis != "key"
        or probability_scale_mode != "log"
        or tile_common_log_denominator
        or narrow_int8_log_denominator
    ):
        raise ValueError("scale-forward recurrence requires exact per-key log scaling")
    if fp32_scale_forward_metadata and not scale_forward_log_recurrence:
        raise ValueError("FP32 scale metadata requires scale-forward recurrence")
    if precompute_pv_multiplier and not scale_forward_log_recurrence:
        raise ValueError("precomputed PV multiplier requires scale-forward recurrence")
    key_length = key.shape[2]
    if storage_key_length is None:
        storage_key_length = key_length
    if storage_key_length < key_length:
        raise ValueError("storage key length must cover the semantic key length")
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        scale,
        grouped_qk,
        storage_key_length,
    )
    batch, heads, _, _ = value.shape
    if center_value:
        mean_chunks = int(triton.cdiv(key_length, _VALUE_MEAN_CHUNK))
        value_mean_partials = torch.empty(
            (batch, heads, mean_chunks, head_dim),
            device=value.device,
            dtype=torch.float32,
        )
        value_mean = torch.empty(
            (batch, heads, head_dim),
            device=value.device,
            dtype=torch.float32,
        )
        _value_mean_partial_kernel[
            (
                mean_chunks,
                int(triton.cdiv(head_dim, _VALUE_MEAN_BLOCK_D)),
                batch * heads,
            )
        ](
            value,
            value_mean_partials,
            key_length,
            mean_chunks,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            heads=heads,
            head_dim=head_dim,
            chunk_n=_VALUE_MEAN_CHUNK,
            block_n=_VALUE_MEAN_BLOCK_N,
            block_d=_VALUE_MEAN_BLOCK_D,
            num_warps=4,
        )
        _value_mean_finalize_kernel[
            (batch * heads, int(triton.cdiv(head_dim, _VALUE_MEAN_BLOCK_D)))
        ](
            value_mean_partials,
            value_mean,
            key_length,
            mean_chunks,
            head_dim=head_dim,
            block_chunks=triton.next_power_of_2(mean_chunks),
            block_d=_VALUE_MEAN_BLOCK_D,
            num_warps=4,
        )
    else:
        value_mean = value
    value_blocks = (key_length + _PV_BLOCK - 1) // _PV_BLOCK
    correction_shape = (batch, heads, value_blocks, head_dim)
    scale_shape = (batch, heads, key_length) if value_scale_axis == "key" else correction_shape
    if value_scale_axis == "key" and probability_scale_mode == "log":
        value_log_scale = torch.empty(scale_shape, device=value.device, dtype=torch.float16)
        if scale_forward_log_recurrence:
            value_scale = torch.empty(
                scale_shape,
                device=value.device,
                dtype=torch.float32 if fp32_scale_forward_metadata else torch.float16,
            )
            value_inverse_scale = value_scale
        elif narrow_int8_log_denominator:
            value_inverse_scale = torch.empty(scale_shape, device=value.device, dtype=torch.int8)
            value_scale = torch.empty(
                (batch, heads, value_blocks),
                device=value.device,
                dtype=torch.float16,
            )
        else:
            value_inverse_scale = torch.empty(scale_shape, device=value.device, dtype=torch.float16)
            value_scale = value_inverse_scale
    else:
        value_scale = torch.empty(scale_shape, device=value.device, dtype=torch.float32)
        value_log_scale = value_scale
        value_inverse_scale = value_scale
    value_correction = torch.empty(
        correction_shape if affine_probability and not native_uint8_mma else (1,),
        device=value.device,
        dtype=torch.float16 if scaled_fp16_correction else torch.int16,
    )
    value_int8_shape = (
        (batch, heads, head_dim, storage_key_length)
        if value_transposed
        else (batch, heads, storage_key_length, head_dim)
    )
    value_int8 = (
        torch.zeros(value_int8_shape, device=value.device, dtype=torch.int8)
        if storage_key_length != key_length
        else torch.empty(value_int8_shape, device=value.device, dtype=torch.int8)
    )
    quantize_kernel = (
        _quantize_value_feature_convrot_per_key_int8_kernel
        if value_scale_axis == "key"
        else _quantize_value_feature_convrot_int8_kernel
    )
    quantize_kernel[(value_blocks, heads, batch)](
        value,
        value_mean,
        value_scale,
        value_log_scale,
        value_inverse_scale,
        value_correction,
        value_int8,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_int8.stride(0),
        value_int8.stride(1),
        value_int8.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=_PV_BLOCK,
        rotation_group=rotation_group,
        value_scale_floor=value_scale_floor,
        store_log_scale=probability_scale_mode == "log",
        store_value_scale=(
            probability_scale_mode != "log"
            or value_scale_axis == "feature"
            or scale_forward_log_recurrence
        ),
        store_probability_multiplier=precompute_pv_multiplier,
        store_value_correction=affine_probability and not native_uint8_mma,
        store_scaled_fp16_correction=scaled_fp16_correction,
        probability_range=(
            255.0 if affine_probability or native_uint8_mma else 127.0
        ),
        tile_common_log_denominator=tile_common_log_denominator,
        narrow_int8_log_denominator=narrow_int8_log_denominator,
        scale_forward_log_recurrence=scale_forward_log_recurrence,
        output_transposed=value_transposed,
        center_value=center_value,
        num_warps=4,
    )
    return (
        query_int8,
        key_int8,
        value_int8,
        query_scale,
        key_scale,
        value_scale,
        value_log_scale,
        value_inverse_scale,
        value_correction,
        value_mean,
    )


def _launch_uint8_pv_feature_convrot_attention(
    prepared: tuple[torch.Tensor, ...],
    rotated_output: torch.Tensor,
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    is_causal: bool,
    *,
    grouped_qk: bool,
    rotation_group: int,
    value_scale_axis: Literal["feature", "key"],
    probability_scale_mode: Literal["dynamic", "tile", "log"],
    fuse_output_rotation: bool,
    block_m: int,
    num_warps: int,
    num_stages: int,
    value_transposed: bool = True,
    shift_log_scores: bool = True,
    omit_log_scale_shift: bool = False,
    weighted_log_denominator: bool = True,
    scale_forward_log_recurrence: bool = False,
    running_max_probability_recurrence: bool = False,
    tile_common_log_denominator: bool = False,
    narrow_int8_log_denominator: bool = False,
    affine_probability: bool = True,
    native_uint8_mma: bool = False,
    integer_output_recurrence: bool = False,
    integer_tile_exponent_recurrence: bool = False,
    single_shift_tile_exponent_recurrence: bool = False,
    predot_exponent_alignment: bool = False,
    dithered_predot_alignment: bool = False,
    immediate_k32_pv_conversion: bool = False,
    lazy_int32_exponent_recurrence: bool = False,
    integer_exponent_headroom: int = 0,
    paired_int32_tiles: bool = False,
    probability_fp16: bool = False,
    fp16_pv_scaling: bool = False,
    factored_pv_scaling: bool = False,
    precomputed_pv_multiplier: bool = False,
    use_pv_scale_descriptor: bool = False,
    omit_pv_scaling: bool = False,
    normalized_fp16_recurrence: bool = False,
    scaled_fp16_numerator: bool = False,
    scaled_fp16_correction: bool = False,
    delayed_fp16_correction_group: int = 0,
    scaled_fp16_denominator: bool = False,
    split_pv_head_dim: bool = False,
    unmasked_self_attention: bool = False,
    split_query_tail: bool = False,
    tail_query_only: bool = False,
    use_tensor_descriptors: bool = False,
    storage_key_length: int | None = None,
    maxnreg: int | None = None,
    center_value: bool = False,
) -> torch.Tensor:
    """Launch prequantized attention followed by its feature inverse rotation."""
    (
        query,
        key,
        value,
        query_scale,
        key_scale,
        value_scale,
        value_log_scale,
        value_inverse_scale,
        value_correction,
        value_mean,
    ) = prepared
    batch, heads, _, head_dim = query.shape
    if storage_key_length is None:
        storage_key_length = key_length
    if storage_key_length < key_length:
        raise ValueError("storage key length must cover the semantic key length")
    if storage_key_length != key_length and not use_tensor_descriptors:
        raise ValueError("padded key storage currently requires tensor descriptors")
    if unmasked_self_attention and (
        is_causal
        or query_length != key_length
        or query_length % block_m
        or key_length % _PV_BLOCK
    ):
        raise ValueError(
            "unmasked per-key attention requires noncausal self-attention with complete M/K tiles"
        )
    if split_query_tail and (is_causal or unmasked_self_attention):
        raise ValueError(
            "split query-tail attention requires masked-key noncausal attention"
        )
    if tail_query_only and (
        is_causal
        or unmasked_self_attention
        or split_query_tail
        or query_length % block_m == 0
    ):
        raise ValueError("tail-only attention requires one partial noncausal query tile")
    if probability_scale_mode not in ("dynamic", "tile", "log"):
        raise ValueError(
            "probability scale mode must be 'dynamic', 'tile', or 'log', "
            f"got {probability_scale_mode!r}"
        )
    if probability_scale_mode != "dynamic" and value_scale_axis != "key":
        raise ValueError(f"{probability_scale_mode} probability scaling requires per-key scales")
    if not shift_log_scores and (
        probability_scale_mode != "log" or weighted_log_denominator
    ):
        raise ValueError("unshifted log scores require log scaling with an unweighted denominator")
    if scale_forward_log_recurrence and (
        value_scale_axis != "key"
        or probability_scale_mode != "log"
        or not shift_log_scores
        or not weighted_log_denominator
        or running_max_probability_recurrence
        or integer_output_recurrence
        or lazy_int32_exponent_recurrence
        or probability_fp16
        or normalized_fp16_recurrence
        or tile_common_log_denominator
        or narrow_int8_log_denominator
    ):
        raise ValueError(
            "scale-forward recurrence requires compatible per-key log recurrence"
        )
    if omit_log_scale_shift and (
        not scale_forward_log_recurrence
        or value_scale_axis != "key"
        or probability_scale_mode != "log"
    ):
        raise ValueError("omitting the log-scale shift requires per-key scale-forward recurrence")
    if fp16_pv_scaling and (
        not scale_forward_log_recurrence
        or not affine_probability
        or not native_uint8_mma
        or probability_fp16
    ):
        raise ValueError(
            "FP16 PV scaling requires native UINT8 scale-forward recurrence "
            "with FP32 softmax"
        )
    if factored_pv_scaling and (
        not scale_forward_log_recurrence
        or fp16_pv_scaling
    ):
        raise ValueError(
            "factored PV scaling requires integer-P scale-forward recurrence"
        )
    if precomputed_pv_multiplier and not factored_pv_scaling:
        raise ValueError("precomputed PV multiplier requires factored PV scaling")
    if use_pv_scale_descriptor and (
        not precomputed_pv_multiplier or not use_tensor_descriptors
    ):
        raise ValueError(
            "PV scale descriptor requires a precomputed multiplier and tensor descriptors"
        )
    if omit_pv_scaling and (
        not scale_forward_log_recurrence
        or not affine_probability
        or not native_uint8_mma
        or fp16_pv_scaling
        or factored_pv_scaling
    ):
        raise ValueError(
            "omitted PV scaling is a native UINT8 scale-forward perf control"
        )
    if running_max_probability_recurrence and (
        value_scale_axis != "key"
        or probability_scale_mode != "log"
        or integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
        or paired_int32_tiles
        or probability_fp16
        or normalized_fp16_recurrence
        or tile_common_log_denominator
        or narrow_int8_log_denominator
    ):
        raise ValueError(
            "running-max P requires per-key log scaling with the exact FP32 recurrence"
        )
    if tile_common_log_denominator and (
        value_scale_axis != "key" or probability_scale_mode != "log" or not weighted_log_denominator
    ):
        raise ValueError(
            "tile-common denominator requires weighted per-key log probability scaling"
        )
    if narrow_int8_log_denominator and (
        value_scale_axis != "key"
        or probability_scale_mode != "log"
        or not weighted_log_denominator
        or integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
        or paired_int32_tiles
        or normalized_fp16_recurrence
    ):
        raise ValueError(
            "narrow INT8 denominator requires weighted per-key log scaling and FP32 recurrence"
        )
    if tile_common_log_denominator and narrow_int8_log_denominator:
        raise ValueError("select only one approximate log denominator")
    if (
        integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
    ) and (
        value_scale_axis != "key" or probability_scale_mode != "log"
    ):
        raise ValueError("INT32 output recurrence requires per-key log probability scaling")
    if sum(
        (
            integer_output_recurrence,
            integer_tile_exponent_recurrence,
            lazy_int32_exponent_recurrence,
        )
    ) > 1:
        raise ValueError("select only one INT32 output recurrence")
    if single_shift_tile_exponent_recurrence and not integer_tile_exponent_recurrence:
        raise ValueError("single-shift alignment requires tile-exponent recurrence")
    if predot_exponent_alignment and (
        not integer_tile_exponent_recurrence
        or not affine_probability
        or not native_uint8_mma
    ):
        raise ValueError(
            "pre-dot exponent alignment requires affine native-UINT8 tile-exponent recurrence"
        )
    if dithered_predot_alignment and not predot_exponent_alignment:
        raise ValueError("dithered alignment requires pre-dot exponent alignment")
    if immediate_k32_pv_conversion and (
        not scale_forward_log_recurrence
        or not affine_probability
        or not native_uint8_mma
        or not split_pv_head_dim
        or integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
        or paired_int32_tiles
        or probability_fp16
        or normalized_fp16_recurrence
        or running_max_probability_recurrence
    ):
        raise ValueError(
            "immediate K32 PV conversion requires native-UINT8 split-D128 "
            "scale-forward FP32 recurrence"
        )
    if lazy_int32_exponent_recurrence and (
        not affine_probability
        or not native_uint8_mma
        or not split_pv_head_dim
        or is_causal
        or not shift_log_scores
        or not weighted_log_denominator
        or paired_int32_tiles
        or probability_fp16
        or normalized_fp16_recurrence
        or tile_common_log_denominator
        or narrow_int8_log_denominator
        or running_max_probability_recurrence
    ):
        raise ValueError(
            "lazy INT32 exponent recurrence requires noncausal native UINT8 split-D128 "
            "with exact per-key log scaling"
        )
    if integer_exponent_headroom not in (0, 1, 2):
        raise ValueError("integer exponent headroom must be 0, 1, or 2 bits")
    if integer_exponent_headroom and not lazy_int32_exponent_recurrence:
        raise ValueError("integer exponent headroom requires lazy INT32 exponent recurrence")
    if paired_int32_tiles and (
        value_scale_axis != "key"
        or probability_scale_mode != "log"
        or not affine_probability
        or integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
    ):
        raise ValueError(
            "paired INT32 tiles require affine/native UINT8, per-key log scaling, "
            "and FP32 recurrence"
        )
    if normalized_fp16_recurrence and (
        value_scale_axis != "key"
        or probability_scale_mode != "log"
        or integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
        or paired_int32_tiles
    ):
        raise ValueError(
            "normalized FP16 recurrence requires per-key log scaling without an INT32 recurrence"
        )
    if scaled_fp16_numerator and (
        value_scale_axis != "key"
        or probability_scale_mode != "log"
        or not shift_log_scores
        or not weighted_log_denominator
        or head_dim != 128
        or rotation_group != 0
        or normalized_fp16_recurrence
        or integer_output_recurrence
        or integer_tile_exponent_recurrence
        or lazy_int32_exponent_recurrence
        or paired_int32_tiles
        or probability_fp16
        or running_max_probability_recurrence
        or immediate_k32_pv_conversion
        or tile_common_log_denominator
        or key_length > 131072
    ):
        raise ValueError(
            "scaled FP16 numerator requires a compatible integer-P D128 "
            "per-key log recurrence with K <= 131072"
        )
    if scaled_fp16_denominator and not scaled_fp16_numerator:
        raise ValueError("scaled FP16 denominator requires the scaled FP16 numerator")
    if split_pv_head_dim and (
        head_dim != 128
        or value_scale_axis != "key"
        or probability_scale_mode != "log"
        or rotation_group != 0
        or integer_output_recurrence
    ):
        raise ValueError(
            "split PV requires D128, rotation-free per-key log scaling, and compatible recurrence"
        )
    if scaled_fp16_correction and (
        native_uint8_mma
        or not affine_probability
        or not split_pv_head_dim
        or not scaled_fp16_numerator
    ):
        raise ValueError(
            "scaled FP16 correction requires affine signed-MMA emulation "
            "with a split scaled-FP16 numerator"
        )
    if delayed_fp16_correction_group not in (0, 8, 16):
        raise ValueError("delayed FP16 correction group must be 0, 8, or 16")
    if delayed_fp16_correction_group and (
        not scaled_fp16_correction
        or not unmasked_self_attention
        or key_length % (delayed_fp16_correction_group * _PV_BLOCK)
    ):
        raise ValueError(
            "delayed correction requires complete noncausal self-attention groups "
            "with the scaled FP16 correction path"
        )
    key_argument, value_argument = _sage_backend._make_attention_arguments(
        key,
        value,
        batch,
        heads,
        storage_key_length,
        head_dim,
        value_transposed,
        use_tensor_descriptors,
        head_dim // 2 if split_pv_head_dim else None,
    )
    if immediate_k32_pv_conversion and use_tensor_descriptors:
        value_argument = TensorDescriptor(
            value,
            shape=[batch * heads, head_dim, key_length],
            strides=[head_dim * key_length, key_length, 1],
            block_shape=[1, head_dim // 2, _PV_BLOCK // 2],
        )
    value_scale_argument: torch.Tensor | TensorDescriptor = value_scale
    if use_pv_scale_descriptor:
        value_scale_argument = TensorDescriptor(
            value_scale,
            shape=[batch * heads, key_length],
            strides=[key_length, 1],
            block_shape=[1, _PV_BLOCK],
        )
    attention_output = output if rotation_group == 0 or fuse_output_rotation else rotated_output
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    attention_kernel = cast(Any, _uint8_pv_feature_convrot_attention_kernel)

    def launch_attention(
        query_blocks: int,
        query_block_offset: int,
        unmasked_query_tiles: bool,
    ) -> None:
        attention_kernel[(query_blocks, heads, batch)](
            query,
            key_argument,
            value_argument,
            query_scale,
            key_scale,
            value_scale_argument,
            value_log_scale,
            value_inverse_scale,
            value_correction,
            value_mean,
            attention_output,
            query_length,
            key_length,
            query_block_offset=query_block_offset,
            is_causal=is_causal,
            grouped_qk=grouped_qk,
            value_scale_per_key=value_scale_axis == "key",
            tile_probability_scale=probability_scale_mode == "tile",
            log_probability_scale=probability_scale_mode == "log",
            shift_log_scores=shift_log_scores,
            omit_log_scale_shift=omit_log_scale_shift,
            weighted_log_denominator=weighted_log_denominator,
            scale_forward_log_recurrence=scale_forward_log_recurrence,
            running_max_probability_recurrence=running_max_probability_recurrence,
            tile_common_log_denominator=tile_common_log_denominator,
            narrow_int8_log_denominator=narrow_int8_log_denominator,
            affine_probability=affine_probability,
            native_uint8_mma=native_uint8_mma,
            integer_output_recurrence=integer_output_recurrence,
            integer_tile_exponent_recurrence=integer_tile_exponent_recurrence,
            single_shift_tile_exponent_recurrence=single_shift_tile_exponent_recurrence,
            predot_exponent_alignment=predot_exponent_alignment,
            dithered_predot_alignment=dithered_predot_alignment,
            immediate_k32_pv_conversion=immediate_k32_pv_conversion,
            lazy_int32_exponent_recurrence=lazy_int32_exponent_recurrence,
            integer_exponent_headroom=integer_exponent_headroom,
            paired_int32_tiles=paired_int32_tiles,
            probability_fp16=probability_fp16,
            fp16_pv_scaling=fp16_pv_scaling,
            factored_pv_scaling=factored_pv_scaling,
            precomputed_pv_multiplier=precomputed_pv_multiplier,
            use_pv_scale_descriptor=use_pv_scale_descriptor,
            omit_pv_scaling=omit_pv_scaling,
            normalized_fp16_recurrence=normalized_fp16_recurrence,
            scaled_fp16_numerator=scaled_fp16_numerator,
            scaled_fp16_denominator=scaled_fp16_denominator,
            split_pv_head_dim=split_pv_head_dim,
            scaled_fp16_correction=scaled_fp16_correction,
            delayed_fp16_correction_group=delayed_fp16_correction_group,
            unmasked_query_tiles=unmasked_query_tiles,
            unmasked_self_attention=unmasked_self_attention,
            output_rotation_group=rotation_group if fuse_output_rotation else 0,
            center_value=center_value,
            heads=heads,
            head_dim=head_dim,
            block_m=block_m,
            block_n=_PV_BLOCK,
            value_transposed=value_transposed,
            use_tensor_descriptors=use_tensor_descriptors,
            **launch_options,
        )

    if tail_query_only:
        launch_attention(1, query_length // block_m, False)
    elif split_query_tail:
        full_query_blocks = query_length // block_m
        if full_query_blocks:
            launch_attention(full_query_blocks, 0, True)
        if query_length % block_m:
            launch_attention(1, full_query_blocks, False)
    else:
        launch_attention(
            int(triton.cdiv(query_length, block_m)),
            0,
            unmasked_self_attention,
        )
    if rotation_group == 0 or fuse_output_rotation:
        return output
    rows = batch * heads * query_length
    _inverse_rotate_output_kernel[(rows,)](
        rotated_output,
        output,
        head_dim=head_dim,
        rotation_group=rotation_group,
        num_warps=4,
    )
    return output


def triton_sage_attention_uint8_pv_feature_convrot(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int = 0,
    value_scale_axis: Literal["feature", "key"] = "key",
    probability_scale_mode: Literal["dynamic", "tile", "log"] | None = None,
    value_scale_floor: float = 0.0,
    fuse_output_rotation: bool = True,
    grouped_qk: bool | None = None,
    affine_probability: bool = True,
    native_uint8_mma: bool = False,
    integer_output_recurrence: bool = False,
    integer_tile_exponent_recurrence: bool = False,
    predot_exponent_alignment: bool = False,
    dithered_predot_alignment: bool = False,
    paired_int32_tiles: bool = False,
    probability_fp16: bool = False,
    normalized_fp16_recurrence: bool = False,
    scaled_fp16_numerator: bool = False,
    scaled_fp16_correction: bool = False,
    delayed_fp16_correction_group: int = 0,
    scaled_fp16_denominator: bool = False,
    split_pv_head_dim: bool = False,
    tile_common_log_denominator: bool = False,
    narrow_int8_log_denominator: bool = False,
    running_max_probability_recurrence: bool = False,
    scale_forward_log_recurrence: bool | None = None,
    optimize_pv_scaling: bool | None = None,
    fp32_pv_scale_metadata: bool | None = None,
    center_value: bool | None = None,
) -> torch.Tensor:
    """Run per-key-scaled feature-ConvRot V with integer P attention.

    Log-domain scaling is algebraically equivalent to dynamic ``P * scale_v`` normalization
    while avoiding its second per-query reduction. ``probability_scale_mode="tile"`` and
    ``value_scale_floor`` remain quality/performance ablations.  The algebraically equivalent
    scale-forward recurrence is selected for measured noncausal SM120 D128 exact-log shapes
    from N=1024; an explicit boolean controls the experiment. ``optimize_pv_scaling`` stores
    the final ``probability_range * scale_v`` multiplier during V preparation, removing its
    formation from the attention loop. The scaled-FP16 numerator uses FP16 multiplier metadata
    by default; ``fp32_pv_scale_metadata`` retains the older FP32 control. ``center_value``
    computes a compact FP32 sequence mean, subtracts it inside V quantization, and restores it
    in the attention epilogue without materializing a centered V tensor. It is selected by
    default for long noncausal SM12x per-key log scaling; pass an explicit boolean to override
    that policy.
    """
    if grouped_qk is None:
        grouped_qk = torch.cuda.get_device_capability(query.device)[0] == 12
    if probability_scale_mode is None:
        probability_scale_mode = "log" if value_scale_axis == "key" else "dynamic"
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    if center_value is None:
        center_value = (
            torch.cuda.get_device_capability(query.device)[0] == 12
            and not is_causal
            and rotation_group == 0
            and value_scale_axis == "key"
            and probability_scale_mode == "log"
            and head_dim == 128
            and query_length >= 1024
            and key_length >= 1024
        )
    if center_value and rotation_group:
        raise ValueError("centered V currently requires rotation_group=0")
    if scale_forward_log_recurrence is None:
        scale_forward_log_recurrence = (
            torch.cuda.get_device_capability(query.device)[0] == 12
            and not is_causal
            and value_scale_axis == "key"
            and probability_scale_mode == "log"
            and rotation_group == 0
            and head_dim == 128
            and query_length >= 1024
            and key_length >= 1024
            and not integer_output_recurrence
            and not integer_tile_exponent_recurrence
            and not paired_int32_tiles
            and not probability_fp16
            and not normalized_fp16_recurrence
            and not tile_common_log_denominator
            and not narrow_int8_log_denominator
            and not running_max_probability_recurrence
        )
    if optimize_pv_scaling is None:
        optimize_pv_scaling = (
            torch.cuda.get_device_capability(query.device)[0] == 12
            and scale_forward_log_recurrence
            and (native_uint8_mma or not affine_probability)
            and split_pv_head_dim
            and not is_causal
        )
    if optimize_pv_scaling and (
        not scale_forward_log_recurrence
        or not split_pv_head_dim
        or is_causal
    ):
        raise ValueError(
            "optimized PV scaling requires noncausal integer-P split-D128 "
            "scale-forward recurrence"
        )
    if scaled_fp16_correction and (
        native_uint8_mma
        or not affine_probability
        or not split_pv_head_dim
        or not scaled_fp16_numerator
    ):
        raise ValueError(
            "scaled FP16 correction requires affine signed-MMA emulation "
            "with a split scaled-FP16 numerator"
        )
    if delayed_fp16_correction_group not in (0, 8, 16):
        raise ValueError("delayed FP16 correction group must be 0, 8, or 16")
    if delayed_fp16_correction_group and (
        not scaled_fp16_correction
        or is_causal
        or query_length != key_length
        or query_length % (delayed_fp16_correction_group * _PV_BLOCK)
        or not optimize_pv_scaling
    ):
        raise ValueError(
            "delayed correction requires complete noncausal self-attention groups "
            "with optimized scaled-FP16 affine correction"
        )
    if fp32_pv_scale_metadata is None:
        fp32_pv_scale_metadata = optimize_pv_scaling and not scaled_fp16_numerator
    if fp32_pv_scale_metadata and not optimize_pv_scaling:
        raise ValueError("FP32 PV scale metadata requires optimized PV scaling")
    use_padded_descriptor_storage = (
        torch.cuda.get_device_capability(query.device)[0] == 12
        and not is_causal
        and head_dim == 128
        and key_length >= 1024
        and key_length % 16 != 0
        and scale_forward_log_recurrence
        and split_pv_head_dim
        and optimize_pv_scaling
    )
    storage_key_length = (
        int(triton.cdiv(key_length, _PV_BLOCK)) * _PV_BLOCK
        if use_padded_descriptor_storage
        else key_length
    )
    prepared = _prepare_uint8_pv_feature_convrot_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
        rotation_group=rotation_group,
        value_scale_axis=value_scale_axis,
        value_scale_floor=value_scale_floor,
        probability_scale_mode=probability_scale_mode,
        affine_probability=affine_probability,
        native_uint8_mma=native_uint8_mma,
        scaled_fp16_correction=scaled_fp16_correction,
        tile_common_log_denominator=tile_common_log_denominator,
        narrow_int8_log_denominator=narrow_int8_log_denominator,
        scale_forward_log_recurrence=scale_forward_log_recurrence,
        fp32_scale_forward_metadata=fp32_pv_scale_metadata,
        precompute_pv_multiplier=optimize_pv_scaling,
        storage_key_length=storage_key_length,
        center_value=center_value,
    )
    output = torch.empty_like(query)
    rotated_output = (
        output
        if rotation_group == 0 or fuse_output_rotation
        else torch.empty(query.shape, device=query.device, dtype=torch.float32)
    )
    use_tensor_descriptors = False
    maxnreg = None
    if is_causal:
        block_m = 64
        num_stages = 3
    elif split_pv_head_dim:
        block_m = (
            128
            if scaled_fp16_numerator
            and query_length >= 8192
            and key_length >= 8192
            else 64
        )
        use_three_cta_schedule = (
            torch.cuda.get_device_capability(query.device)[0] == 12
            and scale_forward_log_recurrence
            and query_length >= 1024
            and key_length >= 1024
        )
        num_stages = 2 if use_three_cta_schedule else 3
        maxnreg = 168 if use_three_cta_schedule else None
        if scaled_fp16_numerator:
            maxnreg = (
                240
                if scaled_fp16_correction
                and not delayed_fp16_correction_group
                and query_length >= 8192
                and key_length >= 8192
                else None
            )
        use_tensor_descriptors = (
            torch.cuda.get_device_capability(query.device)[0] == 12
            and scaled_fp16_numerator
            and block_m == 128
            and head_dim == 128
            and storage_key_length % 16 == 0
        ) or _sage_backend._should_use_split_pv_tensor_descriptors(
            query,
            block_m,
            head_dim,
            storage_key_length,
            True,
        )
    elif value_scale_axis == "key":
        if rotation_group:
            block_m = 32 if query_length <= 1152 else 64
            num_stages = 2 if block_m == 32 else 3
        elif not affine_probability or native_uint8_mma:
            candidate_block_m = _sage_backend._select_query_block(
                query,
                batch,
                heads,
                query_length,
            )
            use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
                query,
                candidate_block_m,
                head_dim,
                key_length,
                True,
            )
            block_m = candidate_block_m if use_tensor_descriptors else 64
            num_stages = 2 if use_tensor_descriptors else 3
        else:
            block_m = 64
            num_stages = 3
    else:
        block_m = _sage_backend._select_query_block(
            query,
            batch,
            heads,
            query_length,
        )
        num_stages = 3
    unmasked_self_attention = (
        torch.cuda.get_device_capability(query.device)[0] == 12
        and not is_causal
        and query_length == key_length
        and query_length % block_m == 0
        and key_length % _PV_BLOCK == 0
        and scale_forward_log_recurrence
        and split_pv_head_dim
        and optimize_pv_scaling
    )
    split_query_tail = (
        torch.cuda.get_device_capability(query.device)[0] == 12
        and not is_causal
        and query_length >= 16384
        and use_tensor_descriptors
        and scale_forward_log_recurrence
        and split_pv_head_dim
        and optimize_pv_scaling
        and not unmasked_self_attention
    )
    return _launch_uint8_pv_feature_convrot_attention(
        prepared,
        rotated_output,
        output,
        query_length,
        key_length,
        is_causal,
        grouped_qk=grouped_qk,
        rotation_group=rotation_group,
        value_scale_axis=value_scale_axis,
        probability_scale_mode=probability_scale_mode,
        fuse_output_rotation=fuse_output_rotation,
        block_m=block_m,
        num_warps=4,
        num_stages=num_stages,
        running_max_probability_recurrence=running_max_probability_recurrence,
        scale_forward_log_recurrence=scale_forward_log_recurrence,
        tile_common_log_denominator=tile_common_log_denominator,
        narrow_int8_log_denominator=narrow_int8_log_denominator,
        affine_probability=affine_probability,
        native_uint8_mma=native_uint8_mma,
        integer_output_recurrence=integer_output_recurrence,
        integer_tile_exponent_recurrence=integer_tile_exponent_recurrence,
        predot_exponent_alignment=predot_exponent_alignment,
        dithered_predot_alignment=dithered_predot_alignment,
        paired_int32_tiles=paired_int32_tiles,
        probability_fp16=probability_fp16,
        factored_pv_scaling=optimize_pv_scaling,
        precomputed_pv_multiplier=optimize_pv_scaling,
        normalized_fp16_recurrence=normalized_fp16_recurrence,
        scaled_fp16_numerator=scaled_fp16_numerator,
        scaled_fp16_denominator=scaled_fp16_denominator,
        split_pv_head_dim=split_pv_head_dim,
        scaled_fp16_correction=scaled_fp16_correction,
        delayed_fp16_correction_group=delayed_fp16_correction_group,
        unmasked_self_attention=unmasked_self_attention,
        split_query_tail=split_query_tail,
        use_tensor_descriptors=use_tensor_descriptors,
        storage_key_length=storage_key_length,
        maxnreg=maxnreg,
        center_value=center_value,
    )


def triton_sage_attention_uint8_pv_int32_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
    affine_probability: bool = True,
    native_uint8_mma: bool = False,
    tile_exponent: bool = False,
) -> torch.Tensor:
    """Run per-key INT8 V with a persistent INT32 PV numerator."""
    return triton_sage_attention_uint8_pv_feature_convrot(
        query,
        key,
        value,
        scale,
        is_causal,
        rotation_group=0,
        value_scale_axis="key",
        probability_scale_mode="log",
        grouped_qk=grouped_qk,
        affine_probability=affine_probability,
        native_uint8_mma=native_uint8_mma,
        integer_output_recurrence=not tile_exponent,
        integer_tile_exponent_recurrence=tile_exponent,
    )


def triton_sage_attention_int8_pv_per_key_log(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
) -> torch.Tensor:
    """Run exact per-key-scaled V with nonnegative signed-INT8 P."""
    use_sm120_scaled_path = (
        torch.cuda.get_device_capability(query.device)[0] == 12
        and not is_causal
        and query.shape[-1] == 128
        and key.shape[2] <= 131072
    )
    return triton_sage_attention_uint8_pv_feature_convrot(
        query,
        key,
        value,
        scale,
        is_causal,
        rotation_group=0,
        value_scale_axis="key",
        probability_scale_mode="log",
        grouped_qk=grouped_qk,
        affine_probability=False,
        split_pv_head_dim=use_sm120_scaled_path,
        scale_forward_log_recurrence=use_sm120_scaled_path,
        optimize_pv_scaling=use_sm120_scaled_path,
        scaled_fp16_numerator=use_sm120_scaled_path,
    )
