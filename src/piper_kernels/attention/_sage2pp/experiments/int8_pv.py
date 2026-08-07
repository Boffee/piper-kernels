"""Direct integer-PV baselines for SageAttention2++.

These experiments deliberately omit ConvRot. They keep the production INT8 QK
and FP32 online softmax, then compare Sage2++'s FP8 PV path with:

* a fastest-case fixed-scale control with one V scale per head/output channel;
* a quality-oriented path with block-local P normalization and V scales.

Both map nonnegative ``P`` to signed-INT8 codes ``[0, 127]`` and use symmetric
signed-INT8 V. Their tensor-core dot accumulates in INT32 before the FP32 online
recurrence. The fixed-scale control factors all PV scales out of the loop; the
block-scaled path pays one partial-output rescale per tile to retain quality.
The fixed-scale path can optionally use native UINT8 ``P`` codes ``[0, 255]``
through Piper's stock-Triton compiler extension, retaining signed-INT8 V.
"""

# ruff: noqa: ANN001, ANN202, PLR0912, PLR0913, PLR0915, PLR0917

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels._triton.mixed_int8 import (
    enable_uint8_int8_dot,
    uint8_int8_dot,
)
from piper_kernels.attention._convrot_triton import (
    rotate_attention_rows,
    rotate_rows_in_registers,
)
from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.uint4_pv_convrot import _prepare_qk

_INT8_RANGE = tl.constexpr(127.0)
_UINT8_RANGE = tl.constexpr(255.0)
_FP8_FOLDED_RANGE = tl.constexpr(1008.0)
_INT8_FOLDED_RANGE = tl.constexpr(16129.0)
_UINT8_INT8_FOLDED_RANGE = tl.constexpr(32385.0)


@triton.jit
def _bounded_int32_to_fp32(value):
    """Exactly convert signed integers in [-2**22, 2**22] without I2FP."""
    biased_bits = value + 0x4B400000
    return biased_bits.to(tl.float32, bitcast=True) - 12582912.0
_BASE_RANGE_BUCKETS = 32
_BASE_RANGE_BUCKET_OFFSET = 16


@triton.jit
def _rounded_shift_int32_elements(values, shifts):
    """Align element-wise block-floating INT32 values with rounded shifts."""
    clamped_shifts = tl.minimum(tl.maximum(shifts, 0), 31)
    safe_shifts = tl.maximum(clamped_shifts, 1)
    one = tl.full(shifts.shape, 1, dtype=tl.int32)
    rounding = one << (safe_shifts - 1)
    shifted = (values + rounding) >> clamped_shifts
    return tl.where(clamped_shifts == 0, values, shifted)


@triton.jit
def _rounded_shift_int32_rows(values, shifts):
    """Align row-wise block-floating INT32 values with rounded shifts."""
    clamped_shifts = tl.minimum(tl.maximum(shifts, 0), 31)
    safe_shifts = tl.maximum(clamped_shifts, 1)
    one = tl.full(shifts.shape, 1, dtype=tl.int32)
    rounding = one << (safe_shifts - 1)
    shifted = (values + rounding[:, None]) >> clamped_shifts[:, None]
    return tl.where(clamped_shifts[:, None] == 0, values, shifted)


@triton.jit
def _quantize_value_block_int8_kernel(
    value_ptr,
    scale_ptr,
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
    output_transposed: tl.constexpr,
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
    value_scale = tl.max(tl.abs(value), axis=0) / _INT8_RANGE + 1e-7
    quantized = _sage_backend._round_to_int8(value / value_scale[None, :], _INT8_RANGE)
    scale_block = (batch * heads + head) * tl.cdiv(key_length, block_n) + key_block
    tl.store(scale_ptr + scale_block * head_dim + offsets_d, value_scale)
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
def _quantize_value_grouped_int8_kernel(
    value_ptr,
    scale_ptr,
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
    feature_group: tl.constexpr,
    store_log2_scale: tl.constexpr,
    output_transposed: tl.constexpr,
):
    """Quantize a K tile with one symmetric V scale per feature group."""
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
    feature_groups: tl.constexpr = head_dim // feature_group
    grouped = value.reshape((block_n, feature_groups, feature_group))
    value_scale = tl.max(tl.max(tl.abs(grouped), axis=0), axis=1) / _INT8_RANGE + 1e-7
    quantized = _sage_backend._round_to_int8(
        grouped / value_scale[None, :, None],
        _INT8_RANGE,
    ).reshape((block_n, head_dim))
    scale_block = (batch * heads + head) * tl.cdiv(key_length, block_n) + key_block
    # The FP32 path folds P's inverse range into metadata. The INT32 path needs
    # log2(s_v) to form its per-query block-floating coefficient.
    stored_scale = (
        tl.log2(value_scale)
        if store_log2_scale
        else value_scale * (1.0 / 255.0)
    )
    tl.store(
        scale_ptr + scale_block * feature_groups + tl.arange(0, feature_groups),
        stored_scale,
    )
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
def _grouped_scale_ratio_kernel(
    scale_ptr,
    ratio_ptr,
    heads: tl.constexpr,
    scale_runs: tl.constexpr,
    feature_groups: tl.constexpr,
):
    """Precompute adjacent sorted-run scale-coordinate transitions."""
    scale_run = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_g = tl.arange(0, feature_groups)
    current_base = ((batch * heads + head) * scale_runs + scale_run) * feature_groups
    previous_run = tl.maximum(scale_run - 1, 0)
    previous_base = ((batch * heads + head) * scale_runs + previous_run) * feature_groups
    current = tl.load(scale_ptr + current_base + offsets_g).to(tl.float32)
    previous = tl.load(scale_ptr + previous_base + offsets_g).to(tl.float32)
    ratio = tl.where(scale_run == 0, 1.0, previous / current)
    tl.store(ratio_ptr + current_base + offsets_g, ratio)


@triton.jit
def _reduce_grouped_value_run_scale_kernel(
    value_ptr,
    scale_max_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    scale_run_n: tl.constexpr,
    feature_group: tl.constexpr,
):
    """Reduce K128 tiles into one FP32 V maximum per sorted scale run."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=offsets_n[:, None] < key_length,
        other=0.0,
    ).to(tl.float32)
    feature_groups: tl.constexpr = head_dim // feature_group
    grouped = value.reshape((block_n, feature_groups, feature_group))
    tile_max = tl.max(tl.max(tl.abs(grouped), axis=0), axis=1)
    scale_runs = tl.cdiv(key_length, scale_run_n)
    scale_run = key_block // (scale_run_n // block_n)
    scale_base = ((batch * heads + head) * scale_runs + scale_run) * feature_groups
    tl.atomic_max(
        scale_max_ptr + scale_base + tl.arange(0, feature_groups),
        tile_max,
        sem="relaxed",
    )


@triton.jit
def _quantize_value_grouped_int8_with_run_scale_kernel(
    value_ptr,
    scale_max_ptr,
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
    scale_run_n: tl.constexpr,
    feature_group: tl.constexpr,
    output_transposed: tl.constexpr,
):
    """Quantize a K128 tile using its pre-reduced sorted-run V scale."""
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
    feature_groups: tl.constexpr = head_dim // feature_group
    scale_runs = tl.cdiv(key_length, scale_run_n)
    scale_run = key_block // (scale_run_n // block_n)
    scale_base = ((batch * heads + head) * scale_runs + scale_run) * feature_groups
    scale_max = tl.load(
        scale_max_ptr + scale_base + tl.arange(0, feature_groups)
    )
    value_scale = scale_max / _INT8_RANGE + 1e-7
    grouped = value.reshape((block_n, feature_groups, feature_group))
    quantized = _sage_backend._round_to_int8(
        grouped / value_scale[None, :, None],
        _INT8_RANGE,
    ).reshape((block_n, head_dim))
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
def _finalize_grouped_value_run_scale_kernel(
    scale_max_ptr,
    scale_ptr,
    elements,
    block: tl.constexpr,
):
    """Fold the UINT8 probability range into pre-reduced V scales."""
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    scale_max = tl.load(scale_max_ptr + offsets, mask=offsets < elements)
    folded_scale = (scale_max / _INT8_RANGE + 1e-7) * (1.0 / 255.0)
    tl.store(scale_ptr + offsets, folded_scale, mask=offsets < elements)


@triton.jit
def _bucket_histogram_kernel(
    value_ptr,
    bucket_ptr,
    histogram_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    range_buckets: tl.constexpr,
    range_bucket_offset: tl.constexpr,
    range_bucket_log2_scale: tl.constexpr,
    block_n: tl.constexpr,
):
    """Assign rows by rotated V range and accumulate one histogram per head."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid = offsets_n < key_length
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    value = rotate_rows_in_registers(value, offsets_d, block_n, 64)
    value_range = tl.max(tl.abs(value), axis=1) + 1e-30
    bucket = (
        tl.floor(tl.log2(value_range) * range_bucket_log2_scale).to(tl.int32)
        + range_bucket_offset
    )
    bucket = tl.minimum(tl.maximum(bucket, 0), range_buckets - 1)
    batch_head = batch * heads + head
    tl.store(bucket_ptr + batch_head * key_length + offsets_n, bucket, mask=valid)
    tl.atomic_add(
        histogram_ptr + batch_head * range_buckets + bucket,
        1,
        mask=valid,
    )


@triton.jit
def _bucket_prefix_kernel(
    histogram_ptr,
    offset_ptr,
    cursor_ptr,
    range_buckets: tl.constexpr,
):
    """Exclusive-prefix bucket counts and reset the scatter cursors."""
    batch_head = tl.program_id(0)
    offsets = tl.arange(0, range_buckets)
    counts = tl.load(histogram_ptr + batch_head * range_buckets + offsets)
    prefix = tl.cumsum(counts, axis=0) - counts
    tl.store(offset_ptr + batch_head * range_buckets + offsets, prefix)
    tl.store(cursor_ptr + batch_head * range_buckets + offsets, 0)


@triton.jit
def _bucket_scatter_kv_kernel(
    key_ptr,
    value_ptr,
    bucket_ptr,
    offset_ptr,
    cursor_ptr,
    packed_key_ptr,
    packed_value_ptr,
    key_length,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    range_buckets: tl.constexpr,
    block_n: tl.constexpr,
):
    """Atomically pack paired K/V rows; order within a bucket is unspecified."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid = offsets_n < key_length
    batch_head = batch * heads + head
    bucket = tl.load(
        bucket_ptr + batch_head * key_length + offsets_n,
        mask=valid,
        other=0,
    )
    rank = tl.atomic_add(
        cursor_ptr + batch_head * range_buckets + bucket,
        1,
        mask=valid,
    )
    bucket_offset = tl.load(
        offset_ptr + batch_head * range_buckets + bucket,
        mask=valid,
        other=0,
    )
    destination = bucket_offset + rank
    key = tl.load(
        key_ptr
        + batch * stride_kb
        + head * stride_kh
        + offsets_n[:, None] * stride_kn
        + offsets_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    value = rotate_rows_in_registers(value, offsets_d, block_n, 64)
    output_offsets = (
        (batch_head * key_length + destination[:, None]) * head_dim
        + offsets_d[None, :]
    )
    tl.store(packed_key_ptr + output_offsets, key, mask=valid[:, None])
    tl.store(packed_value_ptr + output_offsets, value, mask=valid[:, None])


@triton.jit
def _quantize_value_int8_kernel(
    value_ptr,
    folded_fp8_scale_ptr,
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
    output_transposed: tl.constexpr,
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

    # The production statistics kernel stores max(abs(V)) / (448 * 2.25).
    # Recover max(abs(V)) / 127 without launching another reduction.
    folded_scale = tl.load(folded_fp8_scale_ptr + (batch * heads + head) * head_dim + offsets_d)
    value_scale = folded_scale * (_FP8_FOLDED_RANGE / 127.0)
    quantized = _sage_backend._round_to_int8(value / value_scale[None, :], _INT8_RANGE)
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
def _int8_pv_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    folded_fp8_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    block_scaled_pv: tl.constexpr,
    block_global_probability: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    native_unsigned_probability: tl.constexpr,
    integer_pv_recurrence: tl.constexpr,
    raw_integer_pv_recurrence: tl.constexpr,
    raw_fp32_pv_recurrence: tl.constexpr,
    magic_score_conversion: tl.constexpr,
    magic_pv_conversion: tl.constexpr,
    fp16_pv_conversion: tl.constexpr,
    bf16_pv_conversion: tl.constexpr,
    unmasked_self_attention: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    value_transposed: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
    use_tensor_descriptors: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    if split_pv_head_dim:
        offsets_vd = tl.arange(0, half_head_dim)
    if unmasked_self_attention:
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

    if integer_pv_recurrence or raw_integer_pv_recurrence:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.int32)
    elif split_pv_head_dim:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    else:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    if integer_pv_recurrence:
        running_exponent = tl.full((block_m,), -(1 << 30), dtype=tl.int32)
    else:
        running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
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
        if magic_score_conversion:
            float_scores = _bounded_int32_to_fp32(integer_scores)
        else:
            float_scores = integer_scores.to(tl.float32)
        if grouped_qk:
            if block_n == 64:
                key_scale = tl.load(
                    key_scale_ptr + (batch * heads + head) * tl.cdiv(key_length, 64) + start_n // 64
                )
                scores = float_scores * (query_scale * key_scale)[:, None]
            else:
                key_scale = tl.load(
                    key_scale_ptr
                    + (batch * heads + head) * tl.cdiv(key_length, 64)
                    + current_n // 64,
                    mask=current_n < key_length,
                    other=0.0,
                )
                scores = float_scores * query_scale[:, None] * key_scale[None, :]
        else:
            key_scale = tl.load(
                key_scale_ptr + (batch * heads + head) * key_length + current_n,
                mask=current_n < key_length,
                other=0.0,
            )
            scores = float_scores * query_scale[:, None] * key_scale[None, :]

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

        block_max = tl.max(scores, axis=1)
        if integer_pv_recurrence:
            block_exponent = tl.ceil(tl.where(valid_queries, block_max, 0.0)).to(tl.int32)
            next_exponent = tl.maximum(running_exponent, block_exponent)
            exponent_shift = next_exponent - running_exponent
            accumulator = _rounded_shift_int32_rows(accumulator, exponent_shift)
            old_weight = tl.where(
                valid_queries,
                tl.exp2((running_exponent - next_exponent).to(tl.float32)),
                0.0,
            )
            current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - next_exponent.to(tl.float32)[:, None]),
                0.0,
            )
        else:
            next_max = tl.maximum(running_max, block_max)
            old_weight = tl.where(valid_queries, tl.exp2(running_max - next_max), 0.0)
        if integer_pv_recurrence:
            pass
        elif block_scaled_pv:
            if block_global_probability:
                current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
                probabilities = tl.where(
                    valid_queries[:, None] & valid_keys,
                    tl.exp2(scores - next_max[:, None]),
                    0.0,
                )
            else:
                current_weight = tl.where(valid_queries, tl.exp2(block_max - next_max), 0.0)
                probabilities = tl.where(
                    valid_queries[:, None] & valid_keys,
                    tl.exp2(scores - block_max[:, None]),
                    0.0,
                )
        else:
            current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - next_max[:, None]),
                0.0,
            )
        if integer_pv_recurrence or raw_integer_pv_recurrence or raw_fp32_pv_recurrence:
            pass
        elif split_pv_head_dim:
            accumulator_low *= old_weight[:, None]
            accumulator_high *= old_weight[:, None]
        else:
            accumulator *= old_weight[:, None]
        denominator = denominator * old_weight + tl.sum(probabilities, axis=1) * current_weight

        # Probabilities are already in [0, 1].  Adding one half before the
        # float-to-integer truncation implements round-to-nearest without the
        # sign branch and clamps needed by the general quantizer.
        if native_unsigned_probability:
            probability_integer = (probabilities * _UINT8_RANGE + 0.5).to(tl.uint8)
        else:
            probability_integer = (probabilities * _INT8_RANGE + 0.5).to(tl.int8)
        if split_pv_head_dim:
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
            if native_unsigned_probability:
                partial_low = uint8_int8_dot(probability_integer, value_low)
            else:
                partial_low = tl.dot(probability_integer, value_low, out_dtype=tl.int32)
            if not block_scaled_pv:
                accumulator_low += partial_low.to(tl.float32)
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
            if native_unsigned_probability:
                partial_high = uint8_int8_dot(probability_integer, value_high)
            else:
                partial_high = tl.dot(probability_integer, value_high, out_dtype=tl.int32)
            if not block_scaled_pv:
                accumulator_high += partial_high.to(tl.float32)
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
            if integer_pv_recurrence or raw_integer_pv_recurrence:
                if native_unsigned_probability:
                    accumulator = uint8_int8_dot(
                        probability_integer,
                        value,
                        accumulator,
                    )
                else:
                    accumulator = tl.dot(
                        probability_integer,
                        value,
                        accumulator,
                        out_dtype=tl.int32,
                    )
            elif native_unsigned_probability:
                partial_int32 = uint8_int8_dot(probability_integer, value)
            else:
                partial_int32 = tl.dot(
                    probability_integer,
                    value,
                    out_dtype=tl.int32,
                )
        if block_scaled_pv:
            value_scale_block = (batch * heads + head) * tl.cdiv(
                key_length, block_n
            ) + start_n // block_n
            probability_range: tl.constexpr = (
                _UINT8_RANGE if native_unsigned_probability else _INT8_RANGE
            )
            if split_pv_head_dim:
                value_scale_low = tl.load(
                    folded_fp8_scale_ptr
                    + value_scale_block * head_dim
                    + offsets_vd
                )
                value_scale_high = tl.load(
                    folded_fp8_scale_ptr
                    + value_scale_block * head_dim
                    + half_head_dim
                    + offsets_vd
                )
                accumulator_low += (
                    partial_low.to(tl.float32)
                    * current_weight[:, None]
                    * (value_scale_low[None, :] / probability_range)
                )
                accumulator_high += (
                    partial_high.to(tl.float32)
                    * current_weight[:, None]
                    * (value_scale_high[None, :] / probability_range)
                )
            else:
                value_scale = tl.load(
                    folded_fp8_scale_ptr + value_scale_block * head_dim + offsets_d
                )
                if magic_pv_conversion:
                    partial_fp32 = _bounded_int32_to_fp32(partial_int32)
                else:
                    partial_fp32 = partial_int32.to(tl.float32)
                accumulator += (
                    partial_fp32
                    * current_weight[:, None]
                    * (value_scale[None, :] / probability_range)
                )
        elif (
            not split_pv_head_dim
            and not integer_pv_recurrence
            and not raw_integer_pv_recurrence
        ):
            if fp16_pv_conversion:
                # A K64 INT8 dot is bounded by 64 * 127 * 127. Shifting by
                # five keeps the signed result finite in FP16. The common
                # factor is restored once in the output epilogue.
                partial_fp32 = (partial_int32 >> 5).to(tl.float16).to(tl.float32)
            elif bf16_pv_conversion:
                partial_fp32 = partial_int32.to(tl.bfloat16).to(tl.float32)
            elif magic_pv_conversion:
                partial_fp32 = _bounded_int32_to_fp32(partial_int32)
            else:
                partial_fp32 = partial_int32.to(tl.float32)
            accumulator += partial_fp32
        if integer_pv_recurrence:
            running_exponent = next_exponent
        else:
            running_max = next_max

    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    if split_pv_head_dim:
        output_low = accumulator_low / denominator_safe
        output_high = accumulator_high / denominator_safe
        if not block_scaled_pv:
            folded_scale_low = tl.load(
                folded_fp8_scale_ptr + (batch * heads + head) * head_dim + offsets_vd
            )
            folded_scale_high = tl.load(
                folded_fp8_scale_ptr
                + (batch * heads + head) * head_dim
                + half_head_dim
                + offsets_vd
            )
            output_scale: tl.constexpr = _FP8_FOLDED_RANGE / (
                _UINT8_INT8_FOLDED_RANGE
                if native_unsigned_probability
                else _INT8_FOLDED_RANGE
            )
            output_low *= folded_scale_low[None, :] * output_scale
            output_high *= folded_scale_high[None, :] * output_scale
        output_base = (
            output_ptr + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        )
        tl.store(output_base + offsets_vd[None, :], output_low, mask=valid_queries[:, None])
        tl.store(
            output_base + half_head_dim + offsets_vd[None, :],
            output_high,
            mask=valid_queries[:, None],
        )
    else:
        if integer_pv_recurrence or raw_integer_pv_recurrence:
            output = accumulator.to(tl.float32) / denominator_safe
        else:
            output = accumulator / denominator_safe
        if not block_scaled_pv:
            folded_fp8_scale = tl.load(
                folded_fp8_scale_ptr + (batch * heads + head) * head_dim + offsets_d
            )
            conversion_scale: tl.constexpr = 32.0 if fp16_pv_conversion else 1.0
            probability_value_range: tl.constexpr = (
                _UINT8_INT8_FOLDED_RANGE
                if native_unsigned_probability
                else _INT8_FOLDED_RANGE
            )
            output *= (
                folded_fp8_scale[None, :]
                * (_FP8_FOLDED_RANGE / probability_value_range)
                * conversion_scale
            )
        tl.store(
            output_ptr
            + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
            + offsets_d[None, :],
            output,
            mask=valid_queries[:, None],
        )


@triton.jit
def _uint8_k32_feature_pv_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """K64 QK/softmax with two output-scaled native-UINT8 K32 PV dots."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    offsets_vd = tl.arange(0, half_head_dim)
    valid_queries = offsets_m < query_length
    batch_head = batch * heads + head

    query = tl.load(
        query_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    query_scale = tl.load(
        query_scale_ptr
        + batch_head * tl.cdiv(query_length, 32)
        + offsets_m // 32,
        mask=valid_queries,
        other=0.0,
    )
    accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)

    for start_n in tl.range(0, key_length, block_n, disable_licm=True):
        current_n = start_n + offsets_n
        key = key_ptr.load([batch_head, start_n, 0]).reshape((block_n, head_dim))
        integer_scores = tl.dot(query, key.T, out_dtype=tl.int32)
        if block_n == 64:
            key_scale = tl.load(
                key_scale_ptr
                + batch_head * tl.cdiv(key_length, 64)
                + start_n // 64
            )
            scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
        else:
            key_scale = tl.load(
                key_scale_ptr
                + batch_head * tl.cdiv(key_length, 64)
                + current_n // 64
            )
            scores = (
                integer_scores.to(tl.float32)
                * query_scale[:, None]
                * key_scale[None, :]
            )
        valid_keys = current_n[None, :] < key_length
        scores = tl.where(valid_queries[:, None] & valid_keys, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_weight = tl.where(valid_queries, tl.exp2(running_max - next_max), 0.0)
        current_weight = tl.where(valid_queries, tl.exp2(block_max - next_max), 0.0)
        probabilities = tl.where(
            valid_queries[:, None] & valid_keys,
            tl.exp2(scores - block_max[:, None]),
            0.0,
        )
        accumulator_low *= old_weight[:, None]
        accumulator_high *= old_weight[:, None]
        denominator = (
            denominator * old_weight
            + tl.sum(probabilities, axis=1) * current_weight
        )
        probability_codes = (probabilities * 255.0 + 0.5).to(tl.uint8)
        probability_pairs = probability_codes.reshape(
            (block_m, 2, block_n // 2)
        ).permute((0, 2, 1))
        probability0, probability1 = probability_pairs.split()
        output_weight = current_weight * (1.0 / 255.0)

        value_block0 = start_n // (block_n // 2)
        value_block1 = value_block0 + 1
        scale_base0 = (
            batch_head * tl.cdiv(key_length, block_n // 2) + value_block0
        ) * head_dim
        scale_base1 = (
            batch_head * tl.cdiv(key_length, block_n // 2) + value_block1
        ) * head_dim

        value_low0 = value_ptr.load([batch_head, 0, start_n]).reshape(
            (half_head_dim, block_n // 2)
        ).T
        partial_low0 = uint8_int8_dot(probability0, value_low0)
        scale_low0 = tl.load(value_scale_ptr + scale_base0 + offsets_vd)
        accumulator_low += (
            partial_low0.to(tl.float32)
            * scale_low0[None, :]
            * output_weight[:, None]
        )
        value_low1 = value_ptr.load(
            [batch_head, 0, start_n + block_n // 2]
        ).reshape((half_head_dim, block_n // 2)).T
        partial_low1 = uint8_int8_dot(probability1, value_low1)
        scale_low1 = tl.load(value_scale_ptr + scale_base1 + offsets_vd)
        accumulator_low += (
            partial_low1.to(tl.float32)
            * scale_low1[None, :]
            * output_weight[:, None]
        )

        value_high0 = value_ptr.load(
            [batch_head, half_head_dim, start_n]
        ).reshape((half_head_dim, block_n // 2)).T
        partial_high0 = uint8_int8_dot(probability0, value_high0)
        scale_high0 = tl.load(
            value_scale_ptr + scale_base0 + half_head_dim + offsets_vd
        )
        accumulator_high += (
            partial_high0.to(tl.float32)
            * scale_high0[None, :]
            * output_weight[:, None]
        )
        value_high1 = value_ptr.load(
            [batch_head, half_head_dim, start_n + block_n // 2]
        ).reshape((half_head_dim, block_n // 2)).T
        partial_high1 = uint8_int8_dot(probability1, value_high1)
        scale_high1 = tl.load(
            value_scale_ptr + scale_base1 + half_head_dim + offsets_vd
        )
        accumulator_high += (
            partial_high1.to(tl.float32)
            * scale_high1[None, :]
            * output_weight[:, None]
        )
        running_max = next_max

    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    output_base = (
        output_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
    )
    tl.store(
        output_base + offsets_vd[None, :],
        accumulator_low / denominator_safe,
        mask=valid_queries[:, None],
    )
    tl.store(
        output_base + half_head_dim + offsets_vd[None, :],
        accumulator_high / denominator_safe,
        mask=valid_queries[:, None],
    )


@triton.jit
def _uint8_grouped_output_pv_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    feature_group: tl.constexpr,
    local_probability_codes: tl.constexpr,
    integer_output_recurrence: tl.constexpr,
    common_feature_exponent: tl.constexpr,
    unmasked_self_attention: tl.constexpr,
):
    """Native-UINT8 PV with one output-side V scale per feature group."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    half_groups: tl.constexpr = (
        tl.constexpr(1) if feature_group == head_dim else half_head_dim // feature_group
    )
    feature_groups: tl.constexpr = head_dim // feature_group
    high_group_offset: tl.constexpr = (
        tl.constexpr(0) if feature_group == head_dim else half_groups
    )
    scale_repeat: tl.constexpr = half_head_dim if feature_group == head_dim else feature_group
    offsets_vd = tl.arange(0, half_head_dim)
    offsets_g = tl.arange(0, half_groups)
    if unmasked_self_attention:
        valid_queries = tl.full((block_m,), True, dtype=tl.int1)
    else:
        valid_queries = offsets_m < query_length
    batch_head = batch * heads + head

    query = tl.load(
        query_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    query_scale = tl.load(
        query_scale_ptr
        + batch_head * tl.cdiv(query_length, 32)
        + offsets_m // 32,
        mask=valid_queries,
        other=0.0,
    )
    if integer_output_recurrence:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.int32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.int32)
        if common_feature_exponent:
            running_exponent = tl.full((block_m,), -(1 << 30), dtype=tl.int32)
        else:
            running_exponent_low = tl.full(
                (block_m, half_groups),
                -(1 << 30),
                dtype=tl.int32,
            )
            running_exponent_high = tl.full(
                (block_m, half_groups),
                -(1 << 30),
                dtype=tl.int32,
            )
    else:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)

    for start_n in tl.range(0, key_length, block_n, disable_licm=True):
        current_n = start_n + offsets_n
        key = key_ptr.load([batch_head, start_n, 0]).reshape((block_n, head_dim))
        integer_scores = tl.dot(query, key.T, out_dtype=tl.int32)
        if block_n == 64:
            key_scale = tl.load(
                key_scale_ptr
                + batch_head * tl.cdiv(key_length, 64)
                + start_n // 64
            )
            scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
        else:
            key_scale = tl.load(
                key_scale_ptr
                + batch_head * tl.cdiv(key_length, 64)
                + current_n // 64
            )
            scores = (
                integer_scores.to(tl.float32)
                * query_scale[:, None]
                * key_scale[None, :]
            )
        if unmasked_self_attention:
            valid_keys = tl.full((block_m, block_n), True, dtype=tl.int1)
        else:
            valid_keys = current_n[None, :] < key_length
            scores = tl.where(
                valid_queries[:, None] & valid_keys,
                scores,
                -float("inf"),
            )

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        if unmasked_self_attention:
            old_weight = tl.exp2(running_max - next_max)
        else:
            old_weight = tl.where(
                valid_queries,
                tl.exp2(running_max - next_max),
                0.0,
            )
        if local_probability_codes:
            if unmasked_self_attention:
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
        elif unmasked_self_attention:
            probabilities = tl.exp2(scores - next_max[:, None])
        else:
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - next_max[:, None]),
                0.0,
            )
        if not integer_output_recurrence:
            accumulator_low *= old_weight[:, None]
            accumulator_high *= old_weight[:, None]
        if local_probability_codes:
            denominator = (
                denominator * old_weight
                + tl.sum(probabilities, axis=1) * current_weight
            )
        else:
            denominator = denominator * old_weight + tl.sum(probabilities, axis=1)
        probability_codes = (probabilities * 255.0 + 0.5).to(tl.uint8)
        scale_base = (
            batch_head * tl.cdiv(key_length, block_n) + start_n // block_n
        ) * feature_groups

        compact_scale_low = tl.load(value_scale_ptr + scale_base + offsets_g)
        value_low = value_ptr.load([batch_head, 0, start_n]).reshape(
            (half_head_dim, block_n)
        ).T
        partial_low = uint8_int8_dot(probability_codes, value_low)
        if integer_output_recurrence:
            safe_block_max = tl.where(valid_queries, block_max, 0.0)
            coefficient_low = safe_block_max[:, None] + compact_scale_low[None, :]
            if common_feature_exponent:
                compact_scale_high_for_exponent = tl.load(
                    value_scale_ptr + scale_base + high_group_offset + offsets_g
                )
                common_scale = tl.maximum(
                    tl.max(compact_scale_low, axis=0),
                    tl.max(compact_scale_high_for_exponent, axis=0),
                )
                common_tile_exponent = tl.ceil(
                    safe_block_max + common_scale
                ).to(tl.int32)
                mantissa_exponent_low = common_tile_exponent[:, None]
            else:
                tile_exponent_low = tl.ceil(coefficient_low).to(tl.int32)
                mantissa_exponent_low = tile_exponent_low
            mantissa_low = (
                tl.exp2(coefficient_low - mantissa_exponent_low.to(tl.float32))
                * 256.0
                + 0.5
            ).to(tl.int32)
            expanded_mantissa_low = tl.broadcast_to(
                mantissa_low[:, :, None],
                (block_m, half_groups, scale_repeat),
            ).reshape((block_m, half_head_dim))
            partial_low = (partial_low * expanded_mantissa_low + 128) >> 8
            if common_feature_exponent:
                common_next_exponent = tl.maximum(
                    running_exponent,
                    common_tile_exponent,
                )
                old_shift_low = tl.broadcast_to(
                    (common_next_exponent - running_exponent)[:, None],
                    (block_m, half_head_dim),
                )
                tile_shift_low = tl.broadcast_to(
                    (common_next_exponent - common_tile_exponent)[:, None],
                    (block_m, half_head_dim),
                )
            else:
                next_exponent_low = tl.maximum(running_exponent_low, tile_exponent_low)
                old_shift_low = tl.broadcast_to(
                    (next_exponent_low - running_exponent_low)[:, :, None],
                    (block_m, half_groups, scale_repeat),
                ).reshape((block_m, half_head_dim))
                tile_shift_low = tl.broadcast_to(
                    (next_exponent_low - tile_exponent_low)[:, :, None],
                    (block_m, half_groups, scale_repeat),
                ).reshape((block_m, half_head_dim))
            accumulator_low = _rounded_shift_int32_elements(
                accumulator_low,
                old_shift_low,
            ) + _rounded_shift_int32_elements(partial_low, tile_shift_low)
            if not common_feature_exponent:
                running_exponent_low = next_exponent_low
        else:
            scale_low = tl.broadcast_to(
                compact_scale_low[:, None],
                (half_groups, scale_repeat),
            ).reshape((half_head_dim,))
            if local_probability_codes:
                output_weight = current_weight
                accumulator_low += (
                    partial_low.to(tl.float32)
                    * scale_low[None, :]
                    * output_weight[:, None]
                )
            else:
                accumulator_low += partial_low.to(tl.float32) * scale_low[None, :]

        compact_scale_high = tl.load(
            value_scale_ptr + scale_base + high_group_offset + offsets_g
        )
        value_high = value_ptr.load([batch_head, half_head_dim, start_n]).reshape(
            (half_head_dim, block_n)
        ).T
        partial_high = uint8_int8_dot(probability_codes, value_high)
        if integer_output_recurrence:
            safe_block_max = tl.where(valid_queries, block_max, 0.0)
            coefficient_high = safe_block_max[:, None] + compact_scale_high[None, :]
            if common_feature_exponent:
                mantissa_exponent_high = common_tile_exponent[:, None]
            else:
                tile_exponent_high = tl.ceil(coefficient_high).to(tl.int32)
                mantissa_exponent_high = tile_exponent_high
            mantissa_high = (
                tl.exp2(coefficient_high - mantissa_exponent_high.to(tl.float32))
                * 256.0
                + 0.5
            ).to(tl.int32)
            expanded_mantissa_high = tl.broadcast_to(
                mantissa_high[:, :, None],
                (block_m, half_groups, scale_repeat),
            ).reshape((block_m, half_head_dim))
            partial_high = (partial_high * expanded_mantissa_high + 128) >> 8
            if common_feature_exponent:
                old_shift_high = tl.broadcast_to(
                    (common_next_exponent - running_exponent)[:, None],
                    (block_m, half_head_dim),
                )
                tile_shift_high = tl.broadcast_to(
                    (common_next_exponent - common_tile_exponent)[:, None],
                    (block_m, half_head_dim),
                )
            else:
                next_exponent_high = tl.maximum(running_exponent_high, tile_exponent_high)
                old_shift_high = tl.broadcast_to(
                    (next_exponent_high - running_exponent_high)[:, :, None],
                    (block_m, half_groups, scale_repeat),
                ).reshape((block_m, half_head_dim))
                tile_shift_high = tl.broadcast_to(
                    (next_exponent_high - tile_exponent_high)[:, :, None],
                    (block_m, half_groups, scale_repeat),
                ).reshape((block_m, half_head_dim))
            accumulator_high = _rounded_shift_int32_elements(
                accumulator_high,
                old_shift_high,
            ) + _rounded_shift_int32_elements(partial_high, tile_shift_high)
            if common_feature_exponent:
                running_exponent = common_next_exponent
            else:
                running_exponent_high = next_exponent_high
        else:
            scale_high = tl.broadcast_to(
                compact_scale_high[:, None],
                (half_groups, scale_repeat),
            ).reshape((half_head_dim,))
            if local_probability_codes:
                accumulator_high += (
                    partial_high.to(tl.float32)
                    * scale_high[None, :]
                    * output_weight[:, None]
                )
            else:
                accumulator_high += partial_high.to(tl.float32) * scale_high[None, :]
        running_max = next_max

    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    if integer_output_recurrence:
        if common_feature_exponent:
            expanded_exponent_low = tl.broadcast_to(
                running_exponent[:, None],
                (block_m, half_head_dim),
            )
            expanded_exponent_high = expanded_exponent_low
        else:
            expanded_exponent_low = tl.broadcast_to(
                running_exponent_low[:, :, None],
                (block_m, half_groups, scale_repeat),
            ).reshape((block_m, half_head_dim))
            expanded_exponent_high = tl.broadcast_to(
                running_exponent_high[:, :, None],
                (block_m, half_groups, scale_repeat),
            ).reshape((block_m, half_head_dim))
        output_low = (
            accumulator_low.to(tl.float32)
            * tl.exp2(expanded_exponent_low.to(tl.float32) - running_max[:, None])
            * (1.0 / 255.0)
            / denominator_safe
        )
        output_high = (
            accumulator_high.to(tl.float32)
            * tl.exp2(expanded_exponent_high.to(tl.float32) - running_max[:, None])
            * (1.0 / 255.0)
            / denominator_safe
        )
    else:
        output_low = accumulator_low / denominator_safe
        output_high = accumulator_high / denominator_safe
    output_base = output_ptr + (batch_head * query_length + offsets_m[:, None]) * head_dim
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


@triton.jit
def _uint8_run_scaled_output_pv_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    value_scale_ratio_ptr,
    output_ptr,
    query_length,
    key_length,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    scale_run_n: tl.constexpr,
    feature_group: tl.constexpr,
    global_probability_codes: tl.constexpr,
    dominant_weight_merge: tl.constexpr,
    scaled_fp16_numerator: tl.constexpr,
    unmasked_self_attention: tl.constexpr,
):
    """Keep PV in one sorted run's V-scale coordinate."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    half_groups: tl.constexpr = half_head_dim // feature_group
    feature_groups: tl.constexpr = head_dim // feature_group
    offsets_vd = tl.arange(0, half_head_dim)
    offsets_g = tl.arange(0, half_groups)
    if unmasked_self_attention:
        valid_queries = tl.full((block_m,), True, dtype=tl.int1)
    else:
        valid_queries = offsets_m < query_length
    batch_head = batch * heads + head

    query = tl.load(
        query_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    query_scale = tl.load(
        query_scale_ptr
        + batch_head * tl.cdiv(query_length, 32)
        + offsets_m // 32,
        mask=valid_queries,
        other=0.0,
    )
    if scaled_fp16_numerator:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
    else:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    for start_n in tl.range(0, key_length, block_n, disable_licm=True):
        if start_n % scale_run_n == 0:
            scale_base = (
                batch_head * tl.cdiv(key_length, scale_run_n)
                + start_n // scale_run_n
            ) * feature_groups
            compact_ratio_low = tl.load(
                value_scale_ratio_ptr + scale_base + offsets_g
            )
            compact_ratio_high = tl.load(
                value_scale_ratio_ptr + scale_base + half_groups + offsets_g
            )
            ratio_low = tl.broadcast_to(
                compact_ratio_low[:, None],
                (half_groups, feature_group),
            ).reshape((half_head_dim,)).to(tl.float32)
            ratio_high = tl.broadcast_to(
                compact_ratio_high[:, None],
                (half_groups, feature_group),
            ).reshape((half_head_dim,)).to(tl.float32)
            if scaled_fp16_numerator:
                accumulator_low *= ratio_low[None, :].to(tl.float16)
                accumulator_high *= ratio_high[None, :].to(tl.float16)
            else:
                accumulator_low *= ratio_low[None, :]
                accumulator_high *= ratio_high[None, :]

        current_n = start_n + offsets_n
        key = key_ptr.load([batch_head, start_n, 0]).reshape(
            (block_n, head_dim)
        )
        integer_scores = tl.dot(query, key.T, out_dtype=tl.int32)
        key_scale = tl.load(
            key_scale_ptr
            + batch_head * tl.cdiv(key_length, 64)
            + current_n // 64
        )
        scores = (
            integer_scores.to(tl.float32)
            * query_scale[:, None]
            * key_scale[None, :]
        )
        if unmasked_self_attention:
            valid_keys = tl.full((block_m, block_n), True, dtype=tl.int1)
        else:
            valid_keys = current_n[None, :] < key_length
            scores = tl.where(
                valid_queries[:, None] & valid_keys,
                scores,
                -float("inf"),
            )

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        if unmasked_self_attention:
            old_weight = tl.exp2(running_max - next_max)
            if global_probability_codes:
                current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
                probabilities = tl.exp2(scores - next_max[:, None])
            else:
                current_weight = tl.exp2(block_max - next_max)
                probabilities = tl.exp2(scores - block_max[:, None])
        else:
            old_weight = tl.where(
                valid_queries,
                tl.exp2(running_max - next_max),
                0.0,
            )
            if global_probability_codes:
                current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
                probabilities = tl.where(
                    valid_queries[:, None] & valid_keys,
                    tl.exp2(scores - next_max[:, None]),
                    0.0,
                )
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
        probability_sum = tl.sum(probabilities, axis=1)
        next_denominator = denominator * old_weight + probability_sum * current_weight
        if scaled_fp16_numerator:
            denominator = next_denominator
        elif dominant_weight_merge:
            block_wins = block_max > running_max
            merge_weight = tl.where(block_wins, old_weight, current_weight)
            denominator_dominant = tl.where(
                block_wins, probability_sum, denominator
            )
            denominator_subordinate = tl.where(
                block_wins, denominator, probability_sum
            )
            denominator = (
                denominator_dominant + denominator_subordinate * merge_weight
            )
        else:
            accumulator_low *= old_weight[:, None]
            accumulator_high *= old_weight[:, None]
            denominator = next_denominator
        probability_codes = (probabilities * 255.0 + 0.5).to(tl.uint8)

        value_low = value_ptr.load([batch_head, 0, start_n]).reshape(
            (half_head_dim, block_n)
        ).T
        partial_low = uint8_int8_dot(probability_codes, value_low)
        if scaled_fp16_numerator:
            partial_low_scaled_fp16 = (
                partial_low.to(tl.float32) * (1.0 / 65536.0)
            ).to(tl.float16)
        else:
            partial_low_fp32 = partial_low.to(tl.float32)
        if scaled_fp16_numerator:
            accumulator_low = (
                accumulator_low * old_weight[:, None].to(tl.float16)
                + partial_low_scaled_fp16
                * current_weight[:, None].to(tl.float16)
            )
        elif dominant_weight_merge:
            accumulator_low_dominant = tl.where(
                block_wins[:, None], partial_low_fp32, accumulator_low
            )
            accumulator_low_subordinate = tl.where(
                block_wins[:, None], accumulator_low, partial_low_fp32
            )
            accumulator_low = (
                accumulator_low_dominant
                + accumulator_low_subordinate * merge_weight[:, None]
            )
        else:
            accumulator_low += partial_low_fp32 * current_weight[:, None]

        value_high = value_ptr.load(
            [batch_head, half_head_dim, start_n]
        ).reshape((half_head_dim, block_n)).T
        partial_high = uint8_int8_dot(probability_codes, value_high)
        if scaled_fp16_numerator:
            partial_high_scaled_fp16 = (
                partial_high.to(tl.float32) * (1.0 / 65536.0)
            ).to(tl.float16)
        else:
            partial_high_fp32 = partial_high.to(tl.float32)
        if scaled_fp16_numerator:
            accumulator_high = (
                accumulator_high * old_weight[:, None].to(tl.float16)
                + partial_high_scaled_fp16
                * current_weight[:, None].to(tl.float16)
            )
        elif dominant_weight_merge:
            accumulator_high_dominant = tl.where(
                block_wins[:, None], partial_high_fp32, accumulator_high
            )
            accumulator_high_subordinate = tl.where(
                block_wins[:, None], accumulator_high, partial_high_fp32
            )
            accumulator_high = (
                accumulator_high_dominant
                + accumulator_high_subordinate * merge_weight[:, None]
            )
        else:
            accumulator_high += partial_high_fp32 * current_weight[:, None]
        running_max = next_max

    final_scale_base = (
        batch_head * tl.cdiv(key_length, scale_run_n)
        + tl.cdiv(key_length, scale_run_n)
        - 1
    ) * feature_groups
    compact_final_scale_low = tl.load(
        value_scale_ptr + final_scale_base + offsets_g
    )
    compact_final_scale_high = tl.load(
        value_scale_ptr + final_scale_base + half_groups + offsets_g
    )
    final_scale_low = tl.broadcast_to(
        compact_final_scale_low[:, None],
        (half_groups, feature_group),
    ).reshape((half_head_dim,))
    final_scale_high = tl.broadcast_to(
        compact_final_scale_high[:, None],
        (half_groups, feature_group),
    ).reshape((half_head_dim,))
    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    numerator_restore: tl.constexpr = 65536.0 if scaled_fp16_numerator else 1.0
    output_base = output_ptr + (
        batch_head * query_length + offsets_m[:, None]
    ) * head_dim
    tl.store(
        output_base + offsets_vd[None, :],
        accumulator_low.to(tl.float32)
        * numerator_restore
        * final_scale_low[None, :]
        / denominator_safe,
        mask=valid_queries[:, None],
    )
    tl.store(
        output_base + half_head_dim + offsets_vd[None, :],
        accumulator_high.to(tl.float32)
        * numerator_restore
        * final_scale_high[None, :]
        / denominator_safe,
        mask=valid_queries[:, None],
    )


def _prepare_int8_pv_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    value_transposed: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Prepare Q/K/V once for hot-kernel benchmarking."""
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    statistics_block = 256
    num_partials = (key_length + statistics_block - 1) // statistics_block
    partial_shape = (batch, heads, num_partials, head_dim)
    key_sum_partial = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    value_max_partial = torch.empty_like(key_sum_partial)
    _sage_backend._kv_statistics_partial_kernel[(num_partials, batch * heads)](
        key,
        value,
        key_sum_partial,
        value_max_partial,
        key_length,
        num_partials,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=statistics_block,
        num_warps=4,
    )

    key_mean = torch.empty((batch, heads, head_dim), device=query.device, dtype=torch.float32)
    folded_fp8_scale = torch.empty_like(key_mean)
    _sage_backend._finish_kv_statistics_kernel[(triton.cdiv(head_dim, 32), batch * heads)](
        key_sum_partial,
        value_max_partial,
        key_mean,
        folded_fp8_scale,
        key_length,
        num_partials,
        head_dim=head_dim,
        partial_block=triton.next_power_of_2(num_partials),
        block_d=32,
        num_warps=4,
    )

    query_int8 = torch.empty(query.shape, device=query.device, dtype=torch.int8)
    key_int8 = torch.empty(key.shape, device=query.device, dtype=torch.int8)
    if grouped_qk:
        query_scale = torch.empty(
            (batch, heads, (query_length + 31) // 32),
            device=query.device,
            dtype=torch.float32,
        )
        key_scale = torch.empty(
            (batch, heads, (key_length + 63) // 64),
            device=query.device,
            dtype=torch.float32,
        )
        _sage_backend._quantize_query_per_warp_kernel[
            (triton.cdiv(query_length, 32), heads, batch)
        ](
            query,
            query_int8,
            query_scale,
            query_length,
            scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query_int8.stride(0),
            query_int8.stride(1),
            query_int8.stride(2),
            query_scale.stride(0),
            query_scale.stride(1),
            head_dim=head_dim,
            quantization_range=127,
            rotation_group=0,
            num_warps=4,
        )
        _sage_backend._quantize_key_per_block_kernel[(triton.cdiv(key_length, 64), heads, batch)](
            key,
            key_mean,
            key_int8,
            key_scale,
            key_length,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key_int8.stride(0),
            key_int8.stride(1),
            key_int8.stride(2),
            key_scale.stride(0),
            key_scale.stride(1),
            heads=heads,
            head_dim=head_dim,
            block_n=64,
            quantization_range=127,
            rotation_group=0,
            num_warps=4,
        )
    else:
        query_scale = torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        key_scale = torch.empty(key.shape[:3], device=query.device, dtype=torch.float32)
        _sage_backend._quantize_query_kernel[(triton.cdiv(query_length, 32) * 8, heads, batch)](
            query,
            query_int8,
            query_scale,
            query_length,
            scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query_int8.stride(0),
            query_int8.stride(1),
            query_int8.stride(2),
            query_scale.stride(0),
            query_scale.stride(1),
            head_dim=head_dim,
            quantization_range=127,
            rotation_group=0,
            num_warps=4,
        )
        _sage_backend._quantize_key_kernel[(triton.cdiv(key_length, 64) * 4, heads, batch)](
            key,
            key_mean,
            key_int8,
            key_scale,
            key_length,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key_int8.stride(0),
            key_int8.stride(1),
            key_int8.stride(2),
            key_scale.stride(0),
            key_scale.stride(1),
            heads=heads,
            head_dim=head_dim,
            quantization_range=127,
            rotation_group=0,
            num_warps=4,
        )

    value_int8_shape = (batch, heads, head_dim, key_length) if value_transposed else value.shape
    value_int8 = torch.empty(value_int8_shape, device=query.device, dtype=torch.int8)
    _quantize_value_int8_kernel[(triton.cdiv(key_length, 64), heads, batch)](
        value,
        folded_fp8_scale,
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
        block_n=64,
        output_transposed=value_transposed,
        num_warps=4,
    )
    return query_int8, key_int8, value_int8, query_scale, key_scale, folded_fp8_scale


def _launch_int8_pv_attention(
    prepared: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    is_causal: bool,
    *,
    grouped_qk: bool,
    block_m: int,
    num_warps: int,
    num_stages: int,
    block_n: int = 64,
    block_scaled_pv: bool = False,
    block_global_probability: bool = False,
    split_pv_head_dim: bool = False,
    native_unsigned_probability: bool = False,
    integer_pv_recurrence: bool = False,
    raw_integer_pv_recurrence: bool = False,
    raw_fp32_pv_recurrence: bool = False,
    magic_score_conversion: bool = False,
    magic_pv_conversion: bool = False,
    fp16_pv_conversion: bool = False,
    bf16_pv_conversion: bool = False,
    unmasked_self_attention: bool = False,
    value_transposed: bool = True,
    use_tensor_descriptors: bool = False,
    maxnreg: int | None = None,
) -> torch.Tensor:
    query, key, value, query_scale, key_scale, value_scale = prepared
    if native_unsigned_probability:
        enable_uint8_int8_dot()
    batch, heads, _, head_dim = query.shape
    if unmasked_self_attention and (
        is_causal
        or query_length != key_length
        or query_length % block_m
        or key_length % block_n
    ):
        raise ValueError(
            "unmasked self-attention requires noncausal self-attention with complete M/K tiles"
        )
    if split_pv_head_dim and head_dim != 128:
        raise ValueError("split integer PV requires D128")
    if block_global_probability and not block_scaled_pv:
        raise ValueError("global P is a block-scaled PV control")
    if (integer_pv_recurrence or raw_integer_pv_recurrence) and (
        split_pv_head_dim or block_scaled_pv
    ):
        raise ValueError("integer fixed-INT8 recurrence requires unsplit fixed scaling")
    key_argument, value_argument = _sage_backend._make_attention_arguments(
        key,
        value,
        batch,
        heads,
        key_length,
        head_dim,
        value_transposed,
        use_tensor_descriptors,
        head_dim // 2 if split_pv_head_dim else None,
        block_n,
    )
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    kernel = cast(Any, _int8_pv_attention_kernel)
    kernel[(triton.cdiv(query_length, block_m), heads, batch)](
        query,
        key_argument,
        value_argument,
        query_scale,
        key_scale,
        value_scale,
        output,
        query_length,
        key_length,
        is_causal=is_causal,
        grouped_qk=grouped_qk,
        block_scaled_pv=block_scaled_pv,
        block_global_probability=block_global_probability,
        split_pv_head_dim=split_pv_head_dim,
        native_unsigned_probability=native_unsigned_probability,
        integer_pv_recurrence=integer_pv_recurrence,
        raw_integer_pv_recurrence=raw_integer_pv_recurrence,
        raw_fp32_pv_recurrence=raw_fp32_pv_recurrence,
        magic_score_conversion=magic_score_conversion,
        magic_pv_conversion=magic_pv_conversion,
        fp16_pv_conversion=fp16_pv_conversion,
        bf16_pv_conversion=bf16_pv_conversion,
        unmasked_self_attention=unmasked_self_attention,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        value_transposed=value_transposed,
        use_tensor_descriptors=use_tensor_descriptors,
        **launch_options,
    )
    return output


def triton_sage_attention_int8_pv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
    split_pv_head_dim: bool | None = None,
    native_unsigned_probability: bool = False,
) -> torch.Tensor:
    """Run the end-to-end fixed-scale integer PV baseline.

    Complete-tile D128 noncausal self-attention uses a predicate-free split-D64
    schedule on consumer Blackwell, including M128 from N=8192.
    Pass an explicit boolean to select the split or unsplit kernel directly.
    Native UINT8 P uses Piper's stock-Triton compiler extension.
    """
    if grouped_qk is None:
        grouped_qk = torch.cuda.get_device_capability(query.device)[0] == 12
    prepared = _prepare_int8_pv_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
        value_transposed=True,
    )
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    output = torch.empty_like(query)
    device_major = torch.cuda.get_device_capability(query.device)[0]
    complete_sm120_self_attention = (
        device_major == 12
        and not is_causal
        and head_dim == 128
        and query_length == key_length
        and query_length % (128 if key_length >= 8192 else 64) == 0
        and key_length % (64 if key_length > 512 else 128) == 0
    )
    if split_pv_head_dim is None:
        split_pv_head_dim = (
            complete_sm120_self_attention
            or (
                device_major == 12
                and not is_causal
                and head_dim == 128
                and query_length == key_length
                and key_length < 8192
                and key_length % 16 == 0
            )
        )
    if split_pv_head_dim:
        if complete_sm120_self_attention and key_length >= 8192:
            block_m = 128
            block_n = 64
            num_stages = 3
            use_tensor_descriptors = True
        else:
            block_m = 64
            block_n = 128 if key_length <= 512 else 64
            num_stages = 2 if key_length <= 512 else 3
            use_tensor_descriptors = _sage_backend._should_use_split_pv_tensor_descriptors(
                query,
                block_m,
                head_dim,
                key_length,
                True,
            )
    else:
        block_m = (
            64
            if is_causal
            else _sage_backend._select_query_block(query, batch, heads, query_length)
        )
        block_n = 64
        num_stages = 3
        use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
            query,
            block_m,
            head_dim,
            key_length,
            True,
        )
    maxnreg = (
        248
        if (
            device_major == 12
            and not split_pv_head_dim
            and not complete_sm120_self_attention
            and not is_causal
            and block_m == 128
            and head_dim == 128
            and key_length >= 8192
        )
        else None
    )
    return _launch_int8_pv_attention(
        prepared,
        output,
        query_length,
        key_length,
        is_causal,
        grouped_qk=grouped_qk,
        block_m=block_m,
        block_n=block_n,
        num_warps=4,
        num_stages=num_stages,
        split_pv_head_dim=split_pv_head_dim,
        native_unsigned_probability=native_unsigned_probability,
        unmasked_self_attention=complete_sm120_self_attention,
        value_transposed=True,
        use_tensor_descriptors=use_tensor_descriptors,
        maxnreg=maxnreg,
    )


def _prepare_block_int8_pv_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    value_transposed: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Prepare canonical Q/K plus 64-token block-scaled signed-INT8 V."""
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        scale,
        grouped_qk,
    )
    batch, heads, key_length, head_dim = value.shape
    value_int8_shape = (batch, heads, head_dim, key_length) if value_transposed else value.shape
    value_int8 = torch.empty(value_int8_shape, device=value.device, dtype=torch.int8)
    value_scale = torch.empty(
        (batch, heads, (key_length + 63) // 64, head_dim),
        device=value.device,
        dtype=torch.float32,
    )
    _quantize_value_block_int8_kernel[(triton.cdiv(key_length, 64), heads, batch)](
        value,
        value_scale,
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
        block_n=64,
        output_transposed=value_transposed,
        num_warps=4,
    )
    return query_int8, key_int8, value_int8, query_scale, key_scale, value_scale


def _prepare_uint8_k32_feature_pv_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    """Prepare grouped Q/K and per-K32/per-feature INT8 V for profiling."""
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        scale,
        grouped_qk=True,
    )
    batch, heads, key_length, head_dim = value.shape
    value_int8 = torch.empty(
        (batch, heads, head_dim, key_length),
        device=value.device,
        dtype=torch.int8,
    )
    value_scale = torch.empty(
        (batch, heads, (key_length + 31) // 32, head_dim),
        device=value.device,
        dtype=torch.float16,
    )
    _quantize_value_block_int8_kernel[(triton.cdiv(key_length, 32), heads, batch)](
        value,
        value_scale,
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
        block_n=32,
        output_transposed=True,
        num_warps=4,
    )
    return query_int8, key_int8, value_int8, query_scale, key_scale, value_scale


def _launch_uint8_k32_feature_pv_attention(
    prepared: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    *,
    block_m: int,
    num_warps: int,
    num_stages: int,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """Launch the profiler-only native-UINT8 K32 output-scale kernel."""
    enable_uint8_int8_dot()
    query, key, value, query_scale, key_scale, value_scale = prepared
    batch, heads, _, head_dim = query.shape
    if head_dim != 128 or key_length % 64:
        raise ValueError("K32 feature-scaled profiling requires D128 and K divisible by 64")
    key_argument = TensorDescriptor(
        key,
        shape=[batch * heads, key_length, head_dim],
        strides=[key_length * head_dim, head_dim, 1],
        block_shape=[1, 64, head_dim],
    )
    value_argument = TensorDescriptor(
        value,
        shape=[batch * heads, head_dim, key_length],
        strides=[head_dim * key_length, key_length, 1],
        block_shape=[1, head_dim // 2, 32],
    )
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    kernel = cast(Any, _uint8_k32_feature_pv_attention_kernel)
    kernel[(triton.cdiv(query_length, block_m), heads, batch)](
        query,
        key_argument,
        value_argument,
        query_scale,
        key_scale,
        value_scale,
        output,
        query_length,
        key_length,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=64,
        **launch_options,
    )
    return output


def _prepare_uint8_grouped_output_pv_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    feature_group: int,
    block_n: int = 64,
    scale_run_n: int | None = None,
    integer_output_recurrence: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Prepare grouped Q/K and grouped-feature INT8 V for profiling."""
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        scale,
        grouped_qk=True,
    )
    batch, heads, key_length, head_dim = value.shape
    if head_dim % feature_group:
        raise ValueError("feature group must divide the head dimension")
    scale_block_n = scale_run_n or block_n
    if scale_block_n < block_n or scale_block_n % block_n:
        raise ValueError("scale run must be a multiple of the attention block")
    if scale_run_n is not None and integer_output_recurrence:
        raise ValueError("scale runs currently require the FP32 output recurrence")
    value_int8 = torch.empty(
        (batch, heads, head_dim, key_length),
        device=value.device,
        dtype=torch.int8,
    )
    feature_groups = head_dim // feature_group
    value_scale = torch.empty(
        (
            batch,
            heads,
            (key_length + scale_block_n - 1) // scale_block_n,
            feature_groups,
        ),
        device=value.device,
        dtype=torch.float16,
    )
    if scale_run_n is None:
        _quantize_value_grouped_int8_kernel[
            (triton.cdiv(key_length, block_n), heads, batch)
        ](
            value,
            value_scale,
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
            block_n=block_n,
            feature_group=feature_group,
            store_log2_scale=integer_output_recurrence,
            output_transposed=True,
            num_warps=4,
        )
    else:
        scale_max = torch.zeros_like(value_scale, dtype=torch.float32)
        quantize_grid = (triton.cdiv(key_length, block_n), heads, batch)
        _reduce_grouped_value_run_scale_kernel[quantize_grid](
            value,
            scale_max,
            key_length,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            heads=heads,
            head_dim=head_dim,
            block_n=block_n,
            scale_run_n=scale_run_n,
            feature_group=feature_group,
            num_warps=4,
        )
        _quantize_value_grouped_int8_with_run_scale_kernel[quantize_grid](
            value,
            scale_max,
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
            block_n=block_n,
            scale_run_n=scale_run_n,
            feature_group=feature_group,
            output_transposed=True,
            num_warps=4,
        )
        scale_elements = value_scale.numel()
        scale_finalize_block = 256
        _finalize_grouped_value_run_scale_kernel[
            (triton.cdiv(scale_elements, scale_finalize_block),)
        ](
            scale_max,
            value_scale,
            scale_elements,
            block=scale_finalize_block,
            num_warps=4,
        )
    prepared = (
        query_int8,
        key_int8,
        value_int8,
        query_scale,
        key_scale,
        value_scale,
    )
    if scale_run_n is None:
        return prepared
    scale_runs = triton.cdiv(key_length, scale_block_n)
    value_scale_ratio = torch.empty(
        value_scale.shape,
        device=value.device,
        dtype=value_scale.dtype,
    )
    _grouped_scale_ratio_kernel[(scale_runs, heads, batch)](
        value_scale,
        value_scale_ratio,
        heads=heads,
        scale_runs=scale_runs,
        feature_groups=feature_groups,
        num_warps=1,
    )
    return (*prepared, value_scale_ratio)


def _pack_bucketed_rotated_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    range_bucket_log2_scale: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack noncausal K/V by rotated V range using only Triton kernels."""
    if key.shape != value.shape or key.ndim != 4 or key.shape[-1] != 128:
        raise ValueError("bucketed K/V packing requires matching [B, H, K, 128] tensors")
    batch, heads, key_length, head_dim = key.shape
    if range_bucket_log2_scale not in (1, 2, 4):
        raise ValueError("range bucket log2 scale must be 1, 2, or 4")
    range_buckets = _BASE_RANGE_BUCKETS * range_bucket_log2_scale
    range_bucket_offset = _BASE_RANGE_BUCKET_OFFSET * range_bucket_log2_scale
    packed_key = torch.empty_like(key)
    packed_value = torch.empty_like(value)
    buckets = torch.empty(
        (batch, heads, key_length),
        device=key.device,
        dtype=torch.int32,
    )
    histogram = torch.zeros(
        (batch, heads, range_buckets),
        device=key.device,
        dtype=torch.int32,
    )
    offsets = torch.empty_like(histogram)
    cursors = torch.empty_like(histogram)
    histogram_block = 32
    _bucket_histogram_kernel[(triton.cdiv(key_length, histogram_block), heads, batch)](
        value,
        buckets,
        histogram,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        heads=heads,
        head_dim=head_dim,
        range_buckets=range_buckets,
        range_bucket_offset=range_bucket_offset,
        range_bucket_log2_scale=range_bucket_log2_scale,
        block_n=histogram_block,
        num_warps=4,
    )
    _bucket_prefix_kernel[(batch * heads,)](
        histogram,
        offsets,
        cursors,
        range_buckets=range_buckets,
        num_warps=1,
    )
    scatter_block = 16
    _bucket_scatter_kv_kernel[(triton.cdiv(key_length, scatter_block), heads, batch)](
        key,
        value,
        buckets,
        offsets,
        cursors,
        packed_key,
        packed_value,
        key_length,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        heads=heads,
        head_dim=head_dim,
        range_buckets=range_buckets,
        block_n=scatter_block,
        num_warps=4,
    )
    return packed_key, packed_value


def _prepare_bucketed_grouped_output_pv_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    feature_group: int,
    block_n: int,
    range_bucket_log2_scale: int = 2,
    scale_run_n: int | None = None,
    integer_output_recurrence: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Pack K/V by V range, rotate V by H64, then quantize attention inputs."""
    packed_key, packed_value = _pack_bucketed_rotated_kv(
        key,
        value,
        range_bucket_log2_scale=range_bucket_log2_scale,
    )
    return _prepare_uint8_grouped_output_pv_inputs(
        query,
        packed_key,
        packed_value,
        scale,
        feature_group=feature_group,
        block_n=block_n,
        scale_run_n=scale_run_n,
        integer_output_recurrence=integer_output_recurrence,
    )


def _launch_uint8_grouped_output_pv_attention(
    prepared: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    *,
    feature_group: int,
    block_n: int = 64,
    block_m: int,
    num_warps: int,
    num_stages: int,
    local_probability_codes: bool = True,
    integer_output_recurrence: bool = False,
    common_feature_exponent: bool = False,
    unmasked_self_attention: bool = False,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """Launch the profiler-only grouped-output-scale native-UINT8 kernel."""
    enable_uint8_int8_dot()
    if integer_output_recurrence and not local_probability_codes:
        raise ValueError("INT32 output recurrence requires tile-local probability codes")
    if common_feature_exponent and not integer_output_recurrence:
        raise ValueError("a common feature exponent requires INT32 output recurrence")
    query, key, value, query_scale, key_scale, value_scale = prepared
    batch, heads, _, head_dim = query.shape
    if unmasked_self_attention and (
        query_length != key_length
        or query_length % block_m
        or key_length % block_n
    ):
        raise ValueError(
            "unmasked grouped-output attention requires self-attention with complete M/K tiles"
        )
    if (
        head_dim != 128
        or key_length % block_n
        or (feature_group != head_dim and 64 % feature_group)
    ):
        raise ValueError(
            "grouped output-scale profiling requires D128, K divisible by 64, "
            "and a feature group that divides D64"
        )
    key_argument = TensorDescriptor(
        key,
        shape=[batch * heads, key_length, head_dim],
        strides=[key_length * head_dim, head_dim, 1],
        block_shape=[1, block_n, head_dim],
    )
    value_argument = TensorDescriptor(
        value,
        shape=[batch * heads, head_dim, key_length],
        strides=[head_dim * key_length, key_length, 1],
        block_shape=[1, head_dim // 2, block_n],
    )
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    kernel = cast(Any, _uint8_grouped_output_pv_attention_kernel)
    kernel[(triton.cdiv(query_length, block_m), heads, batch)](
        query,
        key_argument,
        value_argument,
        query_scale,
        key_scale,
        value_scale,
        output,
        query_length,
        key_length,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        feature_group=feature_group,
        local_probability_codes=local_probability_codes,
        integer_output_recurrence=integer_output_recurrence,
        common_feature_exponent=common_feature_exponent,
        unmasked_self_attention=unmasked_self_attention,
        **launch_options,
    )
    return output


def _launch_uint8_run_scaled_output_pv_attention(
    prepared: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    *,
    feature_group: int,
    block_n: int,
    scale_run_n: int,
    block_m: int,
    num_warps: int,
    num_stages: int,
    global_probability_codes: bool = False,
    dominant_weight_merge: bool = False,
    scaled_fp16_numerator: bool = False,
    unmasked_self_attention: bool = False,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """Launch the sorted-run V-scale amortization kernel."""
    enable_uint8_int8_dot()
    query, key, value, query_scale, key_scale, value_scale, value_scale_ratio = prepared
    batch, heads, _, head_dim = query.shape
    if unmasked_self_attention and (
        query_length != key_length
        or query_length % block_m
        or key_length % block_n
    ):
        raise ValueError(
            "unmasked run-scaled attention requires self-attention with complete M/K tiles"
        )
    if (
        head_dim != 128
        or key_length % block_n
        or scale_run_n < block_n
        or scale_run_n % block_n
        or 64 % feature_group
    ):
        raise ValueError(
            "run-scaled output attention requires D128, a K-divisible scale run, "
            "and feature groups that divide D64"
        )
    key_argument = TensorDescriptor(
        key,
        shape=[batch * heads, key_length, head_dim],
        strides=[key_length * head_dim, head_dim, 1],
        block_shape=[1, block_n, head_dim],
    )
    value_argument = TensorDescriptor(
        value,
        shape=[batch * heads, head_dim, key_length],
        strides=[head_dim * key_length, key_length, 1],
        block_shape=[1, head_dim // 2, block_n],
    )
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg
    kernel = cast(Any, _uint8_run_scaled_output_pv_attention_kernel)
    kernel[(triton.cdiv(query_length, block_m), heads, batch)](
        query,
        key_argument,
        value_argument,
        query_scale,
        key_scale,
        value_scale,
        value_scale_ratio,
        output,
        query_length,
        key_length,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        scale_run_n=scale_run_n,
        feature_group=feature_group,
        global_probability_codes=global_probability_codes,
        dominant_weight_merge=dominant_weight_merge,
        scaled_fp16_numerator=scaled_fp16_numerator,
        unmasked_self_attention=unmasked_self_attention,
        **launch_options,
    )
    return output


def triton_sage_attention_uint8_pv_bucketed_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    feature_group: int = 4,
    block_n: int = 128,
    range_bucket_log2_scale: int = 2,
    scale_run_n: int | None = None,
    local_probability_codes: bool = True,
    integer_output_recurrence: bool = False,
    common_feature_exponent: bool = False,
    scaled_fp16_numerator: bool = False,
    grouped_qk: bool | None = None,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """Run the experimental packed grouped-output UINT8-by-INT8 path.

    This validation wrapper explicitly applies the inverse H64 output rotation.
    Model integration should instead fold H64 into the V and output projections.
    """
    if is_causal:
        raise ValueError("bucketed K/V packing is valid only for noncausal attention")
    if scale_run_n is not None and (
        not local_probability_codes
        or integer_output_recurrence
        or common_feature_exponent
    ):
        raise ValueError("scale runs require a tile-local probability recurrence")
    if scaled_fp16_numerator and (
        scale_run_n is None
        or block_n != 64
        or query.shape[2] % 128
        or query.shape[2] > 131072
    ):
        raise ValueError(
            "scaled FP16 numerator requires K64 scale runs and at most 131072 keys"
        )
    del grouped_qk
    query_length, head_dim = query.shape[2:]
    key_length = key.shape[2]
    if (
        torch.cuda.get_device_capability(query.device)[0] != 12
        or head_dim != 128
        or query_length != key_length
        or key_length % block_n
    ):
        raise ValueError(
            "bucketed grouped-output attention currently requires SM12x D128 "
            "self-attention with K divisible by its block size"
        )
    prepared = _prepare_bucketed_grouped_output_pv_inputs(
        query,
        key,
        value,
        scale,
        feature_group=feature_group,
        block_n=block_n,
        range_bucket_log2_scale=range_bucket_log2_scale,
        scale_run_n=scale_run_n,
        integer_output_recurrence=integer_output_recurrence,
    )
    rotated_output = torch.empty_like(query)
    if scale_run_n is not None:
        _launch_uint8_run_scaled_output_pv_attention(
            prepared,
            rotated_output,
            query_length,
            key_length,
            feature_group=feature_group,
            block_n=block_n,
            scale_run_n=scale_run_n,
            block_m=128 if scaled_fp16_numerator else 64,
            num_warps=4,
            num_stages=2,
            scaled_fp16_numerator=scaled_fp16_numerator,
            unmasked_self_attention=True,
            maxnreg=maxnreg,
        )
    else:
        _launch_uint8_grouped_output_pv_attention(
            prepared,
            rotated_output,
            query_length,
            key_length,
            feature_group=feature_group,
            block_n=block_n,
            block_m=64,
            num_warps=4,
            num_stages=2,
            local_probability_codes=local_probability_codes,
            integer_output_recurrence=integer_output_recurrence,
            common_feature_exponent=common_feature_exponent,
            maxnreg=maxnreg,
        )
    return rotate_attention_rows(rotated_output, 64)


def triton_sage_attention_int8_pv_block_scaled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
) -> torch.Tensor:
    """Run signed-INT8 PV with block-local P normalization and V scales."""
    if grouped_qk is None:
        grouped_qk = torch.cuda.get_device_capability(query.device)[0] == 12
    prepared = _prepare_block_int8_pv_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
        value_transposed=True,
    )
    batch, heads, query_length, _ = query.shape
    key_length = key.shape[2]
    output = torch.empty_like(query)
    block_m = (
        64 if is_causal else _sage_backend._select_query_block(query, batch, heads, query_length)
    )
    use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
        query,
        block_m,
        query.shape[-1],
        key_length,
        True,
    )
    return _launch_int8_pv_attention(
        prepared,
        output,
        query_length,
        key_length,
        is_causal,
        grouped_qk=grouped_qk,
        block_m=block_m,
        num_warps=4,
        num_stages=3,
        block_scaled_pv=True,
        value_transposed=True,
        use_tensor_descriptors=use_tensor_descriptors,
    )
