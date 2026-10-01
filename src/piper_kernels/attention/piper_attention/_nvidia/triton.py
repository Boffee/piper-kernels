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

from dataclasses import dataclass
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
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

from .. import _quantization
from .._validation import resolve_query_window, validate_output_buffer, validate_query_offset
from . import gluon_async_copy as _gluon_async_copy
from . import policy as _policy
from ._plan import PiperAttentionExecutionPlan

_BLOCK_N = 64
_P_UINT8_RANGE = tl.constexpr(255.0)
_P_UINT8_LOG2_RANGE = tl.constexpr(7.994353436858858)
# Pad the analytical maximum to the next effective FP32 constant so the
# lowered expression remains an upper bound after integer-to-FP32 rounding.
_VALUE_LOG_BOUND_CORRECTION = tl.constexpr(0.086085)


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
def _conservative_value_log_scale_bound(value_scale_multiplier):
    """Bound log2(scale) from the positive FP32 multiplier's IEEE-754 bits."""
    multiplier_bits = value_scale_multiplier.to(tl.int32, bitcast=True)
    return multiplier_bits.to(tl.float32) * (1.0 / 8388608.0) - (
        127.0 + _P_UINT8_LOG2_RANGE - _VALUE_LOG_BOUND_CORRECTION
    )


@triton.jit(do_not_specialize=["key_length", "heads"])
def _quantize_value_per_key_kernel(
    value_ptr,
    value_mean_ptr,
    scale_multiplier_ptr,
    log_scale_ptr,
    output_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_od,
    stride_ok,
    is_causal: tl.constexpr,
    store_log_scale: tl.constexpr,
    heads,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Standalone launcher for the shared per-key V component."""
    _quantization.quantize_value_per_key_block(
        value_ptr,
        value_mean_ptr,
        output_ptr,
        scale_multiplier_ptr,
        log_scale_ptr,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        key_length,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        stride_od,
        stride_ok,
        is_causal,
        store_log_scale,
        heads,
        head_dim,
        block_n,
    )


# The same launcher without specializing on the key-length-dependent V row
# stride: one variant for every key length, and on SM89 3-5x faster than the
# variant that vectorizes stores for a 16-byte-divisible stride.
_quantize_value_per_key_unspecialized_kernel = triton.jit(
    do_not_specialize=["key_length", "heads", "stride_od"],
)(_quantize_value_per_key_kernel.fn)


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
    running_max,
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
    mode. The common scale is removed once in the epilogue, before non-causal
    value-mean restoration.
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
    if grouped_qk:
        key_scale = tl.load(
            key_scale_ptr + batch_head * tl.cdiv(key_length, block_n) + start_n // block_n
        )
        scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
    else:
        key_scale = tl.load(
            key_scale_ptr + batch_head * key_length + current_n,
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
            value_scale_multiplier_ptr
            + batch_head * (tl.cdiv(key_length, 64) * 64 if padded_kv else key_length)
            + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
        value_log_scale = _conservative_value_log_scale_bound(value_scale_multiplier)
    else:
        value_log_scale = tl.load(
            value_log_scale_ptr
            + batch_head * (tl.cdiv(key_length, 64) * 64 if padded_kv else key_length)
            + current_n,
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
    next_max = tl.maximum(running_max, block_max)
    old_weight = tl.where(
        valid_queries,
        tl.exp2(running_max - next_max),
        0.0,
    )
    current_weight = tl.where(
        valid_queries,
        tl.exp2(block_max - next_max),
        0.0,
    )
    probabilities = tl.where(
        valid_queries[:, None] & valid_keys,
        tl.exp2(scores - block_max[:, None]),  # pyright: ignore[reportArgumentType]
        0.0,
    )
    denominator = denominator * old_weight + tl.sum(probabilities, axis=1) * current_weight
    if not derive_value_log_bound:
        value_scale_multiplier = tl.load(
            value_scale_multiplier_ptr
            + batch_head * (tl.cdiv(key_length, 64) * 64 if padded_kv else key_length)
            + current_n,
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
    return numerator, denominator, next_max


@triton.jit
def _piper_attention_query_tile(  # noqa: PLR0912, PLR0915
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
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
    stride_ob,
    stride_oh,
    stride_om,
    heads,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    unmasked_key_tiles: tl.constexpr,
    head_groups: tl.constexpr,
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
    """Evaluate one complete or masked query tile with the same FP32 recurrence."""
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // head_groups) + head // head_groups
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

    if split_pv_head_dim:
        half_head_dim: tl.constexpr = head_dim // 2
        offsets_vd = tl.arange(0, half_head_dim)
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    else:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, global_query_start + (query_block + 1) * block_m)

    if is_causal and optimize_causal_traversal:
        numerator = (accumulator_low, accumulator_high) if split_pv_head_dim else accumulator  # pyright: ignore[reportPossiblyUnboundVariable]
        # Only complete K tiles strictly before the first query row are
        # mask-free. Keep ragged tails and diagonal overlap in the boundary.
        full_key_end = key_length // block_n * block_n
        causal_prefix_end = (global_query_start + query_block * block_m) // block_n * block_n
        prefix_end = tl.minimum(causal_prefix_end, full_key_end)
        for start_n in tl.range(  # pyright: ignore[reportGeneralTypeIssues]
            0,
            prefix_end,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=not loop_licm,
        ):
            numerator, denominator, running_max = _attention_tile(
                query,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                numerator,
                denominator,
                running_max,
                kv_batch_head,
                start_n,
                global_rows,
                offsets_n,
                offsets_d,
                valid_queries,
                key_length,
                mask_keys=tl.constexpr(False),
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
        for start_n in tl.range(  # pyright: ignore[reportGeneralTypeIssues]
            prefix_end,
            end_n,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=not loop_licm,
        ):
            numerator, denominator, running_max = _attention_tile(
                query,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                numerator,
                denominator,
                running_max,
                kv_batch_head,
                start_n,
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
                padded_kv=padded_kv,
            )
        if split_pv_head_dim:
            accumulator_low, accumulator_high = numerator
        else:
            accumulator = numerator
    else:
        numerator = (accumulator_low, accumulator_high) if split_pv_head_dim else accumulator  # pyright: ignore[reportPossiblyUnboundVariable]
        for start_n in tl.range(  # pyright: ignore[reportGeneralTypeIssues]
            0,
            end_n,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=not loop_licm,
        ):
            numerator, denominator, running_max = _attention_tile(
                query,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                numerator,
                denominator,
                running_max,
                kv_batch_head,
                start_n,
                global_rows,
                offsets_n,
                offsets_d,
                valid_queries,
                key_length,
                mask_keys=tl.constexpr(not unmasked_key_tiles),
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
        if split_pv_head_dim:
            accumulator_low, accumulator_high = numerator
        else:
            accumulator = numerator
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
        output_low = accumulator_low / denominator_code_units  # pyright: ignore[reportPossiblyUnboundVariable]
        output_high = accumulator_high / denominator_code_units  # pyright: ignore[reportPossiblyUnboundVariable]
        if not is_causal:
            value_mean_base = value_mean_ptr + kv_batch_head * head_dim
            output_low += tl.load(value_mean_base + offsets_vd)[None, :]  # pyright: ignore[reportPossiblyUnboundVariable]
            output_high += tl.load(value_mean_base + half_head_dim + offsets_vd)[None, :]  # pyright: ignore[reportPossiblyUnboundVariable]
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
        output = accumulator / denominator_code_units  # pyright: ignore[reportPossiblyUnboundVariable]
        if not is_causal:
            output += tl.load(value_mean_ptr + kv_batch_head * head_dim + offsets_d)[None, :]
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
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    value_mean_ptr,
    output_ptr,
    query_storage_length,
    key_length,
    query_start,
    query_rows,
    global_query_start,
    stride_ob,
    stride_oh,
    stride_om,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    aligned_queries: tl.constexpr,
    unmasked_key_tiles: tl.constexpr,
    heads,
    head_groups: tl.constexpr,
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
    # Preserve the measured full-launch path instead of carrying window
    # coordinates through every Q load. Contiguous outputs likewise retain
    # flattened indexing in the tile epilogue.
    if full_query:
        query_start = 0
        global_query_start = 0
        query_rows = query_storage_length
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    if query_group_size > 0:
        # Visit a small group of Q tiles across all heads before the next
        # group. The final group may contain fewer tiles; map it densely so
        # every (head, query tile) occurs exactly once without padding CTAs.
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
            head_groups,
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
            *tile_args,
            *tile_options,
            unmasked_query_tiles=tl.constexpr(True),
            use_query_tensor_descriptor=use_query_tensor_descriptor,
            padded_kv=padded_kv,
        )
    else:
        _piper_attention_query_tile(
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


@dataclass(frozen=True, slots=True)
class _PreparedPiperContext:
    """Reusable K/V operands and the plan defining their quantization and layout."""

    key: torch.Tensor | TensorDescriptor
    value: torch.Tensor | TensorDescriptor
    key_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_log_scale: torch.Tensor
    value_mean: torch.Tensor
    key_length: int
    is_causal: bool
    execution_plan: PiperAttentionExecutionPlan
    padded_kv: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedPiperQuery:
    """Prepared Q with its unpadded logical shape and original floating dtype.

    With fused query quantization, ``data`` is the floating-point Q, ``scale`` is None, and
    ``softmax_scale`` is applied when the Gluon kernel quantizes each tile.
    """

    data: torch.Tensor
    scale: torch.Tensor | None
    descriptor: TensorDescriptor | None
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    global_row_offset: int = 0
    softmax_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class _PreparedPiperAttention:
    """Prepared K/V and Q with one reusable output for full attention."""

    context: _PreparedPiperContext
    query: _PreparedPiperQuery
    output: torch.Tensor


def _prepare_piper_context(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    execution_plan: PiperAttentionExecutionPlan,
) -> _PreparedPiperContext:
    """Prepare validated K/V once, independently of Q and output storage."""
    batch, kv_heads, key_length, head_dim = key.shape
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

        # A sequence-wide V mean is valid only for non-causal attention. Per-row
        # INT8 rounding would otherwise let future V rows perturb earlier outputs.
        key_mean, value_mean = _quantization.compute_kv_means(
            key,
            value,
            is_causal=is_causal,
        )
        key_int8, key_scale = qk_quantization.prepare_key(
            key,
            key_mean,
            grouped=plan.grouped_qk,
            storage_key_length=storage_key_length,
        )

        value_shape = (batch, kv_heads, head_dim, storage_key_length)
        value_int8 = (
            torch.zeros(value_shape, device=value.device, dtype=torch.int8)
            if storage_key_length != key_length
            else torch.empty(value_shape, device=value.device, dtype=torch.int8)
        )
        value_scale_multiplier = torch.empty(
            (batch, kv_heads, key_length),
            device=value.device,
            dtype=torch.float32,
        )
        value_log_scale = torch.empty(
            (1,) if plan.derive_value_log_bound else (batch, kv_heads, key_length),
            device=value.device,
            dtype=torch.float16,
        )
        value_kernel = (
            _quantize_value_per_key_unspecialized_kernel
            if plan.unspecialized_value_stride
            else _quantize_value_per_key_kernel
        )
        value_kernel[(triton.cdiv(key_length, _BLOCK_N), kv_heads, batch)](
            value,
            value_mean,
            value_scale_multiplier,
            value_log_scale,
            value_int8,
            key_length,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value_int8.stride(0),
            value_int8.stride(1),
            value_int8.stride(2),
            value_int8.stride(3),
            is_causal=is_causal,
            store_log_scale=not plan.derive_value_log_bound,
            heads=kv_heads,
            head_dim=head_dim,
            block_n=_BLOCK_N,
            num_warps=4,
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


def _launch_piper_attention_into(
    context: _PreparedPiperContext,
    query: _PreparedPiperQuery,
    output: torch.Tensor,
    *,
    query_start: int = 0,
    query_rows: int | None = None,
) -> torch.Tensor:
    """Launch a local Q window into caller-owned BHSD output storage.

    Q and K/V must match in batch, head dimension, device, and execution plan;
    Q heads must be divisible by K/V heads. The local start and global origin
    are aligned to plan.block_m; rows may include a ragged tail. Causal masking
    uses the global origin plus local start. query_start is local to prepared Q;
    query_rows=None selects its remaining logical rows. Output may have padded
    or permuted outer strides, with contiguous head features and no overlap.
    """
    batch, heads, query_length, head_dim = query.shape
    plan = context.execution_plan
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
            context.value_scale_multiplier,
            context.value_log_scale,
            context.value_mean,
            output,
            query.data.shape[2],
            context.key_length,
            query_start,
            rows,
            query.global_row_offset + query_start,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            is_causal=context.is_causal,
            grouped_qk=plan.grouped_qk,
            split_pv_head_dim=plan.split_pv_head_dim,
            aligned_queries=aligned_queries,
            unmasked_key_tiles=(not context.is_causal and context.key_length % _BLOCK_N == 0),
            heads=heads,
            head_groups=heads // context.key_scale.shape[1],
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
