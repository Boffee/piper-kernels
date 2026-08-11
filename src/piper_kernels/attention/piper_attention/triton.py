"""Pure-Triton backend for Piper Attention.

The kernel keeps Sage-style INT8 QK and FP32 online softmax, but replaces the
FP8 PV path with one signed-INT8 scale per V row. Each row scale is folded into
the nonnegative probability operand, producing UINT8-by-INT8 tensor-core dots.
Native mixed-sign MMA is used on supported NVIDIA targets; an algebraically
identical affine signed-INT8 formulation is retained as the portable control.
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
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.scheduling import select_query_block

from . import _policy

_BLOCK_N = 64
_MEAN_CHUNK_N = 1024
_MEAN_BLOCK_N = 64
_MEAN_BLOCK_D = 64
_P_UINT8_RANGE = tl.constexpr(255.0)
_P_ZERO_POINT = tl.constexpr(128)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)


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
def _kv_mean_partial_kernel(
    key_ptr,
    value_ptr,
    key_partial_ptr,
    value_partial_ptr,
    key_length,
    num_chunks,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    is_causal: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_n: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """Reduce one raw K chunk and, when non-causal, its V chunk."""
    chunk = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_d = feature_block * block_d + tl.arange(0, block_d)
    offsets_n = tl.arange(0, block_n)
    key_accumulator = tl.zeros((block_d,), dtype=tl.float32)
    if not is_causal:
        value_accumulator = tl.zeros((block_d,), dtype=tl.float32)
    chunk_start = chunk * chunk_n
    for offset in tl.range(0, chunk_n, block_n, disable_licm=True):
        current_n = chunk_start + offset + offsets_n
        mask = (current_n[:, None] < key_length) & (offsets_d[None, :] < head_dim)
        key = tl.load(
            key_ptr
            + batch * stride_kb
            + head * stride_kh
            + current_n[:, None] * stride_kn
            + offsets_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        key_accumulator += tl.sum(key, axis=0)
        if not is_causal:
            value = tl.load(
                value_ptr
                + batch * stride_vb
                + head * stride_vh
                + current_n[:, None] * stride_vn
                + offsets_d[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            value_accumulator += tl.sum(value, axis=0)
    output_offsets = (batch_head * num_chunks + chunk) * head_dim + offsets_d
    tl.store(key_partial_ptr + output_offsets, key_accumulator, mask=offsets_d < head_dim)
    if not is_causal:
        tl.store(
            value_partial_ptr + output_offsets,
            value_accumulator,
            mask=offsets_d < head_dim,
        )


@triton.jit
def _kv_mean_finalize_kernel(
    key_partial_ptr,
    value_partial_ptr,
    key_mean_ptr,
    value_mean_ptr,
    key_length,
    num_chunks,
    is_causal: tl.constexpr,
    head_dim: tl.constexpr,
    block_chunks: tl.constexpr,
    block_d: tl.constexpr,
):
    """Merge K and optional non-causal V partials into compact FP32 means."""
    batch_head = tl.program_id(0)
    feature_block = tl.program_id(1)
    offsets_c = tl.arange(0, block_chunks)
    offsets_d = feature_block * block_d + tl.arange(0, block_d)
    mask = (offsets_c[:, None] < num_chunks) & (offsets_d[None, :] < head_dim)
    partial_offsets = (
        (batch_head * num_chunks + offsets_c[:, None]) * head_dim
        + offsets_d[None, :]
    )
    key_partials = tl.load(key_partial_ptr + partial_offsets, mask=mask, other=0.0)
    output_offsets = batch_head * head_dim + offsets_d
    output_mask = offsets_d < head_dim
    tl.store(
        key_mean_ptr + output_offsets,
        tl.sum(key_partials, axis=0) / key_length,
        mask=output_mask,
    )
    if not is_causal:
        value_partials = tl.load(
            value_partial_ptr + partial_offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            value_mean_ptr + output_offsets,
            tl.sum(value_partials, axis=0) / key_length,
            mask=output_mask,
        )


@triton.jit
def _quantize_value_per_key_kernel(
    value_ptr,
    value_mean_ptr,
    scale_multiplier_ptr,
    log_scale_ptr,
    correction_ptr,
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
    store_correction: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Quantize V with one symmetric signed-INT8 scale per key."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid = offsets_n < key_length
    batch_head = batch * heads + head
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    if not is_causal:
        value_mean = tl.load(value_mean_ptr + batch_head * head_dim + offsets_d)
        value = tl.where(valid[:, None], value - value_mean[None, :], 0.0)
    scale = tl.max(tl.abs(value), axis=1) / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(value / scale[:, None])
    tl.store(
        scale_multiplier_ptr + batch_head * key_length + offsets_n,
        scale * _P_UINT8_RANGE,
        mask=valid,
    )
    tl.store(
        log_scale_ptr + batch_head * key_length + offsets_n,
        tl.log2(scale),
        mask=valid,
    )
    if store_correction:
        correction_offsets = (
            (batch_head * tl.cdiv(key_length, block_n) + key_block) * head_dim
            + offsets_d
        )
        tl.store(
            correction_ptr + correction_offsets,
            tl.sum(quantized.to(tl.int32), axis=0),
        )
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_d[None, :] * stride_od
        + offsets_n[:, None] * stride_ok,
        quantized,
        mask=valid[:, None],
    )


@triton.jit
def _load_key_tile(
    key_ptr,
    batch_head,
    start_n,
    current_n,
    offsets_d,
    key_length,
    storage_key_length,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
):
    if use_tensor_descriptors:
        return key_ptr.load([batch_head, start_n, 0]).reshape((block_n, head_dim)).T
    else:
        return tl.load(
            key_ptr
            + (batch_head * storage_key_length + current_n[None, :]) * head_dim
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
    storage_key_length,
    feature_start: tl.constexpr,
    feature_block: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
):
    if use_tensor_descriptors:
        return value_ptr.load([batch_head, feature_start, start_n]).reshape(
            (feature_block, block_n)
        ).T
    else:
        return tl.load(
            value_ptr
            + (batch_head * head_dim + feature_start + offsets_d[None, :])
            * storage_key_length
            + current_n[:, None],
            mask=current_n[:, None] < key_length,
            other=0,
        )


@triton.jit
def _piper_attention_kernel(  # noqa: PLR0912, PLR0915
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    value_correction_ptr,
    value_mean_ptr,
    output_ptr,
    query_length,
    key_length,
    storage_key_length,
    query_block_offset: tl.constexpr,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    native_uint8: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    scaled_fp16_numerator: tl.constexpr,
    unmasked_query_tiles: tl.constexpr,
    unmasked_self_attention: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    reverse_causal_blocks: tl.constexpr,
    loop_num_stages: tl.constexpr,
    loop_licm: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
):
    """Fused exact-log UINT8-P/INT8-V online attention."""
    query_block = tl.program_id(0)
    if is_causal and reverse_causal_blocks:
        query_block = tl.num_programs(0) - 1 - query_block
    query_block += query_block_offset
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    if unmasked_query_tiles:
        valid_queries = tl.full((block_m,), True, dtype=tl.int1)
    else:
        valid_queries = offsets_m < query_length

    query = tl.load(
        query_ptr
        + ((batch_head * query_length + offsets_m[:, None]) * head_dim)
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr
            + batch_head * tl.cdiv(query_length, 32)
            + offsets_m // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + batch_head * query_length + offsets_m,
            mask=valid_queries,
            other=0.0,
        )

    if split_pv_head_dim:
        half_head_dim: tl.constexpr = head_dim // 2
        offsets_vd = tl.arange(0, half_head_dim)
        if scaled_fp16_numerator:
            accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
            accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
        else:
            accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
            accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    elif scaled_fp16_numerator:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float16)
    else:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, (query_block + 1) * block_m)

    for start_n in tl.range(
        0,
        end_n,
        block_n,
        num_stages=loop_num_stages,
        disable_licm=not loop_licm,
    ):
        current_n = start_n + offsets_n
        key = _load_key_tile(
            key_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            storage_key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores = tl.dot(query, key, out_dtype=tl.int32)
        if grouped_qk:
            key_scale = tl.load(
                key_scale_ptr
                + batch_head * tl.cdiv(key_length, block_n)
                + start_n // block_n
            )
            scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
        else:
            key_scale = tl.load(
                key_scale_ptr + batch_head * key_length + current_n,
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
            scores = tl.where(valid_queries[:, None] & valid_keys, scores, -float("inf"))

        value_log_scale = tl.load(
            value_log_scale_ptr + batch_head * key_length + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
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
            tl.exp2(scores - block_max[:, None]),
            0.0,
        )
        denominator = (
            denominator * old_weight
            + tl.sum(probabilities, axis=1) * current_weight
        )
        value_scale_multiplier = tl.load(
            value_scale_multiplier_ptr + batch_head * key_length + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
        probability_values = probabilities * value_scale_multiplier[None, :] + 0.5
        if native_uint8 and use_packed_probability_conversion:
            probability_uint8 = _ptx_float32_to_uint8x4(probability_values)
        else:
            probability_codes = tl.minimum(
                _P_UINT8_RANGE,
                probability_values,
            ).to(tl.int32)
            if native_uint8:
                probability_uint8 = probability_codes.to(tl.uint8)
            else:
                probability_int8 = (probability_codes - _P_ZERO_POINT).to(tl.int8)
        correction_block = batch_head * tl.cdiv(key_length, block_n) + start_n // block_n

        if split_pv_head_dim:
            value_low = _load_value_tile(
                value_ptr,
                batch_head,
                start_n,
                current_n,
                offsets_vd,
                key_length,
                storage_key_length,
                0,
                half_head_dim,
                head_dim,
                block_n,
                use_tensor_descriptors,
            )
            value_high = _load_value_tile(
                value_ptr,
                batch_head,
                start_n,
                current_n,
                offsets_vd,
                key_length,
                storage_key_length,
                half_head_dim,
                half_head_dim,
                head_dim,
                block_n,
                use_tensor_descriptors,
            )
            if native_uint8:
                partial_low = uint8_int8_dot(probability_uint8, value_low)
                partial_high = uint8_int8_dot(probability_uint8, value_high)
            else:
                correction_base = correction_block * head_dim
                correction_low = (
                    tl.load(value_correction_ptr + correction_base + offsets_vd).to(tl.int32)
                    << 7
                )
                correction_high = (
                    tl.load(
                        value_correction_ptr
                        + correction_base
                        + half_head_dim
                        + offsets_vd
                    ).to(tl.int32)
                    << 7
                )
                partial_low = tl.dot(
                    probability_int8,
                    value_low,
                    acc=tl.zeros((block_m, half_head_dim), dtype=tl.int32)
                    + correction_low[None, :],
                    out_dtype=tl.int32,
                )
                partial_high = tl.dot(
                    probability_int8,
                    value_high,
                    acc=tl.zeros((block_m, half_head_dim), dtype=tl.int32)
                    + correction_high[None, :],
                    out_dtype=tl.int32,
                )
            if scaled_fp16_numerator:
                partial_low_scaled = (partial_low.to(tl.float32) * (1.0 / 65536.0)).to(
                    tl.float16
                )
                partial_high_scaled = (partial_high.to(tl.float32) * (1.0 / 65536.0)).to(
                    tl.float16
                )
                accumulator_low = (
                    accumulator_low * old_weight[:, None].to(tl.float16)
                    + partial_low_scaled * current_weight[:, None].to(tl.float16)
                )
                accumulator_high = (
                    accumulator_high * old_weight[:, None].to(tl.float16)
                    + partial_high_scaled * current_weight[:, None].to(tl.float16)
                )
            else:
                accumulator_low = (
                    accumulator_low * old_weight[:, None]
                    + partial_low.to(tl.float32)
                    * (1.0 / _P_UINT8_RANGE)
                    * current_weight[:, None]
                )
                accumulator_high = (
                    accumulator_high * old_weight[:, None]
                    + partial_high.to(tl.float32)
                    * (1.0 / _P_UINT8_RANGE)
                    * current_weight[:, None]
                )
        else:
            value_tile = _load_value_tile(
                value_ptr,
                batch_head,
                start_n,
                current_n,
                offsets_d,
                key_length,
                storage_key_length,
                0,
                head_dim,
                head_dim,
                block_n,
                use_tensor_descriptors,
            )
            if native_uint8:
                partial = uint8_int8_dot(probability_uint8, value_tile)
            else:
                correction = (
                    tl.load(
                        value_correction_ptr
                        + correction_block * head_dim
                        + offsets_d
                    ).to(tl.int32)
                    << 7
                )
                partial = tl.dot(
                    probability_int8,
                    value_tile,
                    acc=tl.zeros((block_m, head_dim), dtype=tl.int32)
                    + correction[None, :],
                    out_dtype=tl.int32,
                )
            if scaled_fp16_numerator:
                partial_scaled = (partial.to(tl.float32) * (1.0 / 65536.0)).to(tl.float16)
                accumulator = (
                    accumulator * old_weight[:, None].to(tl.float16)
                    + partial_scaled * current_weight[:, None].to(tl.float16)
                )
            else:
                accumulator = (
                    accumulator * old_weight[:, None]
                    + partial.to(tl.float32)
                    * (1.0 / _P_UINT8_RANGE)
                    * current_weight[:, None]
                )
        running_max = next_max

    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    if split_pv_head_dim:
        if scaled_fp16_numerator:
            denominator_code_scale: tl.constexpr = _P_UINT8_RANGE / 65536.0
            output_low = accumulator_low.to(tl.float32) / (
                denominator_safe * denominator_code_scale
            )
            output_high = accumulator_high.to(tl.float32) / (
                denominator_safe * denominator_code_scale
            )
        else:
            output_low = accumulator_low / denominator_safe
            output_high = accumulator_high / denominator_safe
        if not is_causal:
            value_mean_base = value_mean_ptr + batch_head * head_dim
            output_low += tl.load(value_mean_base + offsets_vd)[None, :]
            output_high += tl.load(
                value_mean_base + half_head_dim + offsets_vd
            )[None, :]
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
    else:
        if scaled_fp16_numerator:
            denominator_code_scale: tl.constexpr = _P_UINT8_RANGE / 65536.0
            output = accumulator.to(tl.float32) / (
                denominator_safe * denominator_code_scale
            )
        else:
            output = accumulator / denominator_safe
        if not is_causal:
            output += tl.load(
                value_mean_ptr + batch_head * head_dim + offsets_d
            )[None, :]
        tl.store(
            output_ptr
            + (batch_head * query_length + offsets_m[:, None]) * head_dim
            + offsets_d[None, :],
            output,
            mask=valid_queries[:, None],
        )


def _compute_kv_means(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, key_length, head_dim = key.shape
    num_chunks = int(triton.cdiv(key_length, _MEAN_CHUNK_N))
    partial_shape = (batch, heads, num_chunks, head_dim)
    key_partial = torch.empty(partial_shape, device=key.device, dtype=torch.float32)
    value_partial = (
        torch.empty_like(key_partial)
        if not is_causal
        else torch.empty((1,), device=value.device, dtype=torch.float32)
    )
    key_mean = torch.empty((batch, heads, head_dim), device=key.device, dtype=torch.float32)
    value_mean = (
        torch.empty_like(key_mean)
        if not is_causal
        else torch.empty((1,), device=value.device, dtype=torch.float32)
    )
    _kv_mean_partial_kernel[
        (
            num_chunks,
            int(triton.cdiv(head_dim, _MEAN_BLOCK_D)),
            batch * heads,
        )
    ](
        key,
        value,
        key_partial,
        value_partial,
        key_length,
        num_chunks,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        is_causal=is_causal,
        heads=heads,
        head_dim=head_dim,
        chunk_n=_MEAN_CHUNK_N,
        block_n=_MEAN_BLOCK_N,
        block_d=_MEAN_BLOCK_D,
        num_warps=4,
    )
    _kv_mean_finalize_kernel[
        (batch * heads, int(triton.cdiv(head_dim, _MEAN_BLOCK_D)))
    ](
        key_partial,
        value_partial,
        key_mean,
        value_mean,
        key_length,
        num_chunks,
        is_causal=is_causal,
        head_dim=head_dim,
        block_chunks=triton.next_power_of_2(num_chunks),
        block_d=_MEAN_BLOCK_D,
        num_warps=4,
    )
    return key_mean, value_mean


def _prepare_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    key_mean: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    storage_key_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    query_int8 = torch.empty(query.shape, device=query.device, dtype=torch.int8)
    key_shape = (batch, heads, storage_key_length, head_dim)
    key_int8 = (
        torch.zeros(key_shape, device=key.device, dtype=torch.int8)
        if storage_key_length != key_length
        else torch.empty(key_shape, device=key.device, dtype=torch.int8)
    )
    if grouped_qk:
        query_groups = int(triton.cdiv(query_length, 32))
        key_groups = int(triton.cdiv(key_length, _BLOCK_N))
        query_scale = torch.empty(
            (batch, heads, query_groups),
            device=query.device,
            dtype=torch.float32,
        )
        key_scale = torch.empty(
            (batch, heads, key_groups),
            device=key.device,
            dtype=torch.float32,
        )
        qk_quantization.quantize_query_per_warp_kernel[(query_groups, heads, batch)](
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
            num_warps=4,
        )
        qk_quantization.quantize_key_per_block_kernel[(key_groups, heads, batch)](
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
            block_n=_BLOCK_N,
            num_warps=4,
        )
    else:
        query_scale = torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        key_scale = torch.empty(key.shape[:3], device=key.device, dtype=torch.float32)
        qk_quantization.quantize_query_per_thread_kernel[
            (triton.cdiv(query_length, 32) * 8, heads, batch)
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
            num_warps=4,
        )
        qk_quantization.quantize_key_per_thread_kernel[
            (triton.cdiv(key_length, _BLOCK_N) * 4, heads, batch)
        ](
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
            num_warps=4,
        )
    return query_int8, key_int8, query_scale, key_scale


def _make_descriptors(
    key: torch.Tensor,
    value: torch.Tensor,
    batch: int,
    heads: int,
    storage_key_length: int,
    head_dim: int,
    *,
    split_pv_head_dim: bool,
) -> tuple[TensorDescriptor, TensorDescriptor]:
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


def _default_piper_attention_execution_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool,
    *,
    target: AcceleratorTarget | None = None,
) -> _policy.PiperAttentionExecutionPlan:
    """Resolve production policy for preparation, benchmarks, and tuning."""
    batch, heads, query_length, head_dim = query.shape
    candidate_block_m = (
        select_query_block(query, batch, heads, query_length)
        if query.device.type == "cuda"
        else 64
    )
    target = AcceleratorTarget.from_device(query.device) if target is None else target
    return _policy.select_execution_plan(
        target,
        candidate_block_m=candidate_block_m,
        query_length=query_length,
        key_length=key.shape[2],
        head_dim=head_dim,
        is_causal=is_causal,
    )


@dataclass(frozen=True, slots=True)
class _PreparedPiperAttention:
    query: torch.Tensor
    key: torch.Tensor | TensorDescriptor
    value: torch.Tensor | TensorDescriptor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_log_scale: torch.Tensor
    value_correction: torch.Tensor
    value_mean: torch.Tensor
    output: torch.Tensor
    key_length: int
    storage_key_length: int
    is_causal: bool
    plan: _policy.PiperAttentionExecutionPlan


def _prepare_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    execution_plan: _policy.PiperAttentionExecutionPlan,
) -> _PreparedPiperAttention:
    """Quantize Q/K/V and construct the selected launch specialization."""
    batch, heads, _query_length, head_dim = query.shape
    key_length = key.shape[2]
    plan = execution_plan
    if plan.split_pv_head_dim and head_dim != 128:
        raise ValueError("split-PV Piper Attention requires head_dim=128")
    if plan.reverse_causal_blocks and not is_causal:
        raise ValueError("reverse block order requires causal attention")
    if plan.native_uint8:
        with torch.cuda.device(query.device):
            install_uint8_int8_dot_hook()
    padded_key_length = int(triton.cdiv(key_length, _BLOCK_N)) * _BLOCK_N
    storage_key_length = padded_key_length if plan.use_tensor_descriptors else key_length

    # A sequence-wide V mean is valid only for non-causal attention. Per-row
    # INT8 rounding would otherwise let future V rows perturb earlier outputs.
    key_mean, value_mean = _compute_kv_means(
        key,
        value,
        is_causal=is_causal,
    )
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        key_mean,
        scale,
        grouped_qk=plan.grouped_qk,
        storage_key_length=storage_key_length,
    )

    value_shape = (batch, heads, head_dim, storage_key_length)
    value_int8 = (
        torch.zeros(value_shape, device=value.device, dtype=torch.int8)
        if storage_key_length != key_length
        else torch.empty(value_shape, device=value.device, dtype=torch.int8)
    )
    value_scale_multiplier = torch.empty(
        (batch, heads, key_length),
        device=value.device,
        dtype=torch.float32,
    )
    value_log_scale = torch.empty(
        (batch, heads, key_length),
        device=value.device,
        dtype=torch.float16,
    )
    value_correction = (
        torch.empty(
            (batch, heads, int(triton.cdiv(key_length, _BLOCK_N)), head_dim),
            device=value.device,
            dtype=torch.int16,
        )
        if not plan.native_uint8
        else torch.empty((1,), device=value.device, dtype=torch.int16)
    )
    _quantize_value_per_key_kernel[
        (triton.cdiv(key_length, _BLOCK_N), heads, batch)
    ](
        value,
        value_mean,
        value_scale_multiplier,
        value_log_scale,
        value_correction,
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
        store_correction=not plan.native_uint8,
        heads=heads,
        head_dim=head_dim,
        block_n=_BLOCK_N,
        num_warps=4,
    )

    key_argument: torch.Tensor | TensorDescriptor = key_int8
    value_argument: torch.Tensor | TensorDescriptor = value_int8
    if plan.use_tensor_descriptors:
        key_argument, value_argument = _make_descriptors(
            key_int8,
            value_int8,
            batch,
            heads,
            storage_key_length,
            head_dim,
            split_pv_head_dim=plan.split_pv_head_dim,
        )
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    return _PreparedPiperAttention(
        query=query_int8,
        key=key_argument,
        value=value_argument,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_log_scale=value_log_scale,
        value_correction=value_correction,
        value_mean=value_mean,
        output=output,
        key_length=key_length,
        storage_key_length=storage_key_length,
        is_causal=is_causal,
        plan=plan,
    )


def _launch_piper_attention(prepared: _PreparedPiperAttention) -> torch.Tensor:
    """Launch only the fused attention recurrence on prepared integer inputs."""
    batch, heads, query_length, head_dim = prepared.output.shape
    plan = prepared.plan
    attention_kernel = cast(Any, _piper_attention_kernel)
    launch_options = {
        "num_warps": plan.num_warps,
        "num_stages": plan.num_stages,
    }

    def launch(query_blocks: int, query_block_offset: int, unmasked_queries: bool) -> None:
        attention_kernel[(query_blocks, heads, batch)](
            prepared.query,
            prepared.key,
            prepared.value,
            prepared.query_scale,
            prepared.key_scale,
            prepared.value_scale_multiplier,
            prepared.value_log_scale,
            prepared.value_correction,
            prepared.value_mean,
            prepared.output,
            query_length,
            prepared.key_length,
            prepared.storage_key_length,
            query_block_offset=query_block_offset,
            is_causal=prepared.is_causal,
            grouped_qk=plan.grouped_qk,
            native_uint8=plan.native_uint8,
            split_pv_head_dim=plan.split_pv_head_dim,
            scaled_fp16_numerator=plan.scaled_fp16_numerator,
            unmasked_query_tiles=unmasked_queries,
            unmasked_self_attention=(
                unmasked_queries
                and not prepared.is_causal
                and query_length == prepared.key_length
                and prepared.key_length % _BLOCK_N == 0
            ),
            heads=heads,
            head_dim=head_dim,
            block_m=plan.block_m,
            block_n=_BLOCK_N,
            use_tensor_descriptors=plan.use_tensor_descriptors,
            reverse_causal_blocks=plan.reverse_causal_blocks,
            loop_num_stages=plan.loop_num_stages,
            loop_licm=plan.loop_licm,
            use_packed_probability_conversion=plan.use_packed_probability_conversion,
            **launch_options,
        )

    full_query_blocks = query_length // plan.block_m
    has_partial_query_block = query_length % plan.block_m != 0
    if plan.reverse_causal_blocks and has_partial_query_block:
        launch(1, full_query_blocks, False)
    if full_query_blocks:
        launch(full_query_blocks, 0, True)
    if not plan.reverse_causal_blocks and has_partial_query_block:
        launch(1, full_query_blocks, False)
    return prepared.output


def _run_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    execution_plan: _policy.PiperAttentionExecutionPlan | None = None,
) -> torch.Tensor:
    """Run Piper Attention preprocessing and its fused recurrence."""
    plan = (
        execution_plan
        if execution_plan is not None
        else _default_piper_attention_execution_plan(
            query,
            key,
            is_causal,
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


@torch.library.custom_op("piper_kernels::piper_attention", mutates_args=())
def triton_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Run Piper Attention preprocessing and its fused integer-PV kernel."""
    return _run_piper_attention(
        query,
        key,
        value,
        scale,
        is_causal,
    )


@triton_piper_attention.register_fake
def _triton_piper_attention_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
