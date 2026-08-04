"""Pure-Triton prototype for ConvRot integer attention on consumer GPUs."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels.attention._convrot_triton import (
    hadamard_stage_rows,
    rotate_attention_rows,
)

_LOG2_E = math.log2(math.e)
_SCALE_BITS = 20
_SCALE_ONE = 1 << _SCALE_BITS


@triton.jit
def _rotate_quantize_query_kernel(
    query_ptr,
    key_scale_ptr,
    output_ptr,
    score_scale_ptr,
    query_length,
    score_factor,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotation_group: tl.constexpr,
    block_m: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_m[:, None] < query_length
    pointers = (batch_head * query_length + offsets_m[:, None]) * head_dim + offsets_d[None, :]
    values = tl.load(query_ptr + pointers, mask=mask, other=0.0).to(tl.float32)
    values = hadamard_stage_rows(values, offsets_d, 1, block_m)
    values = hadamard_stage_rows(values, offsets_d, 4, block_m)
    if rotation_group >= 64:
        values = hadamard_stage_rows(values, offsets_d, 16, block_m)
    values *= rotation_group**-0.5
    values *= tl.load(key_scale_ptr + batch_head * head_dim + offsets_d)[None, :]

    scale = tl.maximum(tl.max(tl.abs(values), axis=1) / 127.0, 1e-30)
    normalized = values / scale[:, None]
    rounded = normalized + 0.5 * tl.where(normalized >= 0, 1.0, -1.0)
    quantized = tl.maximum(-128.0, tl.minimum(127.0, rounded)).to(tl.int8)
    tl.store(output_ptr + pointers, quantized, mask=mask)
    multiplier = libdevice.rint(scale * score_factor)
    multiplier = tl.maximum(1.0, tl.minimum(2147483647.0, multiplier))
    tl.store(
        score_scale_ptr + batch_head * query_length + offsets_m,
        multiplier.to(tl.int32),
        mask=offsets_m < query_length,
    )


def _rotate_quantize_query_rows(
    query: torch.Tensor,
    key_scale: torch.Tensor,
    attention_scale: float,
    rotation_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    contiguous = query.contiguous()
    output = torch.empty_like(contiguous, dtype=torch.int8)
    score_scale = torch.empty(contiguous.shape[:-1], device=query.device, dtype=torch.int32)
    block_m = 16
    _rotate_quantize_query_kernel[
        (triton.cdiv(contiguous.shape[2], block_m), contiguous.shape[1], contiguous.shape[0])
    ](
        contiguous,
        key_scale,
        output,
        score_scale,
        query.shape[2],
        attention_scale * _LOG2_E * _SCALE_ONE,
        heads=query.shape[1],
        head_dim=contiguous.shape[-1],
        rotation_group=rotation_group,
        block_m=block_m,
        num_warps=4,
    )
    return output, score_scale


@triton.jit
def _kv_statistics_partial_kernel(
    key_ptr,
    value_ptr,
    key_minimum_ptr,
    key_maximum_ptr,
    value_maximum_ptr,
    key_length,
    num_partials,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    partial = tl.program_id(0)
    batch_head = tl.program_id(1)
    offsets_n = partial * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length
    pointers = (batch_head * key_length + offsets_n[:, None]) * head_dim + offsets_d[None, :]
    key = tl.load(key_ptr + pointers, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + pointers, mask=mask, other=0.0).to(tl.float32)
    output_offsets = (batch_head * num_partials + partial) * head_dim + offsets_d
    tl.store(
        key_minimum_ptr + output_offsets,
        tl.min(tl.where(mask, key, float("inf")), axis=0),
    )
    tl.store(
        key_maximum_ptr + output_offsets,
        tl.max(tl.where(mask, key, -float("inf")), axis=0),
    )
    tl.store(value_maximum_ptr + output_offsets, tl.max(tl.abs(value), axis=0))


@triton.jit
def _finish_kv_statistics_kernel(
    key_minimum_ptr,
    key_maximum_ptr,
    value_maximum_ptr,
    key_midpoint_ptr,
    key_scale_ptr,
    value_scale_ptr,
    num_partials,
    head_dim: tl.constexpr,
    partial_block: tl.constexpr,
    block_d: tl.constexpr,
):
    block_d_id = tl.program_id(0)
    batch_head = tl.program_id(1)
    offsets_p = tl.arange(0, partial_block)
    offsets_d = block_d_id * block_d + tl.arange(0, block_d)
    mask = (offsets_p[:, None] < num_partials) & (offsets_d[None, :] < head_dim)
    pointers = (batch_head * num_partials + offsets_p[:, None]) * head_dim + offsets_d[None, :]
    minimum = tl.min(
        tl.load(key_minimum_ptr + pointers, mask=mask, other=float("inf")),
        axis=0,
    )
    maximum = tl.max(
        tl.load(key_maximum_ptr + pointers, mask=mask, other=-float("inf")),
        axis=0,
    )
    value_maximum = tl.max(
        tl.load(value_maximum_ptr + pointers, mask=mask, other=0.0),
        axis=0,
    )
    output_offsets = batch_head * head_dim + offsets_d
    output_mask = offsets_d < head_dim
    tl.store(key_midpoint_ptr + output_offsets, (minimum + maximum) * 0.5, mask=output_mask)
    tl.store(
        key_scale_ptr + output_offsets,
        tl.maximum((maximum - minimum) * (1.0 / 254.0), 1e-30),
        mask=output_mask,
    )
    tl.store(
        value_scale_ptr + output_offsets,
        tl.maximum(value_maximum * (1.0 / 127.0), 1e-30),
        mask=output_mask,
    )


@triton.jit
def _quantize_kv_kernel(
    key_ptr,
    value_ptr,
    key_midpoint_ptr,
    key_scale_ptr,
    value_scale_ptr,
    key_output_ptr,
    value_output_ptr,
    value_sum_ptr,
    key_length,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length
    pointers = (batch_head * key_length + offsets_n[:, None]) * head_dim + offsets_d[None, :]
    statistic_offsets = batch_head * head_dim + offsets_d

    key = tl.load(key_ptr + pointers, mask=mask, other=0.0).to(tl.float32)
    midpoint = tl.load(key_midpoint_ptr + statistic_offsets)
    key_scale = tl.load(key_scale_ptr + statistic_offsets)
    key_quantized = libdevice.rint((key - midpoint[None, :]) / key_scale[None, :])
    key_quantized = tl.maximum(-128.0, tl.minimum(127.0, key_quantized)).to(tl.int8)
    tl.store(key_output_ptr + pointers, key_quantized, mask=mask)

    value = tl.load(value_ptr + pointers, mask=mask, other=0.0).to(tl.float32)
    value_scale = tl.load(value_scale_ptr + statistic_offsets)
    value_quantized = libdevice.rint(value / value_scale[None, :])
    value_quantized = tl.maximum(-128.0, tl.minimum(127.0, value_quantized)).to(tl.int8)
    tl.store(value_output_ptr + pointers, value_quantized, mask=mask)
    tl.store(
        value_sum_ptr
        + (batch_head * tl.cdiv(key_length, block_n) + key_block) * head_dim
        + offsets_d,
        tl.sum(tl.where(mask, value_quantized.to(tl.int32), 0), axis=0),
    )


def _quantize_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    contiguous_value = value.contiguous()
    batch, heads, key_length, head_dim = key.shape
    batch_heads = batch * heads
    statistics_block = 256
    num_partials = (key_length + statistics_block - 1) // statistics_block
    partial_shape = (batch, heads, num_partials, head_dim)
    key_minimum_partial = torch.empty(partial_shape, device=key.device, dtype=torch.float32)
    key_maximum_partial = torch.empty_like(key_minimum_partial)
    value_maximum_partial = torch.empty_like(key_minimum_partial)
    _kv_statistics_partial_kernel[(num_partials, batch_heads)](
        key,
        contiguous_value,
        key_minimum_partial,
        key_maximum_partial,
        value_maximum_partial,
        key_length,
        num_partials,
        head_dim=head_dim,
        block_n=statistics_block,
        num_warps=4,
    )

    statistic_shape = (batch, heads, head_dim)
    key_midpoint = torch.empty(statistic_shape, device=key.device, dtype=torch.float32)
    key_scale = torch.empty_like(key_midpoint)
    value_scale = torch.empty_like(key_midpoint)
    partial_block = triton.next_power_of_2(num_partials)
    _finish_kv_statistics_kernel[(triton.cdiv(head_dim, 32), batch_heads)](
        key_minimum_partial,
        key_maximum_partial,
        value_maximum_partial,
        key_midpoint,
        key_scale,
        value_scale,
        num_partials,
        head_dim=head_dim,
        partial_block=partial_block,
        block_d=32,
        num_warps=4,
    )

    key_output = torch.empty_like(key, dtype=torch.int8)
    value_output = torch.empty_like(contiguous_value, dtype=torch.int8)
    num_key_blocks = (key_length + block_n - 1) // block_n
    value_sum = torch.empty(
        (batch, heads, num_key_blocks, head_dim),
        device=value.device,
        dtype=torch.int32,
    )
    _quantize_kv_kernel[(num_key_blocks, heads, batch)](
        key,
        contiguous_value,
        key_midpoint,
        key_scale,
        value_scale,
        key_output,
        value_output,
        value_sum,
        key_length,
        heads=heads,
        head_dim=head_dim,
        block_n=block_n,
        num_warps=4,
    )
    return key_output, key_scale, value_output, value_scale, value_sum


@triton.jit
def _multiply_q8(value, weight):
    """Multiply INT32 by unsigned Q8 without an INT64 intermediate."""
    high = value >> 8
    low = value - (high << 8)
    return high * weight + ((low * weight + (1 << 7)) >> 8)


@triton.jit
def _multiply_q15(value, weight):
    """Multiply INT32 by unsigned Q15 without an INT64 intermediate."""
    high = value >> 15
    low = value - (high << 15)
    return high * weight + ((low * weight + (1 << 14)) >> 15)


@triton.jit
def _integer_exp2(
    negative_delta,
    scale_multiplier,
    output_maximum: tl.constexpr,
    polynomial_degree: tl.constexpr,
):
    """Fixed-point approximation of ``output_maximum * 2**x``."""
    if polynomial_degree == 0:
        exponent = (
            negative_delta.to(tl.float32) * scale_multiplier.to(tl.float32) * (1.0 / (1 << 20))
        )
        return (tl.exp2(exponent) * output_maximum + 0.5).to(tl.int32)

    recurrence_one = 1 << 15
    exp_a = -22713
    maximum_whole = 8 if output_maximum == 255 else 15

    magnitude = tl.maximum(-negative_delta * scale_multiplier, 0)
    original_whole = magnitude >> 20
    whole = tl.minimum(original_whole, maximum_whole)
    fraction = (magnitude & ((1 << 20) - 1)) >> 5

    if polynomial_degree == 1:
        polynomial = recurrence_one - (fraction >> 1)
    elif polynomial_degree == 2:
        polynomial = 6329
        polynomial = exp_a + ((polynomial * fraction + (recurrence_one // 2)) >> 15)
        polynomial = recurrence_one + ((polynomial * fraction + (recurrence_one // 2)) >> 15)
    else:
        polynomial = -1302
        polynomial = 7631 + ((polynomial * fraction + (recurrence_one // 2)) >> 15)
        polynomial = exp_a + ((polynomial * fraction + (recurrence_one // 2)) >> 15)
        polynomial = recurrence_one + ((polynomial * fraction + (recurrence_one // 2)) >> 15)

    shift = whole + 15
    rounding = 1 << (shift - 1)
    result = (polynomial * output_maximum + rounding) >> shift
    return tl.where(original_whole > maximum_whole, 0, result)


@triton.jit
def _int8_convrot_attention_kernel(  # noqa: PLR0912, PLR0915
    query_ptr,
    key_ptr,
    value_ptr,
    value_sum_ptr,
    score_scale_ptr,
    value_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    is_causal: tl.constexpr,
    probability_degree: tl.constexpr,
    probability_maximum: tl.constexpr,
    recurrence_bits: tl.constexpr,
    integer_recurrence: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    batch_head = batch * heads + head

    query = tl.load(
        query_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=offsets_m[:, None] < query_length,
        other=0,
    )
    scale_multiplier = tl.load(
        score_scale_ptr + batch_head * query_length + offsets_m,
        mask=offsets_m < query_length,
        other=1,
    )
    if integer_recurrence:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.int32)
        denominator = tl.zeros((block_m,), dtype=tl.int32)
    else:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
        denominator = tl.zeros((block_m,), dtype=tl.float32)
    negative_infinity: tl.constexpr = -(1 << 30)
    running_max = tl.full((block_m,), negative_infinity, dtype=tl.int32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, (query_block + 1) * block_m)

    for start_n in tl.range(0, end_n, block_n, disable_licm=True):
        current_n = start_n + offsets_n
        key = tl.load(
            key_ptr
            + (batch_head * key_length + current_n[None, :]) * head_dim
            + offsets_d[:, None],
            mask=current_n[None, :] < key_length,
            other=0,
        )
        integer_scores = tl.dot(query, key, out_dtype=tl.int32)
        valid_keys = current_n[None, :] < key_length
        if is_causal:
            valid_keys &= current_n[None, :] <= offsets_m[:, None]
        integer_scores = tl.where(valid_keys, integer_scores, negative_infinity)

        next_max = tl.maximum(running_max, tl.max(integer_scores, axis=1))
        score_delta = tl.where(
            valid_keys,
            integer_scores - next_max[:, None],
            -(1 << 22),
        )
        probabilities = _integer_exp2(
            score_delta,
            scale_multiplier[:, None],
            probability_maximum,
            probability_degree,
        )
        probabilities = tl.where(valid_keys, probabilities, 0)
        # Bound the first iteration's sentinel subtraction before multiplying
        # by the Q20 score scale.  Its result is explicitly replaced with zero.
        old_delta = tl.where(
            running_max == negative_infinity,
            -(1 << 22),
            running_max - next_max,
        )
        old_weight = _integer_exp2(
            old_delta,
            scale_multiplier,
            (1 << recurrence_bits) - 1,
            probability_degree,
        )
        old_weight = tl.where(running_max == negative_infinity, 0, old_weight)

        value = tl.load(
            value_ptr
            + (batch_head * key_length + current_n[:, None]) * head_dim
            + offsets_d[None, :],
            mask=current_n[:, None] < key_length,
            other=0,
        )
        value_sum = tl.load(
            value_sum_ptr
            + ((batch_head * tl.cdiv(key_length, block_n) + start_n // block_n) * head_dim)
            + offsets_d
        )
        if integer_recurrence:
            if recurrence_bits == 8:
                denominator = _multiply_q8(denominator, old_weight)
                accumulator = _multiply_q8(accumulator, old_weight[:, None])
            else:
                denominator = _multiply_q15(denominator, old_weight)
                accumulator = _multiply_q15(accumulator, old_weight[:, None])
            denominator += tl.sum(probabilities, axis=1)
            if probability_maximum == 255:
                accumulator = tl.dot(
                    (probabilities - 128).to(tl.int8),
                    value,
                    acc=accumulator,
                    out_dtype=tl.int32,
                )
                accumulator += value_sum[None, :] * 128
            else:
                accumulator = tl.dot(
                    probabilities.to(tl.int8),
                    value,
                    acc=accumulator,
                    out_dtype=tl.int32,
                )
        else:
            old_weight_float = old_weight.to(tl.float32) * (1.0 / (1 << recurrence_bits))
            denominator *= old_weight_float
            denominator += tl.sum(probabilities, axis=1).to(tl.float32)
            if probability_maximum == 255:
                partial = tl.dot(
                    (probabilities - 128).to(tl.int8),
                    value,
                    out_dtype=tl.int32,
                )
                partial += value_sum[None, :] * 128
            else:
                partial = tl.dot(
                    probabilities.to(tl.int8),
                    value,
                    out_dtype=tl.int32,
                )
            accumulator *= old_weight_float[:, None]
            accumulator += partial.to(tl.float32)
        running_max = next_max

    output = accumulator.to(tl.float32) / denominator[:, None]
    value_scale = tl.load(value_scale_ptr + batch_head * head_dim + offsets_d)
    output *= value_scale[None, :]
    tl.store(
        output_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        output,
        mask=offsets_m[:, None] < query_length,
    )


def triton_int8_convrot_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int = 64,
) -> torch.Tensor:
    """Run the correctness-first Triton prototype with fused preprocessing."""
    key_length = key.shape[2]
    block_n = 128 if key_length <= 128 else 64
    key_rotated = rotate_attention_rows(key, rotation_group)
    key_int8, key_scale, value_int8, value_scale, value_sum = _quantize_key_value(
        key_rotated,
        value,
        block_n,
    )
    query_int8, score_scale = _rotate_quantize_query_rows(
        query,
        key_scale,
        float(scale),
        rotation_group,
    )
    batch, heads, query_length, head_dim = query.shape
    output = torch.empty_like(query)
    block_m = 64 if block_n == 128 else 32
    grid = (triton.cdiv(query_length, block_m), heads, batch)
    _int8_convrot_attention_kernel[grid](
        query_int8,
        key_int8,
        value_int8,
        value_sum,
        score_scale,
        value_scale,
        output,
        query_length,
        key_length,
        is_causal=is_causal,
        probability_degree=2,
        probability_maximum=255,
        recurrence_bits=8 if block_n == 128 else 15,
        integer_recurrence=True,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        num_stages=2,
        num_warps=4,
    )
    return output
