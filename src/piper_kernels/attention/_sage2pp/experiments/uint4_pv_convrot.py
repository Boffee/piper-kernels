"""UINT4-P plus INT4-range ConvRot-V Sage2++ quality experiment.

The four-bit codes are intentionally stored in INT8 tensors and consumed by
an INT8 ``tl.dot``.  This isolates numerical quality and kernel orchestration;
it does not claim native packed-INT4/UINT4 throughput.
"""

# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from typing import Literal

import torch
import triton
import triton.language as tl

from piper_kernels.attention._convrot_reference import rotate_attention_groups
from piper_kernels.attention._convrot_triton import rotate_rows_in_registers
from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.reference import (
    _quantize_key_per_thread,
    _quantize_per_group,
    _quantize_query_per_thread,
)

_SCALE_EPSILON = tl.constexpr(1e-7)
_P_RANGE_TL = tl.constexpr(15.0)
_V_RANGE_TL = tl.constexpr(7.0)
_QK_RANGE = 127
_P_RANGE = 15
_V_RANGE = 7
_Q_BLOCK = 32
_K_BLOCK = 64
_PV_BLOCK = 64


@triton.jit
def _round_to_uint4_in_int8(values):
    rounded = values + 0.5
    return tl.maximum(0.0, tl.minimum(15.0, rounded)).to(tl.int8)


@triton.jit
def _key_sum_partial_kernel(
    key_ptr,
    output_ptr,
    key_length,
    num_partials,
    stride_kb,
    stride_kh,
    stride_kn,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    partial = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_n = partial * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length
    key = tl.load(
        key_ptr
        + batch * stride_kb
        + head * stride_kh
        + offsets_n[:, None] * stride_kn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output_offsets = (batch_head * num_partials + partial) * head_dim + offsets_d
    tl.store(output_ptr + output_offsets, tl.sum(key, axis=0))


@triton.jit
def _finish_key_mean_kernel(
    partial_ptr,
    mean_ptr,
    key_length,
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
    partials = tl.load(
        partial_ptr
        + (batch_head * num_partials + offsets_p[:, None]) * head_dim
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(
        mean_ptr + batch_head * head_dim + offsets_d,
        tl.sum(partials, axis=0) / key_length,
        mask=offsets_d < head_dim,
    )


@triton.jit
def _quantize_value_convrot_kernel(
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
    rotation_group: tl.constexpr,
    paired_rotation: tl.constexpr,
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
    if paired_rotation:
        value_transposed = tl.trans(value)
        value_transposed = rotate_rows_in_registers(
            value_transposed, offsets_n, head_dim, rotation_group
        )
        value = tl.trans(value_transposed)
    else:
        value = rotate_rows_in_registers(value, offsets_d, block_n, rotation_group)
    scale = tl.max(tl.abs(value), axis=0) / _V_RANGE_TL + _SCALE_EPSILON
    quantized = _sage_backend._round_to_int8(value / scale[None, :], _V_RANGE_TL)
    scale_block = (batch * heads + head) * tl.cdiv(key_length, block_n) + key_block
    tl.store(scale_ptr + scale_block * head_dim + offsets_d, scale)
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_n[:, None] * stride_on
        + offsets_d[None, :],
        quantized,
        mask=mask,
    )


@triton.jit
def _uint4_pv_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    rotated_output_ptr,
    query_length,
    key_length,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    rotation_group: tl.constexpr,
    paired_rotation: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
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
    accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, (query_block + 1) * block_m)

    for start_n in tl.range(0, end_n, block_n, disable_licm=True):
        current_n = start_n + offsets_n
        key = tl.load(
            key_ptr
            + ((batch * heads + head) * key_length + current_n[None, :]) * head_dim
            + offsets_d[:, None],
            mask=current_n[None, :] < key_length,
            other=0,
        )
        integer_scores = tl.dot(query, key)
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
        valid_keys = current_n[None, :] < key_length
        if is_causal:
            valid_keys &= current_n[None, :] <= offsets_m[:, None]
        scores = tl.where(valid_queries[:, None] & valid_keys, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_weight = tl.where(valid_queries, tl.exp2(running_max - next_max), 0.0)
        probabilities = tl.where(
            valid_queries[:, None] & valid_keys,
            tl.exp2(scores - next_max[:, None]),
            0.0,
        )
        accumulator *= old_weight[:, None]
        denominator = denominator * old_weight + tl.sum(probabilities, axis=1)

        if paired_rotation:
            probability_for_dot = rotate_rows_in_registers(
                probabilities, offsets_n, block_m, rotation_group
            )
            probability_min = tl.min(probability_for_dot, axis=1)
            probability_max = tl.max(probability_for_dot, axis=1)
            probability_scale = tl.where(
                probability_max > probability_min,
                (probability_max - probability_min) / _P_RANGE_TL,
                1.0,
            )
            probability_zero = tl.maximum(
                0.0,
                tl.minimum(
                    _P_RANGE_TL,
                    -probability_min / probability_scale + 0.5,
                ),
            ).to(tl.int32)
            probability_uint4 = _round_to_uint4_in_int8(
                probability_for_dot / probability_scale[:, None] + probability_zero[:, None]
            )
        else:
            probability_max = tl.where(
                valid_queries,
                tl.exp2(block_max - next_max),
                0.0,
            )
            probability_scale = tl.where(probability_max > 0, probability_max / _P_RANGE_TL, 1.0)
            probability_zero = tl.zeros((block_m,), dtype=tl.int32)
            probability_uint4 = _round_to_uint4_in_int8(probabilities / probability_scale[:, None])
        value = tl.load(
            value_ptr
            + ((batch * heads + head) * key_length + current_n[:, None]) * head_dim
            + offsets_d[None, :],
            mask=current_n[:, None] < key_length,
            other=0,
        )
        value_scale_block = (batch * heads + head) * tl.cdiv(
            key_length, block_n
        ) + start_n // block_n
        value_scale = tl.load(value_scale_ptr + value_scale_block * head_dim + offsets_d)
        partial_int32 = tl.dot(probability_uint4, value)
        if paired_rotation:
            value_sum = tl.sum(value, axis=0).to(tl.int32)
            partial_int32 -= probability_zero[:, None] * value_sum[None, :]
        accumulator += (
            partial_int32.to(tl.float32) * probability_scale[:, None] * value_scale[None, :]
        )
        running_max = next_max

    rotated_output = accumulator / tl.maximum(denominator, 1e-30)[:, None]
    tl.store(
        rotated_output_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        rotated_output,
        mask=valid_queries[:, None],
    )


@triton.jit
def _inverse_rotate_output_kernel(
    input_ptr,
    output_ptr,
    head_dim: tl.constexpr,
    rotation_group: tl.constexpr,
):
    row = tl.program_id(0)
    offsets_d = tl.arange(0, head_dim)
    values = tl.load(input_ptr + row * head_dim + offsets_d)
    values = rotate_rows_in_registers(values[None, :], offsets_d, 1, rotation_group)
    tl.store(output_ptr + row * head_dim + offsets_d, values.reshape((head_dim,)))


def _prepare_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    grouped_qk: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    statistics_block = 256
    num_partials = (key_length + statistics_block - 1) // statistics_block
    partial_shape = (batch, heads, num_partials, head_dim)
    key_sum_partial = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    _key_sum_partial_kernel[(num_partials, batch * heads)](
        key,
        key_sum_partial,
        key_length,
        num_partials,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=statistics_block,
        num_warps=4,
    )
    key_mean = torch.empty((batch, heads, head_dim), device=query.device, dtype=torch.float32)
    partial_block = triton.next_power_of_2(num_partials)
    _finish_key_mean_kernel[(triton.cdiv(head_dim, 32), batch * heads)](
        key_sum_partial,
        key_mean,
        key_length,
        num_partials,
        head_dim=head_dim,
        partial_block=partial_block,
        block_d=32,
        num_warps=4,
    )

    query_int8 = torch.empty(query.shape, device=query.device, dtype=torch.int8)
    key_int8 = torch.empty(key.shape, device=query.device, dtype=torch.int8)
    if grouped_qk:
        query_scale_groups = (query_length + _Q_BLOCK - 1) // _Q_BLOCK
        key_scale_groups = (key_length + _K_BLOCK - 1) // _K_BLOCK
        query_scale = torch.empty(
            (batch, heads, query_scale_groups), device=query.device, dtype=torch.float32
        )
        key_scale = torch.empty(
            (batch, heads, key_scale_groups), device=query.device, dtype=torch.float32
        )
        _sage_backend._quantize_query_per_warp_kernel[(query_scale_groups, heads, batch)](
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
            quantization_range=_QK_RANGE,
            rotation_group=0,
            num_warps=4,
        )
        _sage_backend._quantize_key_per_block_kernel[(key_scale_groups, heads, batch)](
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
            block_n=_K_BLOCK,
            quantization_range=_QK_RANGE,
            rotation_group=0,
            num_warps=4,
        )
    else:
        query_scale = torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        key_scale = torch.empty(key.shape[:3], device=query.device, dtype=torch.float32)
        query_grid = (triton.cdiv(query_length, _Q_BLOCK) * 8, heads, batch)
        _sage_backend._quantize_query_kernel[query_grid](
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
            quantization_range=_QK_RANGE,
            rotation_group=0,
            num_warps=4,
        )
        key_grid = (triton.cdiv(key_length, _K_BLOCK) * 4, heads, batch)
        _sage_backend._quantize_key_kernel[key_grid](
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
            quantization_range=_QK_RANGE,
            rotation_group=0,
            num_warps=4,
        )
    return query_int8, key_int8, query_scale, key_scale


def _prepare_value(
    value: torch.Tensor,
    rotation_group: int,
    paired_rotation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, key_length, head_dim = value.shape
    block_n = _PV_BLOCK
    num_partials = (key_length + block_n - 1) // block_n
    value_scale = torch.empty(
        (batch, heads, num_partials, head_dim), device=value.device, dtype=torch.float32
    )
    value_int4 = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    _quantize_value_convrot_kernel[(num_partials, heads, batch)](
        value,
        value_scale,
        value_int4,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_int4.stride(0),
        value_int4.stride(1),
        value_int4.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=block_n,
        rotation_group=rotation_group,
        paired_rotation=paired_rotation,
        num_warps=4,
    )
    return value_int4, value_scale


def _run_uint4_pv_convrot(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int,
    grouped_qk: bool,
    paired_rotation: bool,
) -> torch.Tensor:
    if rotation_group not in (0, 16, 64):
        raise ValueError(f"rotation group must be 0, 16, or 64, got {rotation_group}")
    batch, heads, query_length, head_dim = query.shape
    if rotation_group and head_dim % rotation_group:
        raise ValueError(
            f"head dimension {head_dim} must be divisible by rotation group {rotation_group}"
        )
    key_length = key.shape[2]
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(query, key, scale, grouped_qk)
    value_int4, value_scale = _prepare_value(value, rotation_group, paired_rotation)

    rotated_output = torch.empty(query.shape, device=query.device, dtype=torch.float32)
    block_m = (
        64 if is_causal else _sage_backend._select_query_block(query, batch, heads, query_length)
    )
    attention_grid = (triton.cdiv(query_length, block_m), heads, batch)
    _uint4_pv_attention_kernel[attention_grid](
        query_int8,
        key_int8,
        value_int4,
        query_scale,
        key_scale,
        value_scale,
        rotated_output,
        query_length,
        key_length,
        is_causal=is_causal,
        grouped_qk=grouped_qk,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=_PV_BLOCK,
        rotation_group=rotation_group,
        paired_rotation=paired_rotation,
        num_stages=3,
        num_warps=4,
    )
    output = torch.empty_like(query)
    rows = batch * heads * query_length
    _inverse_rotate_output_kernel[(rows,)](
        rotated_output,
        output,
        head_dim=head_dim,
        rotation_group=0 if paired_rotation else rotation_group,
        num_warps=4,
    )
    return output


def triton_sage_attention_uint4_pv_convrot(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int = 64,
    grouped_qk: bool = False,
) -> torch.Tensor:
    """Run INT8 QK, UINT4 P, and feature-rotated INT4-range V.

    UINT4 and INT4 codes use unpacked INT8 storage and INT8 MMA. The inverse
    output rotation is separate to keep the attention accumulator layout small.
    """
    return _run_uint4_pv_convrot(
        query,
        key,
        value,
        scale,
        is_causal,
        rotation_group=rotation_group,
        grouped_qk=grouped_qk,
        paired_rotation=False,
    )


def triton_sage_attention_uint4_pv_paired_convrot(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int = 64,
    grouped_qk: bool = False,
) -> torch.Tensor:
    """Rotate each P/V contraction tile and encode signed P through affine UINT4."""
    return _run_uint4_pv_convrot(
        query,
        key,
        value,
        scale,
        is_causal,
        rotation_group=rotation_group,
        grouped_qk=grouped_qk,
        paired_rotation=True,
    )


def _quantize_probability_uint4(
    probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = probability.amax(dim=-1)
    scale = torch.where(maximum > 0, maximum / _P_RANGE, torch.ones_like(maximum))
    quantized = (probability / scale[..., None]).round().clamp(0, _P_RANGE).to(torch.int8)
    return quantized, scale


def reference_sage_attention_uint4_pv_convrot(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int = 64,
    qk_quantization: Literal["per_thread", "per_warp"] = "per_thread",
    paired_rotation: bool = False,
) -> torch.Tensor:
    """Evaluate the UINT4-P/ConvRot-INT4-V algorithm with PyTorch operations."""
    output_dtype = query.dtype
    key_centered = key.float() - key.float().mean(dim=2, keepdim=True)
    if qk_quantization == "per_warp":
        query_int8, query_scale = _quantize_per_group(query, _Q_BLOCK, _QK_RANGE)
        key_int8, key_scale = _quantize_per_group(key_centered, _K_BLOCK, _QK_RANGE)
        query_scale = query_scale.repeat_interleave(_Q_BLOCK, dim=2)[:, :, : query.shape[2]]
        key_scale = key_scale.repeat_interleave(_K_BLOCK, dim=2)[:, :, : key.shape[2]]
    elif qk_quantization == "per_thread":
        query_int8, query_scale = _quantize_query_per_thread(query, _QK_RANGE)
        key_int8, key_scale = _quantize_key_per_thread(key_centered, _QK_RANGE)
    else:
        raise ValueError(f"unknown Q/K quantization granularity: {qk_quantization}")

    if paired_rotation:
        value_rotated = value.float()
    else:
        value_rotated = (
            value.float()
            if rotation_group == 0
            else rotate_attention_groups(value.float(), rotation_group)
        )
    batch, heads, query_length, width = query.shape
    key_length = key.shape[2]
    accumulator = torch.zeros(
        (batch, heads, query_length, width), device=query.device, dtype=torch.float32
    )
    denominator = torch.zeros(
        (batch, heads, query_length), device=query.device, dtype=torch.float32
    )
    running_max = torch.full_like(denominator, -float("inf"))
    query_positions = torch.arange(query_length, device=query.device)

    for start in range(0, key_length, _PV_BLOCK):
        stop = min(start + _PV_BLOCK, key_length)
        integer_scores = torch.matmul(
            query_int8.float(), key_int8[:, :, start:stop].transpose(-1, -2).float()
        )
        scores = (
            integer_scores * query_scale[:, :, :, None] * key_scale[:, :, None, start:stop] * scale
        )
        if is_causal:
            key_positions = torch.arange(start, stop, device=query.device)
            scores = scores.masked_fill(
                key_positions[None, None, None, :] > query_positions[None, None, :, None],
                -float("inf"),
            )
        block_max = scores.amax(dim=-1)
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        probabilities = torch.nan_to_num(torch.exp(scores - next_max[..., None]))
        accumulator *= old_weight[..., None]
        denominator = denominator * old_weight + probabilities.sum(dim=-1)

        value_block = value_rotated[:, :, start:stop]
        if paired_rotation:
            block_width = stop - start
            if block_width < _PV_BLOCK:
                padded_probability = probabilities.new_zeros((*probabilities.shape[:-1], _PV_BLOCK))
                padded_probability[..., :block_width] = probabilities
                probabilities = padded_probability
                padded_value = value_block.new_zeros(
                    (*value_block.shape[:2], _PV_BLOCK, value_block.shape[-1])
                )
                padded_value[:, :, :block_width] = value_block
                value_block = padded_value
            probability_for_dot = rotate_attention_groups(probabilities, rotation_group)
            value_for_dot = rotate_attention_groups(
                value_block.transpose(-1, -2), rotation_group
            ).transpose(-1, -2)
            probability_min = probability_for_dot.amin(dim=-1)
            probability_max = probability_for_dot.amax(dim=-1)
            probability_scale = torch.where(
                probability_max > probability_min,
                (probability_max - probability_min) / _P_RANGE,
                torch.ones_like(probability_max),
            )
            probability_zero = (-probability_min / probability_scale).round().clamp(0, _P_RANGE)
            probability_uint4 = (
                (probability_for_dot / probability_scale[..., None] + probability_zero[..., None])
                .round()
                .clamp(0, _P_RANGE)
                .to(torch.int8)
            )
        else:
            probability_uint4, probability_scale = _quantize_probability_uint4(probabilities)
            probability_zero = torch.zeros_like(probability_scale)
            value_for_dot = value_block
        value_scale = value_for_dot.abs().amax(dim=2) / _V_RANGE + 1e-7
        value_int4 = (
            (value_for_dot / value_scale[:, :, None, :])
            .round()
            .clamp(-_V_RANGE, _V_RANGE)
            .to(torch.int8)
        )
        partial = torch.matmul(probability_uint4.float(), value_int4.float())
        if paired_rotation:
            partial -= probability_zero[..., None] * value_int4.float().sum(dim=2)[:, :, None]
        accumulator += partial * probability_scale[..., None] * value_scale[:, :, None, :]
        running_max = next_max

    rotated_output = accumulator / denominator.clamp_min(1e-30)[..., None]
    output = (
        rotated_output
        if paired_rotation or rotation_group == 0
        else rotate_attention_groups(rotated_output, rotation_group)
    )
    return output.to(output_dtype)
