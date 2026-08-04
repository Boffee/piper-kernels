"""Paired-Hadamard signed-INT8 PV with RMS-derived scales."""

# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

from piper_kernels.attention._convrot_triton import rotate_rows_in_registers
from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.uint4_pv_convrot import _prepare_qk

_INT8_RANGE = tl.constexpr(127.0)
_P_RMS_MULTIPLIER = tl.constexpr(2.781)
_V_RMS_MULTIPLIER_G1 = 3.396
_V_RMS_MULTIPLIER_G2 = 3.525
_PV_BLOCK = 64
_PV_BLOCK_TL = tl.constexpr(64)


@triton.jit
def _quantize_value_paired_rms_kernel(
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
    group_tiles: tl.constexpr,
    value_rms_multiplier: tl.constexpr,
):
    group = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    block_n: tl.constexpr = _PV_BLOCK_TL * group_tiles
    offsets_n = group * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid_n = offsets_n < key_length
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid_n[:, None],
        other=0.0,
    ).to(tl.float32)
    valid_count = tl.maximum(1, tl.minimum(block_n, key_length - group * block_n))
    rms = tl.sqrt(tl.sum(value * value, axis=0) / valid_count)
    scale = rms * (value_rms_multiplier / _INT8_RANGE) + 1e-30

    rotated = rotate_rows_in_registers(
        tl.trans(value),
        offsets_n,
        head_dim,
        _PV_BLOCK_TL,
    )
    rotated = tl.trans(rotated)
    quantized = _sage_backend._round_to_int8(rotated / scale[None, :], _INT8_RANGE)
    scale_offset = ((batch * heads + head) * tl.cdiv(key_length, block_n) + group) * head_dim
    tl.store(scale_ptr + scale_offset + offsets_d, scale)
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_n[:, None] * stride_on
        + offsets_d[None, :],
        quantized,
        mask=valid_n[:, None],
    )


@triton.jit
def _value_group_rms_kernel(
    value_ptr,
    scale_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    group_tiles: tl.constexpr,
    value_rms_multiplier: tl.constexpr,
    block_d: tl.constexpr,
):
    group = tl.program_id(0)
    batch_head = tl.program_id(1)
    block_d_id = tl.program_id(2)
    batch = batch_head // heads
    head = batch_head % heads
    block_n: tl.constexpr = _PV_BLOCK_TL * group_tiles
    offsets_n = group * block_n + tl.arange(0, block_n)
    offsets_d = block_d_id * block_d + tl.arange(0, block_d)
    mask = (offsets_n[:, None] < key_length) & (offsets_d[None, :] < head_dim)
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    valid_count = tl.maximum(1, tl.minimum(block_n, key_length - group * block_n))
    rms = tl.sqrt(tl.sum(value * value, axis=0) / valid_count)
    scale = rms * (value_rms_multiplier / _INT8_RANGE) + 1e-30
    scale_offset = (batch_head * tl.cdiv(key_length, block_n) + group) * head_dim
    tl.store(scale_ptr + scale_offset + offsets_d, scale, mask=offsets_d < head_dim)


@triton.jit
def _quantize_value_paired_with_scale_kernel(
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
    group_tiles: tl.constexpr,
    block_d: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    block_d_id = tl.program_id(2)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_n = tile * _PV_BLOCK_TL + tl.arange(0, _PV_BLOCK_TL)
    offsets_d = block_d_id * block_d + tl.arange(0, block_d)
    mask = (offsets_n[:, None] < key_length) & (offsets_d[None, :] < head_dim)
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    rotated = rotate_rows_in_registers(
        tl.trans(value),
        offsets_n,
        block_d,
        _PV_BLOCK_TL,
    )
    rotated = tl.trans(rotated)
    group = tile // group_tiles
    scale_offset = (batch_head * tl.cdiv(key_length, _PV_BLOCK_TL * group_tiles) + group) * head_dim
    scale = tl.load(
        scale_ptr + scale_offset + offsets_d,
        mask=offsets_d < head_dim,
        other=1.0,
    )
    quantized = _sage_backend._round_to_int8(rotated / scale[None, :], _INT8_RANGE)
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
def _int8_pv_paired_rms_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    local_probability: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    group_tiles: tl.constexpr,
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, _PV_BLOCK_TL)
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

    for start_n in tl.range(0, end_n, _PV_BLOCK_TL, disable_licm=True):
        current_n = start_n + offsets_n
        key = tl.load(
            key_ptr
            + ((batch * heads + head) * key_length + current_n[None, :]) * head_dim
            + offsets_d[:, None],
            mask=current_n[None, :] < key_length,
            other=0,
        )
        integer_scores = tl.dot(query, key, out_dtype=tl.int32)
        if grouped_qk:
            key_scale = tl.load(
                key_scale_ptr + (batch * heads + head) * tl.cdiv(key_length, 64) + start_n // 64
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
        if local_probability:
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
        accumulator *= old_weight[:, None]
        denominator = denominator * old_weight + tl.sum(probabilities, axis=1) * current_weight

        probability_rms = tl.sqrt(tl.sum(probabilities * probabilities, axis=1) / _PV_BLOCK_TL)
        probability_scale = probability_rms * (_P_RMS_MULTIPLIER / _INT8_RANGE) + 1e-30
        rotated_probability = rotate_rows_in_registers(
            probabilities,
            offsets_n,
            block_m,
            _PV_BLOCK_TL,
        )
        probability_int8 = _sage_backend._round_to_int8(
            rotated_probability / probability_scale[:, None],
            _INT8_RANGE,
        )
        value = tl.load(
            value_ptr
            + ((batch * heads + head) * key_length + current_n[:, None]) * head_dim
            + offsets_d[None, :],
            mask=current_n[:, None] < key_length,
            other=0,
        )
        partial_int32 = tl.dot(probability_int8, value, out_dtype=tl.int32)
        value_group = start_n // (_PV_BLOCK_TL * group_tiles)
        value_scale = tl.load(
            value_scale_ptr
            + (
                (batch * heads + head) * tl.cdiv(key_length, _PV_BLOCK_TL * group_tiles)
                + value_group
            )
            * head_dim
            + offsets_d
        )
        accumulator += (
            partial_int32.to(tl.float32)
            * probability_scale[:, None]
            * value_scale[None, :]
            * current_weight[:, None]
        )
        running_max = next_max

    output = accumulator / tl.maximum(denominator, 1e-30)[:, None]
    tl.store(
        output_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        output,
        mask=valid_queries[:, None],
    )


def _value_rms_multiplier(group_tiles: int) -> float:
    if group_tiles == 1:
        return _V_RMS_MULTIPLIER_G1
    if group_tiles == 2:
        return _V_RMS_MULTIPLIER_G2
    raise ValueError(f"V RMS grouping must use one or two K=64 tiles, got {group_tiles}")


def _prepare_int8_pv_convrot_rms_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    value_rms_group_tiles: int,
) -> tuple[torch.Tensor, ...]:
    """Quantize canonical Q/K and paired-Hadamard RMS-scaled V."""
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        scale,
        grouped_qk,
    )
    batch, heads, key_length, head_dim = value.shape
    group_width = _PV_BLOCK * value_rms_group_tiles
    value_int8 = torch.empty(value.shape, device=value.device, dtype=torch.int8)
    value_groups = (key_length + group_width - 1) // group_width
    value_scale = torch.empty(
        (batch, heads, value_groups, head_dim),
        device=value.device,
        dtype=torch.float32,
    )
    if value_rms_group_tiles == 1:
        _quantize_value_paired_rms_kernel[(value_groups, heads, batch)](
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
            group_tiles=value_rms_group_tiles,
            value_rms_multiplier=_value_rms_multiplier(value_rms_group_tiles),
            num_warps=4,
        )
    else:
        block_d = 32
        _value_group_rms_kernel[(value_groups, batch * heads, triton.cdiv(head_dim, block_d))](
            value,
            value_scale,
            key_length,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            heads=heads,
            head_dim=head_dim,
            group_tiles=value_rms_group_tiles,
            value_rms_multiplier=_value_rms_multiplier(value_rms_group_tiles),
            block_d=block_d,
            num_warps=4,
        )
        _quantize_value_paired_with_scale_kernel[
            (triton.cdiv(key_length, _PV_BLOCK), batch * heads, triton.cdiv(head_dim, block_d))
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
            group_tiles=value_rms_group_tiles,
            block_d=block_d,
            num_warps=4,
        )
    return query_int8, key_int8, value_int8, query_scale, key_scale, value_scale


def _launch_int8_pv_convrot_rms_attention(
    prepared: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    is_causal: bool,
    *,
    grouped_qk: bool,
    value_rms_group_tiles: int,
    local_probability: bool,
    block_m: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    query, key, value, query_scale, key_scale, value_scale = prepared
    batch, heads, _, head_dim = query.shape
    _int8_pv_paired_rms_attention_kernel[(triton.cdiv(query_length, block_m), heads, batch)](
        query,
        key,
        value,
        query_scale,
        key_scale,
        value_scale,
        output,
        query_length,
        key_length,
        is_causal=is_causal,
        grouped_qk=grouped_qk,
        local_probability=local_probability,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        group_tiles=value_rms_group_tiles,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def triton_sage_attention_int8_pv_convrot_rms(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
    value_rms_group_tiles: int = 1,
    local_probability: bool = False,
) -> torch.Tensor:
    """Run paired-Hadamard signed-INT8 PV with RMS scales."""
    if grouped_qk is None:
        grouped_qk = torch.cuda.get_device_capability(query.device)[0] == 12
    prepared = _prepare_int8_pv_convrot_rms_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
        value_rms_group_tiles=value_rms_group_tiles,
    )
    batch, heads, query_length, _ = query.shape
    key_length = key.shape[2]
    output = torch.empty_like(query)
    block_m = (
        64 if is_causal else _sage_backend._select_query_block(query, batch, heads, query_length)
    )
    return _launch_int8_pv_convrot_rms_attention(
        prepared,
        output,
        query_length,
        key_length,
        is_causal,
        grouped_qk=grouped_qk,
        value_rms_group_tiles=value_rms_group_tiles,
        local_probability=local_probability,
        block_m=block_m,
        num_warps=4,
        num_stages=3,
    )
