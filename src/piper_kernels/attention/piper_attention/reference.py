"""Portable reference for Piper Attention's key-scaled integer-PV attention.

Piper Attention follows the fused online-softmax structure of FlashAttention
and the INT8 QK smoothing/quantization of SageAttention. Its distinct PV path
uses one signed-INT8 scale per V row and a nonnegative UINT8 probability
operand. The reference is intentionally readable rather than fast.
"""

import contextlib

import torch

from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import (
    rotate_signed_hadamard_heads,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    QKQuantizationGranularity,
    quantize_centered_query_key,
)

_CENTER_CHUNK = 1024
_PV_BLOCK = 64
_PV_TILES_PER_CHUNK = _CENTER_CHUNK // _PV_BLOCK
_P_UINT8_RANGE = 255.0
_V_INT8_RANGE = 127.0
_SCALE_EPSILON = 1e-7
# Keep per-chunk mean restoration finite when a constant chunk has tiny residual scale.
_VALUE_MEAN_SCALE_FLOOR = 2.0**-80


def _quantize_value_per_key(
    value: torch.Tensor,
    value_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_float = value.float()
    mean_floor = (
        value_mean.float().abs().amax(dim=-1) * _VALUE_MEAN_SCALE_FLOOR
    ).repeat_interleave(_CENTER_CHUNK, dim=-1)[:, :, : value.shape[2]]
    scale = torch.maximum(
        value_float.abs().amax(dim=-1) / _V_INT8_RANGE + _SCALE_EPSILON,
        mean_floor,
    )
    quantized = (
        (value_float / scale[..., None]).round().clamp(-_V_INT8_RANGE, _V_INT8_RANGE).to(torch.int8)
    )
    return quantized, scale


def _chunk_means(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract and return per-1024-token chunk means in FP32."""
    float_tensor = tensor.float()
    batch, heads, length, head_dim = float_tensor.shape
    chunks = (length + _CENTER_CHUNK - 1) // _CENTER_CHUNK
    means = float_tensor.new_empty((batch, heads, chunks, head_dim))
    centered = torch.empty_like(float_tensor)
    for chunk in range(chunks):
        start = chunk * _CENTER_CHUNK
        stop = min(start + _CENTER_CHUNK, length)
        chunk_mean = float_tensor[:, :, start:stop].mean(dim=2)
        means[:, :, chunk] = chunk_mean
        centered[:, :, start:stop] = float_tensor[:, :, start:stop] - chunk_mean[:, :, None, :]
    return centered, means


def _prepare_reference_operands(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    qk_quantization: QKQuantizationGranularity,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    key_centered, key_mean = _chunk_means(key)
    value_centered, value_mean = _chunk_means(value)
    key_center = rotate_signed_hadamard_heads(key_mean).float()
    query_int8, key_int8, query_scale, key_scale = quantize_centered_query_key(
        query,
        key_centered,
        granularity=qk_quantization,
    )
    value_int8, value_scale = _quantize_value_per_key(value_centered, value_mean)
    batch, query_heads, query_length, width = query.shape
    heads = key.shape[1]
    groups = query_heads // heads
    query_int8 = query_int8.reshape(batch, heads, groups * query_length, width)
    query_scale = query_scale.reshape(batch, heads, groups * query_length)
    k_bias = (
        torch.matmul(
            query_int8.float() * query_scale[..., None],
            key_center.transpose(-1, -2),
        )
        * scale
    )
    return query_int8, key_int8, query_scale, key_scale, k_bias, value_int8, value_scale, value_mean


def _advance_tile_max(
    running_max: torch.Tensor,
    running_bias: torch.Tensor,
    block_max: torch.Tensor,
    chunk_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update winner-frame running maximum across chunks without round-tripping."""
    has_block = block_max != -float("inf")
    has_running = running_max != -float("inf")
    raw_delta = (block_max - running_max) + (chunk_bias - running_bias)
    delta = torch.where(
        has_block,
        torch.where(has_running, raw_delta, float("inf")),
        -float("inf"),
    )
    tile_wins = delta > 0.0
    next_max = torch.where(tile_wins, block_max, running_max)
    next_bias = torch.where(tile_wins, chunk_bias, running_bias)
    decay = torch.exp(-torch.abs(delta))
    old_weight = torch.where(tile_wins, decay, 1.0)
    current_weight = torch.where(tile_wins, 1.0, decay)
    return next_max, next_bias, old_weight, current_weight


def reference_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    qk_quantization: QKQuantizationGranularity = "per_thread",
) -> torch.Tensor:
    """Evaluate Piper Attention with K1024 + V1024 chunk centering and INT8 Q/K/V."""
    output_dtype = query.dtype
    (
        query_int8,
        key_int8,
        query_scale,
        key_scale,
        k_bias,
        value_int8,
        value_scale,
        value_mean,
    ) = _prepare_reference_operands(query, key, value, scale, qk_quantization)

    batch, heads, query_length, width = query_int8.shape
    key_length = key.shape[2]
    numerator = torch.zeros(
        (batch, heads, query_length, width),
        device=query.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros(
        (batch, heads, query_length),
        device=query.device,
        dtype=torch.float32,
    )
    running_max = torch.full_like(denominator, -float("inf"))
    running_bias = torch.zeros_like(denominator)
    query_positions = torch.arange(query_length, device=query.device) % query.shape[2]

    for chunk_start in range(0, key_length, _CENTER_CHUNK):
        chunk = chunk_start // _CENTER_CHUNK
        chunk_end = min(chunk_start + _CENTER_CHUNK, key_length)
        chunk_bias = k_bias[:, :, :, chunk]
        chunk_denom = torch.zeros_like(denominator)
        for start in range(chunk_start, chunk_end, _PV_BLOCK):
            stop = min(start + _PV_BLOCK, chunk_end)
            key_block = key_int8[:, :, start:stop]
            integer_scores = torch.matmul(
                query_int8.float(),
                key_block.transpose(-1, -2).float(),
            )
            scores = (
                integer_scores
                * query_scale[:, :, :, None]
                * key_scale[:, :, None, start:stop]
                * scale
            )
            if is_causal:
                key_positions = torch.arange(start, stop, device=query.device)
                scores = scores.masked_fill(
                    key_positions[None, None, None, :] > query_positions[None, None, :, None],
                    -float("inf"),
                )

            block_value_scale = value_scale[:, :, start:stop]
            shifted_scores = scores + torch.log(block_value_scale[:, :, None, :])
            block_max = shifted_scores.amax(dim=-1)
            next_max, next_bias, old_weight, current_weight = _advance_tile_max(
                running_max,
                running_bias,
                block_max,
                chunk_bias,
            )
            probabilities = torch.nan_to_num(torch.exp(scores - block_max[..., None]))

            row_sum = probabilities.sum(dim=-1) * current_weight
            numerator *= old_weight[..., None]
            denominator = denominator * old_weight + row_sum
            chunk_denom = chunk_denom * old_weight + row_sum
            probability_codes = (
                (probabilities * block_value_scale[:, :, None, :] * _P_UINT8_RANGE)
                .round()
                .clamp(0, _P_UINT8_RANGE)
            )
            partial = torch.matmul(
                probability_codes,
                value_int8[:, :, start:stop].float(),
            )
            numerator += partial * current_weight[..., None]
            running_max = next_max
            running_bias = next_bias
        numerator += (chunk_denom * _P_UINT8_RANGE)[..., None] * value_mean[:, :, chunk : chunk + 1]

    output = numerator / (denominator.clamp_min(1e-30)[..., None] * _P_UINT8_RANGE)
    return output.reshape(query.shape).to(output_dtype)


def _round_to_int8(values: torch.Tensor) -> torch.Tensor:
    rounded = values + torch.where(values >= 0, 0.5, -0.5)
    return rounded.clamp(-_V_INT8_RANGE, _V_INT8_RANGE).to(torch.int8)


def _scatter_slots_reference(
    slot_ids: torch.Tensor,
    dst_tensors: tuple[torch.Tensor, ...],
    src_tensors: tuple[torch.Tensor, ...],
) -> None:
    num_slots = dst_tensors[0].shape[0]
    num_new = slot_ids.shape[0]
    if num_new == 0 or num_slots == 0:
        return
    valid = (slot_ids >= 0) & (slot_ids < num_slots)
    is_slot0 = valid & (slot_ids == 0)
    slot0_updated = is_slot0.any()
    slot0_src_idx = is_slot0.to(torch.int64).argmax().reshape(1)
    safe_idx = torch.where(valid, slot_ids.to(torch.int64), 0)
    for dst, src in zip(dst_tensors, src_tensors, strict=True):
        slot0_candidate = src.index_select(0, slot0_src_idx).squeeze(0)
        slot0_val = torch.where(slot0_updated, slot0_candidate, dst[0])
        valid_mask = valid.reshape(num_new, *([1] * (dst.ndim - 1)))
        write_vals = torch.where(valid_mask, src, slot0_val.unsqueeze(0))
        dst.index_copy_(0, safe_idx, write_vals)


def freeze_kv_chunks_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    pack_wmma: bool = False,
) -> None:
    """Portable PyTorch reference for chunk-major KV centering and quantization."""
    num_new, kv_heads, _, head_dim = key.shape
    num_slots = key_int8.shape[0]
    if num_new == 0 or num_slots == 0:
        return

    grouped_qk = key_scale.shape[2] == _PV_TILES_PER_CHUNK
    autocast_ctx = (
        torch.amp.autocast(device_type=key.device.type, enabled=False)
        if key.device.type in ("cpu", "cuda")
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        key_f = key.float()
        key_mean = key_f.mean(dim=2)
        src_key_center = rotate_signed_hadamard_heads(key_mean)
        rotated_k = rotate_signed_hadamard_heads(key_f - key_mean.unsqueeze(2))
        if grouped_qk:
            rotated_tiles = rotated_k.reshape(
                num_new, kv_heads, _PV_TILES_PER_CHUNK, _PV_BLOCK, head_dim
            )
            src_key_scale = rotated_tiles.abs().amax(dim=(3, 4)) / _V_INT8_RANGE + _SCALE_EPSILON
            src_key_q = _round_to_int8(
                rotated_tiles / src_key_scale.unsqueeze(-1).unsqueeze(-1)
            ).reshape(num_new, kv_heads, _CENTER_CHUNK, head_dim)
        else:
            grouped_k = rotated_k.reshape(num_new, kv_heads, _PV_TILES_PER_CHUNK, 8, 4, 2, head_dim)
            thread_scale = (
                grouped_k.abs().amax(dim=(3, 5, 6), keepdim=True) / _V_INT8_RANGE + _SCALE_EPSILON
            )
            src_key_scale = (
                thread_scale.squeeze(-1)
                .expand(num_new, kv_heads, _PV_TILES_PER_CHUNK, 8, 4, 2)
                .reshape(num_new, kv_heads, _CENTER_CHUNK)
            )
            src_key_q = _round_to_int8(rotated_k / src_key_scale.unsqueeze(-1))

        value_f = value.float()
        src_value_mean = value_f.mean(dim=2)
        centered_v = value_f - src_value_mean.unsqueeze(2)
        scale_floor = src_value_mean.abs().amax(dim=-1, keepdim=True) * _VALUE_MEAN_SCALE_FLOOR
        src_value_scale = torch.maximum(
            centered_v.abs().amax(dim=-1) / _V_INT8_RANGE + _SCALE_EPSILON,
            scale_floor,
        )
        value_q_tok = _round_to_int8(centered_v / src_value_scale.unsqueeze(-1))
        src_value_q = (
            value_q_tok.reshape(num_new, kv_heads, _PV_TILES_PER_CHUNK, _PV_BLOCK, head_dim)
            .transpose(-2, -1)
            .contiguous()
        )
        if pack_wmma:
            tokens = torch.arange(_PV_BLOCK, device=value.device, dtype=torch.int64)
            perm = (tokens & ~24) | ((tokens & 8) << 1) | ((tokens & 16) >> 1)
            src_value_q = src_value_q.index_select(-1, perm)

        if key_int8.dtype is torch.uint8:
            src_key_q = src_key_q.view(torch.uint8)
            src_value_q = src_value_q.view(torch.uint8)

        _scatter_slots_reference(
            slot_ids,
            (key_int8, key_scale, key_center, value_int8, value_scale, value_mean),
            (
                src_key_q,
                src_key_scale,
                src_key_center,
                src_value_q,
                src_value_scale,
                src_value_mean,
            ),
        )
