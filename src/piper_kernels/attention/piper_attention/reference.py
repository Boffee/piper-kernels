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

from ._validation import normalize_window_bounds, resolve_kv_window_modes

_CENTER_CHUNK = 1024
_PV_BLOCK = 64
_PV_TILES_PER_CHUNK = _CENTER_CHUNK // _PV_BLOCK
_P_UINT8_RANGE = 255.0
_V_INT8_RANGE = 127.0
_SCALE_EPSILON = 1e-7
# Keep per-chunk mean restoration finite when a constant chunk has tiny residual scale.
_VALUE_MEAN_SCALE_FLOOR = 2.0**-80


def _disable_autocast(device: torch.device) -> contextlib.AbstractContextManager[object]:
    if torch.amp.is_autocast_available(device.type):
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def _wmma_token_perm(device: torch.device) -> torch.Tensor:
    tokens = torch.arange(_PV_BLOCK, device=device, dtype=torch.int64)
    return (tokens & ~24) | ((tokens & 8) << 1) | ((tokens & 16) >> 1)


def _advance_winner_frame(
    running_max: torch.Tensor,
    running_bias: torch.Tensor,
    scores: torch.Tensor,
    block_max: torch.Tensor,
    block_bias: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    has_nan = torch.isnan(scores).any(dim=-1) | torch.isnan(block_max + block_bias)
    m_clean = torch.where(has_nan, float("nan"), block_max)
    has_block = ((m_clean != -float("inf")) & (block_bias != -float("inf"))) | has_nan
    has_running = running_max != -float("inf")
    bias_diff = torch.where(block_bias == running_bias, 0.0, block_bias - running_bias)
    raw_delta = (m_clean - running_max) + bias_diff
    delta = torch.where(
        has_block,
        torch.where(has_running, raw_delta, float("inf")),
        -float("inf"),
    )
    tile_wins = (delta > 0.0) | has_nan
    next_max = torch.where(tile_wins, m_clean, running_max)
    next_bias = torch.where(tile_wins, block_bias, running_bias)
    decay = torch.exp(-torch.abs(delta))
    old_weight = torch.where(has_running, torch.where(tile_wins, decay, 1.0), 0.0)
    current_weight = torch.where(has_block, torch.where(tile_wins, 1.0, decay), 0.0)
    safe_block_max = torch.where(has_block, m_clean, 0.0)
    return next_max, next_bias, old_weight, current_weight, has_block, safe_block_max


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


def reference_piper_attention(  # noqa: PLR0915
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    qk_quantization: QKQuantizationGranularity = "per_thread",
    window_left: int | None = None,
    window_right: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Piper Attention with K1024 + V1024 chunk centering and INT8 Q/K/V."""
    output_dtype = query.dtype
    if query.shape[0] == 0:
        empty_out = torch.empty_like(query, memory_format=torch.contiguous_format)
        if not return_lse:
            return empty_out
        empty_lse = torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        return empty_out, empty_lse

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

    eff_left, eff_right, window_mode = normalize_window_bounds(is_causal, window_left, window_right)
    has_window = window_mode != "none"
    if isinstance(eff_right, int):
        k_max_excl = min(key_length, (query.shape[2] - 1) + eff_right + 1)
    else:
        k_max_excl = key_length

    for chunk_start in range(0, k_max_excl, _CENTER_CHUNK):
        chunk = chunk_start // _CENTER_CHUNK
        chunk_end = min(chunk_start + _CENTER_CHUNK, k_max_excl)
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
            if has_window:
                key_positions = torch.arange(start, stop, device=query.device)
                valid = torch.ones(
                    (query_length, stop - start),
                    dtype=torch.bool,
                    device=query.device,
                )
                if eff_left is not None:
                    valid &= (query_positions[:, None] - key_positions[None, :]) <= eff_left
                if eff_right is not None:
                    valid &= (key_positions[None, :] - query_positions[:, None]) <= eff_right
                scores = scores.masked_fill(~valid[None, None, :, :], -float("inf"))

            block_value_scale = value_scale[:, :, start:stop]
            shifted_scores = torch.where(
                scores != -float("inf"),
                scores + torch.log(block_value_scale[:, :, None, :]),
                -float("inf"),
            )
            block_max = shifted_scores.amax(dim=-1)
            tile_bias = torch.where(block_max != -float("inf"), chunk_bias, 0.0)
            next_max, next_bias, old_weight, current_weight, has_block, safe_block_max = (
                _advance_winner_frame(running_max, running_bias, scores, block_max, tile_bias)
            )
            probabilities = torch.where(
                has_block[..., None],
                torch.exp(scores - safe_block_max[..., None]),
                0.0,
            )

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
        numerator += torch.where(
            (chunk_denom > 0.0)[..., None],
            (chunk_denom * _P_UINT8_RANGE)[..., None] * value_mean[:, :, chunk : chunk + 1],
            0.0,
        )

    has_any = running_max != -float("inf")
    safe_denom = torch.where(has_any & (denominator != 0.0), denominator, 1.0)
    output = (
        torch.where(
            has_any[..., None],
            numerator / (safe_denom[..., None] * _P_UINT8_RANGE),
            0.0,
        )
        .reshape(query.shape)
        .to(output_dtype)
    )
    if not return_lse:
        return output
    lse = torch.where(
        has_any,
        (running_max + running_bias) + torch.log(safe_denom),
        -float("inf"),
    ).reshape(query.shape[:3])
    return output, lse


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
    with _disable_autocast(key.device):
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
            src_value_q = src_value_q.index_select(-1, _wmma_token_perm(value.device))

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


def _online_softmax_update(
    running_max: torch.Tensor,
    running_bias: torch.Tensor,
    running_sum: torch.Tensor,
    acc: torch.Tensor,
    scores: torch.Tensor,
    block_bias: torch.Tensor,
    values: torch.Tensor,
    *,
    chunk_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_max, next_bias, old_weight, current_weight, has_block, safe_block_max = (
        _advance_winner_frame(running_max, running_bias, scores, scores.amax(dim=-1), block_bias)
    )
    probs = torch.where(
        has_block[..., None],
        torch.exp(scores - safe_block_max[..., None]) * current_weight[..., None],
        0.0,
    )
    block_sum = probs.sum(dim=-1)
    next_sum = running_sum * old_weight + block_sum
    next_acc = acc * old_weight[..., None] + torch.where(
        has_block[..., None],
        torch.matmul(probs, values),
        0.0,
    )
    if chunk_mean is not None:
        next_acc = next_acc + torch.where(
            has_block[..., None],
            block_sum[..., None] * chunk_mean,
            0.0,
        )
    return next_max, next_bias, next_sum, next_acc


def _decode_frozen_chunks_reference(  # noqa: PLR0913, PLR0915
    q_rot: torch.Tensor,
    running_max: torch.Tensor,
    running_bias: torch.Tensor,
    running_sum: torch.Tensor,
    acc: torch.Tensor,
    *,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    active_tail: torch.Tensor | None,
    seq_q: int,
    scale: float,
    pack_wmma: bool,
    window_left: int | None,
    window_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_slots, kv_heads, _, head_dim = key_int8.shape
    batch = q_rot.shape[0]
    gq = q_rot.shape[2]
    gqa_group = gq // seq_q
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    if max_chunks == 0 or num_slots == 0:
        return running_max, running_bias, running_sum, acc

    grouped_qk = key_scale.shape[2] == _PV_TILES_PER_CHUNK
    active_chunks = num_chunks.clamp(0, max_chunks)
    wmma_perm = _wmma_token_perm(q_rot.device) if pack_wmma else None

    key_i8_signed = key_int8.view(torch.int8)
    val_i8_signed = value_int8.view(torch.int8)

    has_window = window_left is not None or window_right is not None
    if has_window:
        tail_tokens = (
            active_tail.to(torch.int64)
            if active_tail is not None
            else torch.zeros(batch, device=q_rot.device, dtype=torch.int64)
        )
        total_kv = active_chunks.to(torch.int64) * _CENTER_CHUNK + tail_tokens
        q_idx = torch.arange(seq_q, device=q_rot.device, dtype=torch.int64)
        q_pos = total_kv[:, None] - seq_q + q_idx[None, :]
        local_k = torch.arange(_CENTER_CHUNK, device=q_rot.device, dtype=torch.int64)
    else:
        q_pos = None
        local_k = None

    for chunk_idx in range(max_chunks):
        slots = (
            torch.full((batch,), chunk_idx, device=q_rot.device, dtype=torch.int32)
            if chunk_to_slot is None
            else chunk_to_slot[:, chunk_idx]
        )
        slot_valid = (chunk_idx < active_chunks) & (slots >= 0) & (slots < num_slots)
        safe_idx = torch.where(slot_valid, slots, 0).to(torch.int64)

        k_q = key_i8_signed.index_select(0, safe_idx).float()
        k_s = torch.where(slot_valid[:, None, None], key_scale.index_select(0, safe_idx), 0.0)
        k_c = torch.where(slot_valid[:, None, None], key_center.index_select(0, safe_idx), 0.0)
        v_q_tile = val_i8_signed.index_select(0, safe_idx)
        v_s = torch.where(slot_valid[:, None, None], value_scale.index_select(0, safe_idx), 0.0)
        v_m = torch.where(slot_valid[:, None, None], value_mean.index_select(0, safe_idx), 0.0)

        if wmma_perm is not None:
            v_q_tile = v_q_tile.index_select(-1, wmma_perm)
        v_q_tok = v_q_tile.transpose(-2, -1).reshape(batch, k_q.shape[1], _CENTER_CHUNK, head_dim)

        k_tok_scale = k_s.repeat_interleave(_PV_BLOCK, dim=-1) if grouped_qk else k_s
        k_centered = k_q * k_tok_scale.unsqueeze(-1)
        scores_centered = torch.matmul(q_rot, k_centered.transpose(-1, -2))
        scores = scores_centered * scale
        if has_window:
            assert q_pos is not None
            assert local_k is not None
            k_pos = chunk_idx * _CENTER_CHUNK + local_k
            win_mask = torch.ones(
                (batch, seq_q, _CENTER_CHUNK),
                dtype=torch.bool,
                device=q_rot.device,
            )
            if window_left is not None:
                win_mask &= (q_pos[:, :, None] - k_pos[None, None, :]) <= window_left
            if window_right is not None:
                win_mask &= (k_pos[None, None, :] - q_pos[:, :, None]) <= window_right
            win_mask &= slot_valid[:, None, None]
            full_mask = (
                win_mask[:, None, None, :, :]
                .expand(batch, kv_heads, gqa_group, seq_q, _CENTER_CHUNK)
                .reshape(batch, kv_heads, gq, _CENTER_CHUNK)
            )
            scores = scores.masked_fill(~full_mask, -float("inf"))
            row_has_valid = full_mask.any(dim=-1)
        else:
            scores = scores.masked_fill(~slot_valid[:, None, None, None], -float("inf"))
            row_has_valid = slot_valid[:, None, None]

        k_bias = (q_rot * k_c.unsqueeze(2)).sum(dim=-1)
        chunk_bias = torch.where(row_has_valid, k_bias * scale, 0.0)

        v_centered = torch.where(
            slot_valid[:, None, None, None],
            v_q_tok.float() * v_s.unsqueeze(-1),
            0.0,
        )
        running_max, running_bias, running_sum, acc = _online_softmax_update(
            running_max,
            running_bias,
            running_sum,
            acc,
            scores,
            chunk_bias,
            v_centered,
            chunk_mean=v_m.unsqueeze(2),
        )

    return running_max, running_bias, running_sum, acc


def _decode_tail_reference(  # noqa: PLR0913
    q_grouped: torch.Tensor,
    running_max: torch.Tensor,
    running_bias: torch.Tensor,
    running_sum: torch.Tensor,
    acc: torch.Tensor,
    *,
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    tail_len: torch.Tensor,
    seq_q: int,
    scale: float,
    window_left: int | None,
    window_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tail_cap = tail_key.shape[2]
    if tail_cap == 0:
        return running_max, running_bias, running_sum, acc
    batch, kv_heads, _, _ = tail_key.shape
    q_flat = q_grouped.reshape(batch, kv_heads, -1, q_grouped.shape[-1])
    active_tail = tail_len.clamp(0, tail_cap)
    k_pos = torch.arange(tail_cap, device=q_grouped.device, dtype=torch.int64)
    tail_valid = k_pos[None, :] < active_tail[:, None]
    tail_k_f = torch.where(tail_valid[:, None, :, None], tail_key.float(), 0.0)
    tail_v_f = torch.where(tail_valid[:, None, :, None], tail_value.float(), 0.0)
    scores = torch.matmul(q_flat, tail_k_f.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~tail_valid[:, None, None, :], -float("inf"))
    if window_left is not None or window_right is not None:
        q_pos = torch.arange(seq_q, device=q_grouped.device, dtype=torch.int64)
        # Reshape into [B, kv_heads, gqa_group, seq_q, tail_cap] for row-level window mask.
        scores_q = scores.reshape(batch, kv_heads, -1, seq_q, tail_cap)
        q_tail_pos = active_tail[:, None].to(torch.int64) - seq_q + q_pos[None, :]
        invalid_mask = torch.zeros(
            (batch, seq_q, tail_cap), dtype=torch.bool, device=q_grouped.device
        )
        if window_left is not None:
            invalid_mask |= (q_tail_pos[:, :, None] - k_pos[None, None, :]) > window_left
        if window_right is not None:
            invalid_mask |= (k_pos[None, None, :] - q_tail_pos[:, :, None]) > window_right
        scores = scores_q.masked_fill(invalid_mask[:, None, None, :, :], -float("inf")).reshape(
            batch, kv_heads, -1, tail_cap
        )
    return _online_softmax_update(
        running_max,
        running_bias,
        running_sum,
        acc,
        scores,
        torch.zeros_like(running_bias),
        tail_v_f,
    )


def piper_attention_decode_reference(  # noqa: PLR0913
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    tail_key: torch.Tensor | None = None,
    tail_value: torch.Tensor | None = None,
    tail_len: torch.Tensor | None = None,
    scale: float,
    is_causal: bool = True,
    pack_wmma: bool = False,
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable FP32-accumulation reference for GQA chunked INT8 + BF16/FP16 tail decode."""
    batch, q_heads, seq_q, head_dim = query.shape
    if batch == 0:
        return (
            torch.empty((0, q_heads, seq_q, head_dim), device=query.device, dtype=query.dtype),
            torch.empty((0, q_heads, seq_q), device=query.device, dtype=torch.float32),
        )
    kv_heads = key_int8.shape[1]
    gqa_group = q_heads // kv_heads
    gq = gqa_group * seq_q
    windows = resolve_kv_window_modes(
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
    )
    frozen_left = None if windows.frozen_left < 0 else windows.frozen_left
    frozen_right = None if windows.frozen_right < 0 else windows.frozen_right
    tail_left = None if windows.tail_left < 0 else windows.tail_left
    tail_right = None if windows.tail_right < 0 else windows.tail_right

    with _disable_autocast(query.device):
        q_grouped = query.float().reshape(batch, kv_heads, gqa_group, seq_q, head_dim)
        q_flat = q_grouped.reshape(batch, kv_heads, gq, head_dim)
        running_max = torch.full(
            (batch, kv_heads, gq),
            -float("inf"),
            device=query.device,
            dtype=torch.float32,
        )
        running_bias = torch.zeros_like(running_max)
        running_sum = torch.zeros_like(running_max)
        acc = torch.zeros(
            (batch, kv_heads, gq, head_dim),
            device=query.device,
            dtype=torch.float32,
        )

        active_tail = (
            tail_len.clamp(0, tail_key.shape[2])
            if tail_key is not None and tail_len is not None
            else None
        )
        max_chunks = key_int8.shape[0] if chunk_to_slot is None else chunk_to_slot.shape[1]
        if max_chunks > 0 and key_int8.shape[0] > 0:
            q_rot = rotate_signed_hadamard_heads(q_flat)
            running_max, running_bias, running_sum, acc = _decode_frozen_chunks_reference(
                q_rot,
                running_max,
                running_bias,
                running_sum,
                acc,
                key_int8=key_int8,
                key_scale=key_scale,
                key_center=key_center,
                value_int8=value_int8,
                value_scale=value_scale,
                value_mean=value_mean,
                chunk_to_slot=chunk_to_slot,
                num_chunks=num_chunks,
                active_tail=active_tail,
                seq_q=seq_q,
                scale=scale,
                pack_wmma=pack_wmma,
                window_left=frozen_left,
                window_right=frozen_right,
            )

        if tail_key is not None and tail_value is not None and tail_len is not None:
            running_max, running_bias, running_sum, acc = _decode_tail_reference(
                q_grouped,
                running_max,
                running_bias,
                running_sum,
                acc,
                tail_key=tail_key,
                tail_value=tail_value,
                tail_len=tail_len,
                seq_q=seq_q,
                scale=scale,
                window_left=tail_left,
                window_right=tail_right,
            )

        has_any = running_max != -float("inf")
        safe_denom = torch.where(has_any, running_sum, 1.0)
        output = torch.where(has_any[..., None], acc / safe_denom[..., None], 0.0)
        lse = torch.where(
            has_any, (running_max + running_bias) + torch.log(safe_denom), -float("inf")
        )
        return (
            output.reshape(batch, q_heads, seq_q, head_dim).to(query.dtype),
            lse.reshape(batch, q_heads, seq_q),
        )
