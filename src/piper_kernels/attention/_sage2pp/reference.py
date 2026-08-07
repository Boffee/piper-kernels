"""Portable reference for the canonical SageAttention2++ 8+8 forward path.

SageAttention2++ originates from the SageAttention project. See the repository
NOTICE for upstream attribution.
"""

from typing import Literal

import torch

_Q_BLOCK = 32
_K_BLOCK = 64
_PV_BLOCK = 64
_P_FP8_RANGE = 448.0
_V_FP8_RANGE = 2.25
_SCALE_EPSILON = 1e-7


def _pad_sequence(value: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    length = value.shape[2]
    padded_length = (length + multiple - 1) // multiple * multiple
    if padded_length == length:
        return value, length
    padded = value.new_zeros((*value.shape[:2], padded_length, value.shape[3]))
    padded[:, :, :length] = value
    return padded, length


def _quantize_query_per_thread(
    query: torch.Tensor,
    quantization_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match SageAttention's four interleaved query rows per scale."""
    padded, length = _pad_sequence(query, _Q_BLOCK)
    batch, heads, _, width = padded.shape
    grouped = padded.float().reshape(batch, heads, -1, 4, 8, width)
    scale_group = grouped.abs().amax(dim=(3, 5)) / quantization_range + _SCALE_EPSILON
    scale = scale_group[:, :, :, None, :, None].expand_as(grouped)
    quantized = (
        (grouped / scale).round().clamp(-quantization_range, quantization_range).to(torch.int8)
    )
    return (
        quantized.reshape_as(padded)[:, :, :length],
        scale[..., 0].reshape(*padded.shape[:3])[:, :, :length],
    )


def _quantize_key_per_thread(
    key: torch.Tensor,
    quantization_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match SageAttention's sixteen interleaved key rows per scale."""
    padded, length = _pad_sequence(key, _K_BLOCK)
    batch, heads, _, width = padded.shape
    grouped = padded.float().reshape(batch, heads, -1, 8, 4, 2, width)
    scale_group = grouped.abs().amax(dim=(3, 5, 6)) / quantization_range + _SCALE_EPSILON
    scale = scale_group[:, :, :, None, :, None, None].expand_as(grouped)
    quantized = (
        (grouped / scale).round().clamp(-quantization_range, quantization_range).to(torch.int8)
    )
    return (
        quantized.reshape_as(padded)[:, :, :length],
        scale[..., 0].reshape(*padded.shape[:3])[:, :, :length],
    )


def _quantize_per_group(
    value: torch.Tensor,
    group_size: int,
    quantization_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize consecutive rows with one scale per group."""
    padded, length = _pad_sequence(value, group_size)
    batch, heads, _, width = padded.shape
    grouped = padded.float().reshape(batch, heads, -1, group_size, width)
    scale = grouped.abs().amax(dim=(3, 4)) / quantization_range + _SCALE_EPSILON
    quantized = (
        (grouped / scale[..., None, None])
        .round()
        .clamp(-quantization_range, quantization_range)
        .to(torch.int8)
    )
    return quantized.reshape_as(padded)[:, :, :length], scale


def _quantize_value_per_channel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = value.float().abs().amax(dim=2) / _V_FP8_RANGE + _SCALE_EPSILON
    quantized = (value.float() / scale[:, :, None, :]).to(torch.float8_e4m3fn)
    return quantized, scale


def reference_sage_attention_2pp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    qk_quantization: Literal["per_thread", "per_warp"] = "per_thread",
) -> torch.Tensor:
    """Evaluate a quantized Sage2++ algorithm using ordinary PyTorch operations.

    This intentionally follows the 64-key online-softmax loop. Probability FP8
    quantization depends on the running maximum, so quantizing a fully materialized
    softmax matrix would not be an equivalent reference.
    """
    output_dtype = query.dtype
    quantization_range = 127
    key_float = key.float()
    key_centered = key_float - key_float.mean(dim=2, keepdim=True)
    if qk_quantization == "per_warp":
        query_int8, query_scale = _quantize_per_group(
            query,
            _Q_BLOCK,
            quantization_range,
        )
        key_int8, key_scale = _quantize_per_group(
            key_centered,
            _K_BLOCK,
            quantization_range,
        )
        query_scale = query_scale.repeat_interleave(_Q_BLOCK, dim=2)[:, :, : query.shape[2]]
        key_scale = key_scale.repeat_interleave(_K_BLOCK, dim=2)[:, :, : key.shape[2]]
    elif qk_quantization == "per_thread":
        query_int8, query_scale = _quantize_query_per_thread(query, quantization_range)
        key_int8, key_scale = _quantize_key_per_thread(key_centered, quantization_range)
    else:
        raise ValueError(f"unknown Q/K quantization granularity: {qk_quantization}")
    value_fp8, value_scale = _quantize_value_per_channel(value)

    batch, heads, query_length, width = query.shape
    key_length = key.shape[2]
    accumulator = torch.zeros(
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
    query_positions = torch.arange(query_length, device=query.device)

    for start in range(0, key_length, _PV_BLOCK):
        stop = min(start + _PV_BLOCK, key_length)
        key_block = key_int8[:, :, start:stop]
        integer_scores = torch.matmul(
            query_int8.float(),
            key_block.transpose(-1, -2).float(),
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
        probabilities = torch.exp(scores - next_max[..., None])
        probabilities = torch.nan_to_num(probabilities)

        accumulator *= old_weight[..., None]
        denominator = denominator * old_weight + probabilities.sum(dim=-1)

        probability_fp8 = (probabilities * _P_FP8_RANGE).to(torch.float8_e4m3fn)
        value_block = value_fp8[:, :, start:stop]
        # Rounding the complete 64-wide product to FP16 approximates the two
        # K=32 hardware MMAs that share an FP16 accumulator.
        partial = torch.matmul(probability_fp8.float(), value_block.float())
        partial = partial.to(torch.float16).float()
        accumulator += partial * (value_scale[:, :, None, :] / _P_FP8_RANGE)
        running_max = next_max

    output = accumulator / denominator.clamp_min(1e-30)[..., None]
    return output.to(output_dtype)
