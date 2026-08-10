"""Portable reference for Piper Attention's key-scaled integer-PV attention.

Piper Attention follows the fused online-softmax structure of FlashAttention
and the INT8 QK smoothing/quantization of SageAttention. Its distinct PV path
uses one signed-INT8 scale per V row and a nonnegative UINT8 probability
operand. The reference is intentionally readable rather than fast.
"""

from typing import Literal

import torch

from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    KEY_BLOCK,
    QUERY_BLOCK,
    quantize_key_per_thread,
    quantize_per_group,
    quantize_query_per_thread,
)

_PV_BLOCK = 64
_P_UINT8_RANGE = 255.0
_V_INT8_RANGE = 127.0
_SCALE_EPSILON = 1e-7


def _quantize_value_per_key(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = value.float().abs().amax(dim=-1) / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = (
        (value.float() / scale[..., None])
        .round()
        .clamp(-_V_INT8_RANGE, _V_INT8_RANGE)
        .to(torch.int8)
    )
    return quantized, scale


def reference_piper_attention(  # noqa: PLR0915
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    qk_quantization: Literal["per_thread", "per_warp"] = "per_thread",
    sort_value_rows: bool = False,
) -> torch.Tensor:
    """Evaluate Piper Attention with ordinary PyTorch operations."""
    if sort_value_rows and is_causal:
        raise ValueError("value-row ordering is valid only for non-causal attention")

    output_dtype = query.dtype
    quantization_range = 127
    key_float = key.float()
    key_centered = key_float - key_float.mean(dim=2, keepdim=True)
    value_float = value.float()
    value_mean = value_float.mean(dim=2, keepdim=True)
    value_centered = value_float - value_mean

    if sort_value_rows:
        order = torch.argsort(value_centered.abs().amax(dim=-1), dim=-1, stable=True)
        order_4d = order[..., None].expand_as(key_centered)
        key_centered = torch.gather(key_centered, 2, order_4d)
        value_centered = torch.gather(value_centered, 2, order_4d)

    if qk_quantization == "per_warp":
        query_int8, query_scale = quantize_per_group(
            query,
            QUERY_BLOCK,
            quantization_range,
        )
        key_int8, key_scale = quantize_per_group(
            key_centered,
            KEY_BLOCK,
            quantization_range,
        )
        query_scale = query_scale.repeat_interleave(QUERY_BLOCK, dim=2)[:, :, : query.shape[2]]
        key_scale = key_scale.repeat_interleave(KEY_BLOCK, dim=2)[:, :, : key.shape[2]]
    elif qk_quantization == "per_thread":
        query_int8, query_scale = quantize_query_per_thread(query, quantization_range)
        key_int8, key_scale = quantize_key_per_thread(key_centered, quantization_range)
    else:
        raise ValueError(f"unknown Q/K quantization granularity: {qk_quantization}")
    value_int8, value_scale = _quantize_value_per_key(value_centered)

    batch, heads, query_length, width = query.shape
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
    query_positions = torch.arange(query_length, device=query.device)

    for start in range(0, key_length, _PV_BLOCK):
        stop = min(start + _PV_BLOCK, key_length)
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
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        current_weight = torch.exp(block_max - next_max)
        probabilities = torch.exp(scores - block_max[..., None])
        probabilities = torch.nan_to_num(probabilities)

        numerator *= old_weight[..., None]
        denominator = (
            denominator * old_weight
            + probabilities.sum(dim=-1) * current_weight
        )
        probability_uint8 = (
            probabilities
            * block_value_scale[:, :, None, :]
            * _P_UINT8_RANGE
        ).round().clamp(0, _P_UINT8_RANGE)
        partial = torch.matmul(
            probability_uint8,
            value_int8[:, :, start:stop].float(),
        )
        numerator += partial * (current_weight[..., None] / _P_UINT8_RANGE)
        running_max = next_max

    output = numerator / denominator.clamp_min(1e-30)[..., None]
    output += value_mean
    return output.to(output_dtype)
