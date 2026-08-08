"""Portable reference for the canonical SageAttention2++ 8+8 forward path.

SageAttention2++ originates from the SageAttention project. See the repository
NOTICE for upstream attribution.
"""

import math
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
_P_FP8_LN_RANGE = math.log(448.0)
_V_FP8_RANGE = 2.25
_SCALE_EPSILON = 1e-7


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

        # Canonical Sage2++ shifts the online-softmax frame so the largest
        # probability is already in FP8's usable range.
        block_max = scores.amax(dim=-1) - _P_FP8_LN_RANGE
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        probabilities = torch.exp(scores - next_max[..., None])
        probabilities = torch.nan_to_num(probabilities)

        accumulator *= old_weight[..., None]
        denominator = denominator * old_weight + probabilities.sum(dim=-1)

        probability_fp8 = probabilities.to(torch.float8_e4m3fn)
        value_block = value_fp8[:, :, start:stop]
        # Rounding the complete 64-wide product to FP16 approximates the two
        # K=32 hardware MMAs that share an FP16 accumulator.
        partial = torch.matmul(probability_fp8.float(), value_block.float())
        partial = partial.to(torch.float16).float()
        accumulator += partial * value_scale[:, :, None, :]
        running_max = next_max

    output = accumulator / denominator.clamp_min(1e-30)[..., None]
    return output.to(output_dtype)
