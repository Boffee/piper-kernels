"""Portable reference for the canonical SageAttention2++ 8+8 forward path.

SageAttention2++ originates from the SageAttention project. See the repository
NOTICE for upstream attribution.
"""

import math

import torch

from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    QKQuantizationGranularity,
    quantize_query_key,
)

_PV_BLOCK = 64
_P_FP8_LN_MAX = math.log(448.0)
_V_FP8_MAX = 2.25
_SCALE_EPSILON = 1e-7


def _quantize_value_per_channel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = value.float().abs().amax(dim=2) / _V_FP8_MAX + _SCALE_EPSILON
    quantized = (value.float() / scale[:, :, None, :]).to(torch.float8_e4m3fn)
    return quantized, scale


def reference_sage_attention_2pp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    qk_quantization: QKQuantizationGranularity = "per_thread",
) -> torch.Tensor:
    """Evaluate quantized SageAttention2++ using ordinary PyTorch operations.

    This intentionally follows the 64-key online-softmax loop. Probability FP8
    quantization depends on the running maximum, so quantizing a fully materialized
    softmax matrix would not be an equivalent reference.
    """
    output_dtype = query.dtype
    query_int8, key_int8, query_scale, key_scale = quantize_query_key(
        query,
        key,
        granularity=qk_quantization,
    )
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

        # Canonical SageAttention2++ shifts the online-softmax frame so the largest
        # probability is already in FP8's usable range.
        block_max = scores.amax(dim=-1) - _P_FP8_LN_MAX
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
