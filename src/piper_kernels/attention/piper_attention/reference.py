"""Portable reference for Piper Attention's key-scaled integer-PV attention.

Piper Attention follows the fused online-softmax structure of FlashAttention
and the INT8 QK smoothing/quantization of SageAttention. Its distinct PV path
uses one signed-INT8 scale per V row and a nonnegative UINT8 probability
operand. The reference is intentionally readable rather than fast.
"""

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
        if chunk > 0:
            # Shift running_max into the new chunk's unbiased coordinate frame once so
            # large shared chunk offsets never enter intra-chunk max/exp normalization.
            running_max = torch.where(
                running_max == -float("inf"),
                running_max,
                running_max + (running_bias - chunk_bias),
            )
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
            next_max = torch.maximum(running_max, block_max)
            old_weight = torch.exp(running_max - next_max)
            current_weight = torch.exp(block_max - next_max)
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
        numerator += (chunk_denom * _P_UINT8_RANGE)[..., None] * value_mean[:, :, chunk : chunk + 1]
        running_bias = chunk_bias

    output = numerator / (denominator.clamp_min(1e-30)[..., None] * _P_UINT8_RANGE)
    return output.reshape(query.shape).to(output_dtype)
