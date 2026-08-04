"""Portable reference for experimental ConvRot integer attention.

This is intentionally not part of the public attention API yet.  It tests the
algorithmic premise needed by an integer-only hot loop: ConvRot makes one K
scale per head accurate enough for raw INT32 scores to remain comparable across
all key tiles.
"""

import math

import torch

from piper_kernels.convrot._rotation import rotate_groups

_KV_BLOCK = 64
_INT8_MAX = 127
_PROBABILITY_MAX = 255
_RECURRENCE_BITS = 15
_RECURRENCE_ONE = 1 << _RECURRENCE_BITS
_SCALE_BITS = 20
_SCALE_ONE = 1 << _SCALE_BITS
_SCALE_EPSILON = 1e-30
_NEGATIVE_INFINITY = -(1 << 30)

# Quadratic approximation of 2**(-x) for x in [0, 1].  It matches both
# endpoints and the derivative at zero and is evaluated in Q15.
_EXP_A = -round(math.log(2.0) * _RECURRENCE_ONE)
_EXP_B = round((math.log(2.0) - 0.5) * _RECURRENCE_ONE)


def _quantize_query_per_row(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use one Q scale per row; it is constant across every score in that row."""
    scale = query.float().abs().amax(dim=3, keepdim=True).clamp_min(_SCALE_EPSILON)
    scale /= _INT8_MAX
    quantized = (query.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(3)


def _quantize_key_per_channel(key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K around each channel midpoint and fold its scales into Q."""
    key_float = key.float()
    minimum = key_float.amin(dim=2, keepdim=True)
    maximum = key_float.amax(dim=2, keepdim=True)
    midpoint = (minimum + maximum) * 0.5
    scale = ((maximum - minimum) / (2 * _INT8_MAX)).clamp_min(_SCALE_EPSILON)
    quantized = ((key_float - midpoint) / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(2)


def _quantize_value_per_channel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep one V scale per output channel; it factors out after integer PV."""
    scale = value.float().abs().amax(dim=2).clamp_min(_SCALE_EPSILON) / _INT8_MAX
    quantized = (value.float() / scale[:, :, None, :]).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def _expand_trailing(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def _integer_exp2(
    negative_delta: torch.Tensor,
    log2_scale: torch.Tensor,
    output_maximum: int,
) -> torch.Tensor:
    """Approximate ``round(output_maximum * 2**(delta * scale))`` with integers.

    ``negative_delta`` is an INT32/INT64 score difference no greater than zero.
    A Q20 scale retains enough resolution for the small score scales produced by
    D=64/128 attention while keeping the eventual Triton product in INT32 range.
    """
    scale_multiplier = _expand_trailing(
        (log2_scale * _SCALE_ONE).round().to(torch.int64),
        negative_delta,
    )
    magnitude = (-negative_delta.to(torch.int64) * scale_multiplier).clamp_min(0)
    whole = (magnitude >> _SCALE_BITS).clamp(0, 30)
    fraction = (magnitude & (_SCALE_ONE - 1)) >> (_SCALE_BITS - _RECURRENCE_BITS)

    polynomial = _EXP_B
    polynomial = _EXP_A + ((polynomial * fraction + (_RECURRENCE_ONE // 2)) >> _RECURRENCE_BITS)
    polynomial = _RECURRENCE_ONE + (
        (polynomial * fraction + (_RECURRENCE_ONE // 2)) >> _RECURRENCE_BITS
    )

    shift = whole + _RECURRENCE_BITS
    rounding = 1 << (shift - 1)
    return ((polynomial * output_maximum + rounding) >> shift).clamp(0, output_maximum)


def _integer_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Evaluate small integer reductions exactly on CPU or consumer CUDA GPUs."""
    if left.device.type == "cpu":
        return torch.matmul(left.to(torch.int32), right.to(torch.int32)).to(torch.int64)
    # Float32 exactly represents these D<=128 and block-K<=64 INT8 reductions.
    return torch.matmul(left.float(), right.float()).to(torch.int64)


def reference_int8_convrot_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int = 64,
) -> torch.Tensor:
    """Evaluate a ConvRot INT8 QK, integer-softmax, INT8 PV attention path.

    The online loop keeps scores, probabilities, its denominator, and its PV
    accumulator in integer formats.  Only the final normalized output is
    converted to floating point and multiplied by the per-channel V scale.
    """
    if query.shape[-1] % rotation_group:
        raise ValueError(
            f"head dimension {query.shape[-1]} must be divisible by rotation group {rotation_group}"
        )
    if is_causal and query.shape[2] != key.shape[2]:
        raise ValueError("causal integer attention requires equal query and key lengths")

    query_rotated = rotate_groups(query.float(), rotation_group)
    key_rotated = rotate_groups(key.float(), rotation_group)
    key_int8, key_scale = _quantize_key_per_channel(key_rotated)
    # Folding K's per-channel dequantization scales into Q keeps every score in
    # a query row on one scale while avoiding a lossy global K scale.
    query_int8, query_scale = _quantize_query_per_row(query_rotated * key_scale[:, :, None, :])
    value_int8, value_scale = _quantize_value_per_channel(value)

    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    accumulator = torch.zeros(
        (batch, heads, query_length, head_dim),
        device=query.device,
        dtype=torch.int64,
    )
    denominator = torch.zeros(
        (batch, heads, query_length),
        device=query.device,
        dtype=torch.int64,
    )
    running_max = torch.full_like(denominator, _NEGATIVE_INFINITY)
    log2_scale = query_scale * (scale * math.log2(math.e))
    query_positions = torch.arange(query_length, device=query.device)

    for start in range(0, key_length, _KV_BLOCK):
        stop = min(start + _KV_BLOCK, key_length)
        integer_scores = _integer_matmul(
            query_int8,
            key_int8[:, :, start:stop].transpose(-1, -2),
        )
        if is_causal:
            key_positions = torch.arange(start, stop, device=query.device)
            integer_scores = integer_scores.masked_fill(
                key_positions[None, None, None, :] > query_positions[None, None, :, None],
                _NEGATIVE_INFINITY,
            )

        next_max = torch.maximum(running_max, integer_scores.amax(dim=-1))
        probabilities = _integer_exp2(
            integer_scores - next_max[..., None],
            log2_scale,
            _PROBABILITY_MAX,
        )
        old_weight = _integer_exp2(
            running_max - next_max,
            log2_scale,
            _RECURRENCE_ONE - 1,
        )

        denominator = (denominator * old_weight + (_RECURRENCE_ONE // 2)) >> _RECURRENCE_BITS
        denominator += probabilities.sum(dim=-1)
        accumulator = (
            accumulator * old_weight[..., None] + (_RECURRENCE_ONE // 2)
        ) >> _RECURRENCE_BITS
        accumulator += _integer_matmul(
            probabilities,
            value_int8[:, :, start:stop],
        )
        running_max = next_max

    output = accumulator.float() / denominator.clamp_min(1)[..., None]
    output *= value_scale[:, :, None, :]
    return output.to(query.dtype)
