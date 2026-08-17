"""Portable reference for Sage-style INT8 Q/K quantization."""

from typing import Literal

import torch

from ._rotation import rotate_signed_hadamard_heads

QUERY_BLOCK = 32
KEY_BLOCK = 64
SCALE_EPSILON = 1e-7
type QKQuantizationGranularity = Literal["per_thread", "per_warp"]


def _pad_sequence(value: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    length = value.shape[2]
    padded_length = (length + multiple - 1) // multiple * multiple
    if padded_length == length:
        return value, length
    padded = value.new_zeros((*value.shape[:2], padded_length, value.shape[3]))
    padded[:, :, :length] = value
    return padded, length


def quantize_query_per_thread(
    query: torch.Tensor,
    quantization_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match SageAttention's four interleaved query rows per scale."""
    padded, length = _pad_sequence(query, QUERY_BLOCK)
    batch, heads, _, width = padded.shape
    grouped = padded.float().reshape(batch, heads, -1, 4, 8, width)
    scale_group = grouped.abs().amax(dim=(3, 5)) / quantization_range + SCALE_EPSILON
    scale = scale_group[:, :, :, None, :, None].expand_as(grouped)
    quantized = (
        (grouped / scale).round().clamp(-quantization_range, quantization_range).to(torch.int8)
    )
    return (
        quantized.reshape_as(padded)[:, :, :length],
        scale[..., 0].reshape(*padded.shape[:3])[:, :, :length],
    )


def quantize_key_per_thread(
    key: torch.Tensor,
    quantization_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match SageAttention's sixteen interleaved key rows per scale."""
    padded, length = _pad_sequence(key, KEY_BLOCK)
    batch, heads, _, width = padded.shape
    grouped = padded.float().reshape(batch, heads, -1, 8, 4, 2, width)
    scale_group = grouped.abs().amax(dim=(3, 5, 6)) / quantization_range + SCALE_EPSILON
    scale = scale_group[:, :, :, None, :, None, None].expand_as(grouped)
    quantized = (
        (grouped / scale).round().clamp(-quantization_range, quantization_range).to(torch.int8)
    )
    return (
        quantized.reshape_as(padded)[:, :, :length],
        scale[..., 0].reshape(*padded.shape[:3])[:, :, :length],
    )


def quantize_per_group(
    value: torch.Tensor,
    group_size: int,
    quantization_range: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize consecutive rows with one scale per group."""
    padded, length = _pad_sequence(value, group_size)
    batch, heads, _, width = padded.shape
    grouped = padded.float().reshape(batch, heads, -1, group_size, width)
    scale = grouped.abs().amax(dim=(3, 4)) / quantization_range + SCALE_EPSILON
    quantized = (
        (grouped / scale[..., None, None])
        .round()
        .clamp(-quantization_range, quantization_range)
        .to(torch.int8)
    )
    return quantized.reshape_as(padded)[:, :, :length], scale


def quantize_query_key(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    granularity: QKQuantizationGranularity,
    quantization_range: int = 127,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center and smooth K, then quantize signed-Hadamard Q/K."""
    key_float = key.float()
    key_centered = key_float - key_float.mean(dim=2, keepdim=True)
    query_float = rotate_signed_hadamard_heads(query.float())
    key_centered = rotate_signed_hadamard_heads(key_centered)
    if granularity == "per_warp":
        query_int8, query_scale = quantize_per_group(
            query_float,
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
    elif granularity == "per_thread":
        query_int8, query_scale = quantize_query_per_thread(query_float, quantization_range)
        key_int8, key_scale = quantize_key_per_thread(key_centered, quantization_range)
    else:
        raise ValueError(f"unknown Q/K quantization granularity: {granularity}")
    return query_int8, key_int8, query_scale, key_scale
