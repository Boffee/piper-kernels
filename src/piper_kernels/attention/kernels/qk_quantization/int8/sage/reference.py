"""Portable reference for Sage-style INT8 Q/K quantization."""

import torch

QUERY_BLOCK = 32
KEY_BLOCK = 64
SCALE_EPSILON = 1e-7


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
