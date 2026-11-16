"""Portable reference for Sage-style INT8 Q/K quantization."""

from typing import Literal

import torch

from ._rotation import rotate_signed_hadamard_heads

QUERY_BLOCK = 32
KEY_BLOCK = 64
SCALE_EPSILON = 1e-7
type QKQuantizationGranularity = Literal["per_thread", "per_warp"]


def validate_qk_quantization(qk_quantization: str) -> QKQuantizationGranularity:
    """Accept a Sage Q/K scale granularity name."""
    if qk_quantization == "per_thread":
        return "per_thread"
    if qk_quantization == "per_warp":
        return "per_warp"
    raise ValueError(f"unknown Q/K quantization granularity: {qk_quantization}")


def count_query_scales(storage_length: int, qk_quantization: QKQuantizationGranularity) -> int:
    """Count FP32 scales per head of Q64-padded Q: one per Q32 group or per row."""
    return storage_length if qk_quantization == "per_thread" else storage_length // QUERY_BLOCK


def count_key_scales(storage_length: int, qk_quantization: QKQuantizationGranularity) -> int:
    """Count FP32 scales per head of K64-padded K: one per K64 group or per key."""
    return storage_length if qk_quantization == "per_thread" else storage_length // KEY_BLOCK


def resolve_qk_quantization(scales: torch.Tensor, storage_length: int) -> QKQuantizationGranularity:
    """Read Q/K granularity from scale metadata: one scale per stored row is per thread."""
    return "per_thread" if scales.ndim == 3 and scales.shape[2] == storage_length else "per_warp"


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


def _quantize_rotated_query_key(
    query_rotated: torch.Tensor,
    key_rotated: torch.Tensor,
    *,
    granularity: QKQuantizationGranularity,
    query_length: int,
    key_length: int,
    quantization_range: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if granularity == "per_warp":
        query_int8, query_scale = quantize_per_group(
            query_rotated,
            QUERY_BLOCK,
            quantization_range,
        )
        key_int8, key_scale = quantize_per_group(
            key_rotated,
            KEY_BLOCK,
            quantization_range,
        )
        query_scale = query_scale.repeat_interleave(QUERY_BLOCK, dim=2)[:, :, :query_length]
        key_scale = key_scale.repeat_interleave(KEY_BLOCK, dim=2)[:, :, :key_length]
    elif granularity == "per_thread":
        query_int8, query_scale = quantize_query_per_thread(query_rotated, quantization_range)
        key_int8, key_scale = quantize_key_per_thread(key_rotated, quantization_range)
    else:
        raise ValueError(f"unknown Q/K quantization granularity: {granularity}")
    return query_int8, key_int8, query_scale, key_scale


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
    return _quantize_rotated_query_key(
        rotate_signed_hadamard_heads(query.float()),
        rotate_signed_hadamard_heads(key_centered),
        granularity=granularity,
        query_length=query.shape[2],
        key_length=key.shape[2],
        quantization_range=quantization_range,
    )


def encode_residual_int8(
    values: torch.Tensor,
    *,
    quantization_range: int = 127,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode an FP32 vector as a two-term INT8 residual expansion along dim=-1."""
    values_float = values.float()
    scale_0 = values_float.abs().amax(dim=-1, keepdim=True) / quantization_range + SCALE_EPSILON
    quantized_0 = (
        (values_float / scale_0)
        .round()
        .clamp(-quantization_range, quantization_range)
        .to(torch.int8)
    )
    residual = values_float - quantized_0.float() * scale_0
    scale_1 = residual.abs().amax(dim=-1, keepdim=True) / quantization_range + SCALE_EPSILON
    quantized_1 = (
        (residual / scale_1).round().clamp(-quantization_range, quantization_range).to(torch.int8)
    )
    return torch.stack((quantized_0, quantized_1), dim=-2), torch.cat((scale_0, scale_1), dim=-1)


def quantize_tile_centered_query_key(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    tile_rows: int,
    granularity: QKQuantizationGranularity,
    valid_key_rows: torch.Tensor | None = None,
    quantization_range: int = 127,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tile-center and smooth K, returning Q/K INT8 codes, scales, and 2-term H(K_mean)."""
    batch, kv_heads, key_length, head_dim = key.shape
    padded_key, _ = _pad_sequence(key.float(), tile_rows)
    tile_count = padded_key.shape[2] // tile_rows
    key_tiles = padded_key.reshape(batch, kv_heads, tile_count, tile_rows, head_dim)
    if valid_key_rows is None:
        valid = torch.ones((tile_count, tile_rows), device=key.device, dtype=torch.bool)
        if padded_key.shape[2] != key_length:
            valid.view(-1)[key_length:] = False
        valid_mask = valid[None, None, :, :, None]
    else:
        valid_mask = valid_key_rows.reshape(1, 1, tile_count, tile_rows, 1).bool()
    valid_key_tiles = torch.where(valid_mask, key_tiles, 0.0)
    valid_count = valid_mask.sum(dim=3, keepdim=True).clamp_min(1.0)
    key_mean = valid_key_tiles.sum(dim=3, keepdim=True) / valid_count
    key_centered = torch.where(valid_mask, valid_key_tiles - key_mean, 0.0).reshape_as(padded_key)[
        :, :, :key_length
    ]
    key_mean_rotated = rotate_signed_hadamard_heads(key_mean.squeeze(3))
    key_center, key_center_scale = encode_residual_int8(
        key_mean_rotated,
        quantization_range=quantization_range,
    )

    query_int8, key_int8, query_scale, key_scale = _quantize_rotated_query_key(
        rotate_signed_hadamard_heads(query.float()),
        rotate_signed_hadamard_heads(key_centered),
        granularity=granularity,
        query_length=query.shape[2],
        key_length=key_length,
        quantization_range=quantization_range,
    )
    return query_int8, key_int8, query_scale, key_scale, key_center, key_center_scale
