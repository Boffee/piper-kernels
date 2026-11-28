"""Composed FP32 and independent FP64 references for sparse-Piper projection tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from _accelerators import ensure_convrot_int8_linear_ops
from torch.nn import functional as F  # noqa: N812

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import SIGNED_HADAMARD_MASK
from piper_kernels.attention.kernels.sparse_piper.layout import padded_sequence_length

_BLOCK_ROWS = 64


def _rotate_fp64(values):
    head_dim = values.shape[-1]
    offsets = torch.arange(head_dim, device=values.device)
    words = torch.tensor(SIGNED_HADAMARD_MASK, device=values.device, dtype=torch.int64)
    result = values * (2 * ((words[offsets // 32] >> (offsets % 32)) & 1) - 1)
    for level in range(head_dim.bit_length() - 1):
        distance = 1 << level
        low, high = result.reshape(*result.shape[:-1], -1, 2, distance).unbind(-2)
        result = torch.stack((low + high, low - high), dim=-2).flatten(-3)
    return result / math.sqrt(head_dim)


def assert_int8_codes_close(actual, expected):
    """Allow at most one INT8 code of rounding difference."""
    assert int((actual.short() - expected.short()).abs().max()) <= 1


def _encode_fp64(values, rows):
    grouped = values.reshape(-1, rows, values.shape[-1])
    scale = grouped.abs().amax((-1, -2)) / 127 + 1e-7
    normalized = grouped / scale[:, None, None]
    codes = torch.trunc(normalized + 0.5 * normalized.sign()).clamp(-127, 127).to(torch.int8)
    return codes.reshape_as(values), scale


def _transform_qk_fp64(projected, norm, cos, sin):
    """Apply RMSNorm and RoPE independently in FP64, before storage rounding."""
    rotary = cos.shape[1]
    projected *= torch.rsqrt(projected.square().mean(-1, keepdim=True) + 1e-5)
    if norm is not None:
        projected *= norm.double()
    first, second = projected[:, :rotary].chunk(2, dim=-1)
    rotated = torch.cat((-second, first), -1)
    projected[:, :rotary] = projected[:, :rotary] * cos.double() + rotated * sin.double()
    return projected


def reconstruct_key_center(
    key_center: torch.Tensor,
    key_center_scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode a 2-term residual INT8 rotated K64 center tensor to floating point."""
    return key_center[..., 0, :].to(dtype) * key_center_scale[..., 0:1].to(dtype) + key_center[
        ..., 1, :
    ].to(dtype) * key_center_scale[..., 1:2].to(dtype)


def check_qk_sample_fp64(projected, output, head, start, rows, norm, cos, sin, *, is_query):
    """Check raw summaries and Q32 or tile-centered K64 encoding."""
    head_dim = projected.shape[-1]
    projected = _transform_qk_fp64(
        projected, norm, cos[start : start + rows], sin[start : start + rows]
    )
    maximum, minimum = projected.amax(0), projected.amin(0)
    summary = maximum + minimum if is_query else maximum
    summary_index = 2 if is_query else 4
    torch.testing.assert_close(
        output[summary_index][0, head, start // _BLOCK_ROWS].double(), summary, rtol=3e-5, atol=3e-5
    )
    if not is_query:
        torch.testing.assert_close(
            output[5][0, head, start // _BLOCK_ROWS].double(), minimum, rtol=3e-5, atol=3e-5
        )
    rotated = _rotate_fp64(projected)
    if not is_query:
        rotated_center = 0.5 * (rotated.amax(0) + rotated.amin(0))
        tile = start // _BLOCK_ROWS
        reconstructed_center = reconstruct_key_center(
            output[2][0, head, tile],
            output[3][0, head, tile],
            dtype=torch.float64,
        )
        torch.testing.assert_close(reconstructed_center, rotated_center, rtol=2e-4, atol=2e-4)
        rotated = rotated - rotated_center
    padded = rotated.new_zeros((_BLOCK_ROWS, head_dim), dtype=torch.float64)
    padded[:rows] = rotated
    scale_rows = 32 if is_query else _BLOCK_ROWS
    codes, expected_scale = _encode_fp64(padded, scale_rows)
    actual = output[0][0, head, start : start + _BLOCK_ROWS]
    assert_int8_codes_close(actual, codes)
    if is_query:
        expected_scale *= head_dim**-0.5 * math.log2(math.e)
        if rows <= scale_rows:
            expected_scale[1] = 0
    offset = start // scale_rows
    torch.testing.assert_close(
        output[1][0, head, offset : offset + expected_scale.numel()].double(),
        expected_scale,
        rtol=3e-5,
        atol=1e-7,
    )


def check_value_sample_fp64(projected, output, head, start, rows):
    """Check a tile-centered V64 block, scale multiplier, and BF16 tile mean."""
    value_mean = projected.mean(0).bfloat16().double()
    torch.testing.assert_close(
        output[2][0, head, start // _BLOCK_ROWS].double(),
        value_mean,
        rtol=1.6e-2,
        atol=2e-4,
    )
    centered = projected.new_zeros((_BLOCK_ROWS, projected.shape[-1]), dtype=torch.float64)
    centered[:rows] = projected - value_mean
    codes, expected_scale = _encode_fp64(centered, _BLOCK_ROWS)
    # AMD tests pass final [B,H,tiles,D,64] storage. Decode one selected tile
    # with bit axes rather than repeating the device writer's offset formula.
    stored = output[0][0, head, start // _BLOCK_ROWS]
    actual = stored.reshape(-1, 2, 2, 2, 8).transpose(-3, -2).reshape(-1, 64).T
    assert_int8_codes_close(actual, codes)
    torch.testing.assert_close(
        output[1][0, head, start // _BLOCK_ROWS, 0].double(),
        expected_scale[0] * 255,
        rtol=3e-5,
        atol=1e-5,
    )


@dataclass(frozen=True, slots=True)
class ProjectedQuery:
    query: torch.Tensor
    query_scale: torch.Tensor
    query_summary: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectedKey:
    key: torch.Tensor
    key_scale: torch.Tensor
    key_center: torch.Tensor
    key_center_scale: torch.Tensor
    key_max: torch.Tensor
    key_min: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectedValue:
    value: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_mean: torch.Tensor


def _padded_blocks(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K64-padded sequence blocks and their logical-row mask."""
    sequence_length = value.shape[2]
    storage_length = padded_sequence_length(sequence_length)
    padded = value.new_zeros((*value.shape[:2], storage_length, value.shape[3]))
    padded[:, :, :sequence_length] = value
    blocks = padded.unflatten(2, (storage_length // _BLOCK_ROWS, _BLOCK_ROWS))
    valid = torch.arange(storage_length, device=value.device) < sequence_length
    return blocks, valid.unflatten(0, (storage_length // _BLOCK_ROWS, _BLOCK_ROWS))


def _materialized_fp32_qk(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, sequence_length, _input_features = input_qdata.shape
    head_dim = norm_weight.shape[0]
    heads = weight_qdata.shape[0] // head_dim
    ensure_convrot_int8_linear_ops()
    projected = torch.ops.piper_kernels.convrot_int8_linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        torch.float32,
    ).view(batch, sequence_length, heads, head_dim)
    normalized = F.rms_norm(projected, (head_dim,), norm_weight.float(), norm_epsilon)
    rotary_dim = cos.shape[1]
    rotary = normalized[..., :rotary_dim]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    rotary = rotary * cos[None, :, None, :] + rotated * sin[None, :, None, :]
    return torch.cat((rotary, normalized[..., rotary_dim:]), dim=-1).transpose(1, 2).contiguous()


def composed_query_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
    softmax_scale: float,
    bias: torch.Tensor | None = None,
) -> ProjectedQuery:
    """Materialize the FP32 operations fused by one-pass query projection."""
    sequence_length = input_qdata.shape[1]
    query = _materialized_fp32_qk(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        bias=bias,
    )
    storage_length = padded_sequence_length(sequence_length)
    blocks, valid = _padded_blocks(query.float())
    summary = blocks.masked_fill(~valid[None, None, :, :, None], -torch.inf).amax(
        dim=3
    ) + blocks.masked_fill(~valid[None, None, :, :, None], torch.inf).amin(dim=3)
    query_int8, query_scale = qk_quantization.prepare_query(
        query,
        softmax_scale,
        grouped=True,
        storage_query_length=storage_length,
    )
    return ProjectedQuery(query_int8, query_scale, summary)


def composed_key_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
    bias: torch.Tensor | None = None,
    block_lengths: torch.Tensor | None = None,
) -> ProjectedKey:
    """Compose FP32 transforms, K64 tile-centered INT8 quantization, and raw summaries."""
    sequence_length = input_qdata.shape[1]
    key = _materialized_fp32_qk(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        bias=bias,
    )
    storage_length = padded_sequence_length(sequence_length)
    blocks, valid = _padded_blocks(key)
    if block_lengths is not None:
        valid &= torch.arange(_BLOCK_ROWS, device=key.device)[None, :] < block_lengths[:, None]
    masked = blocks.masked_fill(~valid[None, None, :, :, None], 0).flatten(2, 3)
    masked = masked[:, :, :sequence_length].contiguous()
    key_int8, key_scale, key_center, key_center_scale = qk_quantization.prepare_tile_centered_key(
        masked,
        storage_key_length=storage_length,
        block_lengths=block_lengths,
    )
    key_max = blocks.masked_fill(~valid[None, None, :, :, None], -torch.inf).amax(dim=3)
    key_min = blocks.masked_fill(~valid[None, None, :, :, None], torch.inf).amin(dim=3)
    return ProjectedKey(key_int8, key_scale, key_center, key_center_scale, key_max, key_min)


def composed_mean_pool_summary(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
) -> torch.Tensor:
    """Materialize the exact FP32 valid-prefix Q64/K64 means."""
    projected = _materialized_fp32_qk(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
    )
    blocks, valid = _padded_blocks(projected.float())
    lengths = valid.sum(dim=1)
    return (blocks * valid[None, None, :, :, None]).sum(dim=3) / lengths[None, None, :, None]


def composed_value_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    head_dim: int = 128,
    *,
    bias: torch.Tensor | None = None,
    block_lengths: torch.Tensor | None = None,
) -> ProjectedValue:
    """Materialize the FP32 operations fused by K64 tile-centered value projection."""
    batch, sequence_length, _input_features = input_qdata.shape
    heads = weight_qdata.shape[0] // head_dim
    ensure_convrot_int8_linear_ops()
    projected = torch.ops.piper_kernels.convrot_int8_linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        torch.float32,
    ).view(batch, sequence_length, heads, head_dim)
    projected_blocks, valid = _padded_blocks(projected.permute(0, 2, 1, 3).float())
    if block_lengths is not None:
        valid &= (
            torch.arange(_BLOCK_ROWS, device=projected.device)[None, :] < block_lengths[:, None]
        )
    lengths = valid.sum(dim=1).clamp(min=1)
    value_mean = (
        (projected_blocks * valid[None, None, :, :, None]).sum(dim=3) / lengths[None, None, :, None]
    ).bfloat16()
    centered = (projected_blocks - value_mean.float()[:, :, :, None, :]) * valid[
        None, None, :, :, None
    ]
    value_scale = torch.maximum(
        centered.abs().amax(dim=(-1, -2)) / 127.0 + 1e-7,
        value_mean.float().abs().amax(dim=-1) * (2.0**-80),
    )
    normalized = centered / value_scale[..., None, None]
    quantized = (
        torch.trunc(normalized + 0.5 * torch.where(normalized >= 0, 1.0, -1.0))
        .clamp(-127, 127)
        .to(torch.int8)
    )
    return ProjectedValue(
        quantized.flatten(2, 3).permute(0, 1, 3, 2).contiguous(),
        (value_scale * 255.0).unsqueeze(-1),
        value_mean,
    )
