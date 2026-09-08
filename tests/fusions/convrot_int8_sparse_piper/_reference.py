"""Composed FP32 and independent FP64 references for sparse-Piper projection tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import SIGNED_HADAMARD_MASK
from piper_kernels.fusions.convrot_int8_sparse_piper._layout import padded_sequence_length
from piper_kernels.linear.convrot.int8 import _ops as int8_ops

_BLOCK_ROWS = 64
_HEAD_DIM = 128


def _rotate_fp64(values):
    offsets = torch.arange(_HEAD_DIM, device=values.device)
    words = torch.tensor(SIGNED_HADAMARD_MASK, device=values.device, dtype=torch.int64)
    result = values * (2 * ((words[offsets // 32] >> (offsets % 32)) & 1) - 1)
    for distance in (1, 2, 4, 8, 16, 32, 64):
        low, high = result.reshape(*result.shape[:-1], -1, 2, distance).unbind(-2)
        result = torch.stack((low + high, low - high), dim=-2).flatten(-3)
    return result / math.sqrt(_HEAD_DIM)


def assert_int8_codes_close(actual, expected):
    """Allow at most one INT8 code of rounding difference."""
    assert int((actual.short() - expected.short()).abs().max()) <= 1


def _encode_fp64(values, rows):
    grouped = values.reshape(-1, rows, _HEAD_DIM)
    scale = grouped.abs().amax((-1, -2)) / 127 + 1e-7
    normalized = grouped / scale[:, None, None]
    codes = torch.trunc(normalized + 0.5 * normalized.sign()).clamp(-127, 127).to(torch.int8)
    return codes.reshape_as(values), scale


def check_qk_sample_fp64(projected, output, head, start, rows, norm, cos, sin, *, is_query):
    """Check one minmax Q/K block with H3 normalization and softmax constants."""
    rotary = cos.shape[1]
    projected *= torch.rsqrt(projected.square().mean(-1, keepdim=True) + 1e-5)
    projected *= norm.double()
    first, second = projected[:, :rotary].chunk(2, dim=-1)
    rotated = torch.cat((-second, first), -1)
    projected[:, :rotary] = (
        projected[:, :rotary] * cos[start : start + rows].double()
        + rotated * sin[start : start + rows].double()
    )
    maximum, minimum = projected.amax(0), projected.amin(0)
    summary = maximum + minimum if is_query else maximum
    torch.testing.assert_close(
        output[2][0, head, start // _BLOCK_ROWS].double(), summary, rtol=3e-5, atol=3e-5
    )
    if not is_query:
        torch.testing.assert_close(
            output[3][0, head, start // _BLOCK_ROWS].double(), minimum, rtol=3e-5, atol=3e-5
        )
    padded = projected.new_zeros((_BLOCK_ROWS, _HEAD_DIM), dtype=torch.float64)
    padded[:rows] = _rotate_fp64(projected)
    scale_rows = 32 if is_query else _BLOCK_ROWS
    codes, expected_scale = _encode_fp64(padded, scale_rows)
    actual = output[0][0, head, start : start + _BLOCK_ROWS]
    assert_int8_codes_close(actual, codes)
    if is_query:
        expected_scale *= _HEAD_DIM**-0.5 * math.log2(math.e)
        if rows <= scale_rows:
            expected_scale[1] = 0
    offset = start // scale_rows
    torch.testing.assert_close(
        output[1][0, head, offset : offset + expected_scale.numel()].double(),
        expected_scale,
        rtol=3e-5,
        atol=1e-7,
    )


def check_value_sample_fp64(projected, output, head, start, rows, mean_v):
    """Check a centered V64 block, scale multiplier, and uncentered block mean."""
    torch.testing.assert_close(
        output[3][0, head, start // _BLOCK_ROWS].double(),
        projected.mean(0),
        rtol=2e-5,
        atol=2e-4,
    )
    centered = projected.new_zeros((_BLOCK_ROWS, _HEAD_DIM), dtype=torch.float64)
    centered[:rows] = projected - mean_v
    codes, expected_scale = _encode_fp64(centered, _BLOCK_ROWS)
    assert_int8_codes_close(output[0][0, head, :, start : start + _BLOCK_ROWS].T, codes)
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
    key_max: torch.Tensor
    key_min: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectedValue:
    value: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_mean: torch.Tensor
    block_mean: torch.Tensor


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
) -> torch.Tensor:
    batch, sequence_length, _input_features = input_qdata.shape
    heads = weight_qdata.shape[0] // _HEAD_DIM
    projected = int8_ops.linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        None,
        torch.float32,
    ).view(batch, sequence_length, heads, _HEAD_DIM)
    normalized = F.rms_norm(projected, (_HEAD_DIM,), norm_weight.float(), norm_epsilon)
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
) -> ProjectedKey:
    """Materialize the FP32 operations fused by one-pass key projection."""
    batch, sequence_length, _input_features = input_qdata.shape
    heads = weight_qdata.shape[0] // _HEAD_DIM
    key = _materialized_fp32_qk(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
    )
    storage_length = padded_sequence_length(sequence_length)
    key_int8, key_scale = qk_quantization.prepare_key(
        key,
        torch.zeros((batch, heads, _HEAD_DIM), device=key.device, dtype=torch.float32),
        grouped=True,
        storage_key_length=storage_length,
    )
    blocks, valid = _padded_blocks(key.float())
    key_max = blocks.masked_fill(~valid[None, None, :, :, None], -torch.inf).amax(dim=3)
    key_min = blocks.masked_fill(~valid[None, None, :, :, None], torch.inf).amin(dim=3)
    return ProjectedKey(key_int8, key_scale, key_max, key_min)


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
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
) -> ProjectedValue:
    """Materialize the FP32 operations fused by one-pass value projection."""
    batch, sequence_length, _input_features = input_qdata.shape
    heads = weight_qdata.shape[0] // _HEAD_DIM
    value_mean = ((input_mean @ weight_qdata.float().T) * weight_scale[:, 0]).view(
        batch,
        heads,
        _HEAD_DIM,
    )
    projected = int8_ops.linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        None,
        torch.float32,
    ).view(batch, sequence_length, heads, _HEAD_DIM)
    projected_blocks, valid = _padded_blocks(projected.permute(0, 2, 1, 3).float())
    block_mean = (projected_blocks * valid[None, None, :, :, None]).sum(dim=3) / valid.sum(dim=1)[
        None, None, :, None
    ]
    storage_length = padded_sequence_length(sequence_length)
    centered = projected.new_zeros((batch, heads, storage_length, _HEAD_DIM))
    centered[:, :, :sequence_length] = (
        projected.permute(0, 2, 1, 3).float() - value_mean[:, :, None, :]
    )
    centered = centered.unflatten(
        2,
        (storage_length // _BLOCK_ROWS, _BLOCK_ROWS),
    )
    value_scale = centered.abs().amax(dim=(-1, -2)) / 127.0 + 1e-7
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
        block_mean,
    )
