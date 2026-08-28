"""Materialized FP32 references for ConvRot-to-sparse-Piper fusion tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

_BLOCK_ROWS = 64
_HEAD_DIM = 128


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
    projected = convrot_backend.linear_prepared(
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
    blocks = query.unflatten(2, (sequence_length // _BLOCK_ROWS, _BLOCK_ROWS)).float()
    summary = blocks.amax(dim=3) + blocks.amin(dim=3)
    query_int8, query_scale = qk_quantization.prepare_query(
        query,
        softmax_scale,
        grouped=True,
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
    key_int8, key_scale = qk_quantization.prepare_key(
        key,
        torch.zeros((batch, heads, _HEAD_DIM), device=key.device, dtype=torch.float32),
        grouped=True,
        storage_key_length=sequence_length,
    )
    blocks = key.unflatten(2, (sequence_length // _BLOCK_ROWS, _BLOCK_ROWS)).float()
    key_max = blocks.amax(dim=3)
    key_min = blocks.amin(dim=3)
    return ProjectedKey(key_int8, key_scale, key_max, key_min)


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
    projected = convrot_backend.linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        None,
        torch.float32,
    ).view(batch, sequence_length, heads, _HEAD_DIM)
    value = projected.permute(0, 2, 1, 3).unflatten(
        2,
        (sequence_length // _BLOCK_ROWS, _BLOCK_ROWS),
    )
    centered = value.float() - value_mean[:, :, None, None, :]
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
    )
