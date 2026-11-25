"""Portable PyTorch reference implementations for ConvRot Trellis linear."""

from __future__ import annotations

import math

import torch

from piper_kernels.linear.convrot.int8 import reference as int8_reference
from piper_kernels.trellis._layout import TrellisLayout
from piper_kernels.trellis._reference import decode_reference, decode_variable_rows_reference
from piper_kernels.weights.convrot._rotation import rotate_groups


def _decode_qdata(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor | None,
    block_size: int,
) -> torch.Tensor:
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    if row_widths is None:
        layout = TrellisLayout.from_storage(block_size, bitstreams.shape[2])
        return decode_reference(bitstreams, block_scales, layout).reshape(out_features, in_features)
    return decode_variable_rows_reference(bitstreams, block_scales, row_widths, block_size)


def _empty_result(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)
    if rows == 0 or out_features == 0:
        return input.new_empty((*leading_shape, out_features))
    if in_features == 0:
        if bias is None:
            return input.new_zeros((*leading_shape, out_features))
        return bias.to(input.dtype).expand(*leading_shape, out_features).clone()
    return None


def linear_a16(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Portable FP32-rotated A16 reference for ConvRot Trellis linear."""
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    empty = _empty_result(input, out_features, in_features, bias)
    if empty is not None:
        return empty

    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)
    with torch.amp.autocast(input.device.type, enabled=False):
        qdata = _decode_qdata(bitstreams, block_scales, row_widths, block_size)
        rotated = rotate_groups(input.reshape(rows, in_features).float(), group_size)
        accumulated = torch.matmul(rotated, qdata.float().T)
        accumulated.mul_(scale.reshape(1, out_features).to(torch.float32))
        if bias is not None:
            accumulated.add_(bias.to(torch.float32))
    return accumulated.to(input.dtype).reshape(*leading_shape, out_features)


def linear_w8a8(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Portable W8A8 reference matching decoded ``ConvRotInt8Tensor`` linear."""
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    empty = _empty_result(input, out_features, in_features, bias)
    if empty is not None:
        return empty
    with torch.amp.autocast(input.device.type, enabled=False):
        qdata = _decode_qdata(bitstreams, block_scales, row_widths, block_size)
        return int8_reference.linear(input, qdata, scale, group_size, bias)


def linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Portable CPU/fallback reference for ConvRot Trellis linear."""
    return linear_a16(
        input,
        bitstreams,
        block_scales,
        scale,
        row_widths,
        bias,
        block_size,
        group_size,
    )


__all__ = ["linear", "linear_a16", "linear_w8a8"]
