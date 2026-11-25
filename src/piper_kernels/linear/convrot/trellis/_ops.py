"""Stable ConvRot Trellis linear custom-op schema and runtime selection."""

from __future__ import annotations

import math

import torch

from piper_kernels._triton import runtime


@torch.library.custom_op("piper_kernels::convrot_trellis_linear", mutates_args=())
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
    """Apply trellis-coded rotated storage to a floating-point activation."""
    out_features, blocks_per_row = block_scales.shape
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)
    if input.device.type == "meta" or rows == 0 or out_features == 0:
        return input.new_empty((*leading_shape, out_features))
    if blocks_per_row == 0:
        if bias is None:
            return input.new_zeros((*leading_shape, out_features))
        return bias.to(input.dtype).expand(*leading_shape, out_features).clone()

    if runtime.supports_device(input.device):
        from . import triton as trellis_triton  # noqa: PLC0415

        return trellis_triton.linear(
            input,
            bitstreams,
            block_scales,
            scale,
            row_widths,
            bias,
            block_size,
            group_size,
        )

    from . import reference as trellis_reference  # noqa: PLC0415

    return trellis_reference.linear(
        input,
        bitstreams,
        block_scales,
        scale,
        row_widths,
        bias,
        block_size,
        group_size,
    )


@linear.register_fake
def _linear_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    _bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    _scale: torch.Tensor,
    _row_widths: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _block_size: int,
    _group_size: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], block_scales.shape[0]))


__all__ = ["linear"]
