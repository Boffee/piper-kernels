"""Portable reference operations for static-scale ConvRot INT8 convolutions."""

from __future__ import annotations

import torch
from torch.nn import functional

from piper_kernels.weights.convrot._rotation import rotate_groups


def _prepare_input(
    input: torch.Tensor,  # noqa: A002
    group_size: int,
    input_scale: torch.Tensor,
) -> torch.Tensor:
    tokens = input.permute(0, 2, 3, 4, 1).to(
        dtype=torch.float32, memory_format=torch.contiguous_format
    )
    rotated = rotate_groups(tokens, group_size)
    return (rotated / input_scale).round().clamp(-128, 127).to(torch.int8)


def _prepare_group_norm_silu_input(
    input: torch.Tensor,  # noqa: A002
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_groups: int,
    norm_epsilon: float,
    group_size: int,
    input_scale: torch.Tensor,
) -> torch.Tensor:
    batch, channels, frames, height, width = input.shape
    channels_per_group = channels // norm_groups
    grouped = (
        input.permute(0, 2, 1, 3, 4)
        .float()
        .reshape(
            batch,
            frames,
            norm_groups,
            channels_per_group,
            height,
            width,
        )
    )
    mean = grouped.mean(dim=(3, 4, 5), keepdim=True)
    variance = grouped.var(dim=(3, 4, 5), correction=0, keepdim=True)
    normalized = ((grouped - mean) * torch.rsqrt(variance + norm_epsilon)).reshape(
        batch,
        frames,
        channels,
        height,
        width,
    )
    normalized = normalized * norm_weight.float().view(
        1, 1, channels, 1, 1
    ) + norm_bias.float().view(1, 1, channels, 1, 1)
    activated = normalized * torch.sigmoid(normalized)
    return _prepare_input(
        activated.permute(0, 2, 1, 3, 4),
        group_size,
        input_scale,
    )


def _convolution(
    prepared: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    input_scale: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    activation = prepared.permute(0, 4, 1, 2, 3).float().mul(input_scale)
    if symmetric_spatial_padding:
        activation = functional.pad(activation, (1, 1, 1, 1, 0, 0), mode="reflect")
    elif right_spatial_padding:
        activation = functional.pad(activation, (0, 1, 0, 1, 0, 0), mode="reflect")
    activation = functional.pad(activation, (0, 0, 0, 0, 2, 0))

    weight = weight_qdata.float().mul(weight_scale.view(-1, 1, 1, 1, 1)).permute(0, 4, 1, 2, 3)
    output = functional.conv3d(
        activation,
        weight,
        None if bias is None else bias.float(),
        stride=stride,
    )
    if residual is not None:
        output = output + residual.float()
    return output.to(dtype=output_dtype, memory_format=torch.contiguous_format)


def conv3d(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Execute the specialized quantized convolution with portable PyTorch."""
    prepared = _prepare_input(input, group_size, input_scale)
    return _convolution(
        prepared,
        weight_qdata,
        weight_scale,
        bias,
        input_scale,
        stride,
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
        residual=residual,
        output_dtype=input.dtype,
    )


def group_norm_silu_conv3d(  # noqa: PLR0913, PLR0917
    input: torch.Tensor,  # noqa: A002
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_groups: int,
    norm_epsilon: float,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Fuse isolated GroupNorm and SiLU into portable ConvRot preparation."""
    prepared = _prepare_group_norm_silu_input(
        input,
        norm_weight,
        norm_bias,
        norm_groups,
        norm_epsilon,
        group_size,
        input_scale,
    )
    return _convolution(
        prepared,
        weight_qdata,
        weight_scale,
        bias,
        input_scale,
        stride,
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
        residual=residual,
        output_dtype=input.dtype,
    )


__all__ = ["conv3d", "group_norm_silu_conv3d"]
