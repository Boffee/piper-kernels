"""Portable reference operations for the H3 VAE ConvRot INT8 convolutions."""

from __future__ import annotations

import torch
from torch.nn import functional

from piper_kernels.linear.convrot._rotation import rotate_groups, validate_group_size
from piper_kernels.linear.convrot.int8.reference import dynamic_quantize_rows


def _rotate_weight_groups(value: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply ConvRot's H4 Kronecker transform without a dense CPU matmul."""
    if value.device.type != "cpu":
        return rotate_groups(value.float(), group_size)

    features = value.shape[-1]
    rotated = value.float().reshape(-1, features // group_size, group_size)
    stride = 1
    while stride < group_size:
        stage = rotated.reshape(
            *rotated.shape[:-1],
            group_size // (4 * stride),
            4,
            stride,
        )
        first, second, third, fourth = stage.unbind(-2)
        rotated = torch.stack(
            (
                first + second + third - fourth,
                first + second - third + fourth,
                first - second + third + fourth,
                -first + second + third + fourth,
            ),
            dim=-2,
        ).reshape(rotated.shape)
        stride *= 4
    return rotated.mul_(group_size**-0.5).reshape(value.shape)


def quantize_weight(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one OI-DHW weight after rotating each spatial-channel row."""
    validate_group_size(group_size)
    if weight.ndim != 5 or tuple(weight.shape[2:]) != (3, 3, 3):
        raise ValueError(
            f"H3 VAE ConvRot weight must have shape [out, in, 3, 3, 3], got {tuple(weight.shape)}"
        )
    if weight.shape[1] % group_size:
        raise ValueError(
            f"H3 VAE ConvRot input channels {weight.shape[1]} must be divisible "
            f"by group size {group_size}"
        )
    if weight.device.type == "meta":
        raise ValueError("H3 VAE ConvRot cannot quantize a meta weight")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"H3 VAE ConvRot weight must be float16, bfloat16, or float32, got {weight.dtype}"
        )

    logical = weight.detach().float()
    channel_last = logical.permute(0, 2, 3, 4, 1).contiguous()
    rotated = _rotate_weight_groups(channel_last, group_size)
    qdata, scale = dynamic_quantize_rows(rotated.flatten(1))
    return qdata.view_as(rotated).contiguous(), scale.flatten().contiguous()


def _prepare_input(
    input: torch.Tensor,  # noqa: A002
    group_size: int,
    input_scale: float,
) -> torch.Tensor:
    tokens = input.permute(0, 2, 3, 4, 1).contiguous()
    rotated = rotate_groups(tokens.float(), group_size)
    logical_scale = torch.tensor(input_scale, device=input.device, dtype=torch.float32)
    return (rotated / logical_scale).float().round().clamp(-128, 127).to(torch.int8)


def _prepare_group_norm_silu_input(
    input: torch.Tensor,  # noqa: A002
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_groups: int,
    norm_epsilon: float,
    group_size: int,
    input_scale: float,
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
    input_scale: float,
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

    weight = (
        weight_qdata.float().mul(weight_scale.float().view(-1, 1, 1, 1, 1)).permute(0, 4, 1, 2, 3)
    )
    output = functional.conv3d(
        activation,
        weight,
        None if bias is None else bias.float(),
        stride=stride,
    )
    if residual is not None:
        output = output + residual.float()
    return output.to(output_dtype)


def conv3d(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: float,
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
    input_scale: float,
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


__all__ = ["conv3d", "group_norm_silu_conv3d", "quantize_weight"]
