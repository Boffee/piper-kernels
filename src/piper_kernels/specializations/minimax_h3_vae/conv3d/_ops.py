"""Validated custom operations for the H3 VAE ConvRot INT8 encoder."""

from __future__ import annotations

import math
from typing import Literal, cast

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot._rotation import validate_group_size

from . import reference
from .calibration import P995_ACTIVATION_SCALES

try:
    from . import triton as triton_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton_backend = None

type SpatialPadding = Literal["none", "reflect", "reflect_right"]


def prepare_weight(
    layer_name: str,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Quantize a calibrated H3 encoder convolution into channelwise ConvRot storage."""
    input_scale = P995_ACTIVATION_SCALES.get(layer_name)
    if input_scale is None:
        raise ValueError(f"H3 VAE encoder layer {layer_name!r} has no calibrated scale")
    if weight.ndim != 5:
        raise ValueError(f"H3 VAE encoder weight must be 5-D, got {tuple(weight.shape)}")
    input_channels = weight.shape[1]
    group_size = 64 if input_channels == 128 else 256
    qdata, weight_scale = reference.quantize_weight(weight, group_size)
    return qdata, weight_scale, group_size, input_scale


def _output_shape(
    input: torch.Tensor,  # noqa: A002
    output_channels: int,
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
) -> tuple[int, int, int, int, int]:
    batch, _, frames, height, width = input.shape
    spatial_padding = 2 if symmetric_spatial_padding else int(right_spatial_padding)
    return (
        batch,
        output_channels,
        (frames - 1) // stride[0] + 1,
        (height + spatial_padding - 3) // stride[1] + 1,
        (width + spatial_padding - 3) // stride[2] + 1,
    )


def _validate_common(  # noqa: PLR0912
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: float,
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
) -> None:
    if input.ndim != 5:
        raise ValueError(f"H3 VAE ConvRot input must be NCTHW, got {tuple(input.shape)}")
    if input.dtype is not torch.float16:
        raise ValueError(f"H3 VAE ConvRot input must be float16, got {input.dtype}")
    if input.layout is not torch.strided:
        raise ValueError("H3 VAE ConvRot input must use strided layout")
    if weight_qdata.dtype is not torch.int8 or weight_qdata.ndim != 5:
        raise ValueError(
            "H3 VAE ConvRot qdata must be 5-D int8, "
            f"got {weight_qdata.dtype} {tuple(weight_qdata.shape)}"
        )
    if tuple(weight_qdata.shape[1:4]) != (3, 3, 3):
        raise ValueError(
            f"H3 VAE ConvRot qdata must use a 3x3x3 kernel, got {tuple(weight_qdata.shape[1:4])}"
        )
    if weight_qdata.shape[4] != input.shape[1]:
        raise ValueError(
            f"H3 VAE ConvRot input has {input.shape[1]} channels, "
            f"but the weight expects {weight_qdata.shape[4]}"
        )
    if weight_scale.dtype is not torch.float32 or tuple(weight_scale.shape) not in (
        (weight_qdata.shape[0],),
        (weight_qdata.shape[0], 1),
    ):
        raise ValueError(
            "H3 VAE ConvRot weight scale must be float32 with one value per output channel"
        )
    if not weight_qdata.is_contiguous() or not weight_scale.is_contiguous():
        raise ValueError("H3 VAE ConvRot weight storage must be contiguous")
    validate_group_size(group_size)
    if input.shape[1] % group_size:
        raise ValueError(
            f"H3 VAE ConvRot input channels {input.shape[1]} are not divisible "
            f"by group size {group_size}"
        )
    if not math.isfinite(input_scale) or input_scale <= 0:
        raise ValueError(
            f"H3 VAE ConvRot input scale must be finite and positive, got {input_scale}"
        )
    if len(stride) != 3 or any(type(value) is not int or value <= 0 for value in stride):
        raise ValueError(
            f"H3 VAE ConvRot stride must contain three positive integers, got {stride}"
        )
    if symmetric_spatial_padding and right_spatial_padding:
        raise ValueError("H3 VAE ConvRot spatial padding modes are mutually exclusive")

    tensors = [weight_qdata, weight_scale]
    if bias is not None:
        if tuple(bias.shape) != (weight_qdata.shape[0],) or bias.dtype is not torch.float16:
            raise ValueError(
                "H3 VAE ConvRot bias must be float16 with one value per output channel"
            )
        tensors.append(bias)
    if any(tensor.device != input.device for tensor in tensors):
        raise ValueError("H3 VAE ConvRot inputs and weight storage must share a device")

    expected_shape = _output_shape(
        input,
        weight_qdata.shape[0],
        stride,
        symmetric_spatial_padding,
        right_spatial_padding,
    )
    if residual is not None:
        if residual.device != input.device or residual.dtype is not input.dtype:
            raise ValueError("H3 VAE ConvRot residual must match the input device and dtype")
        if tuple(residual.shape) != expected_shape:
            raise ValueError(
                f"H3 VAE ConvRot residual must have shape {expected_shape}, "
                f"got {tuple(residual.shape)}"
            )
    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight_scale.requires_grad
        or (bias is not None and bias.requires_grad)
        or (residual is not None and residual.requires_grad)
    ):
        raise RuntimeError("H3 VAE ConvRot convolution is inference-only")


def _supports_triton(input: torch.Tensor) -> bool:  # noqa: A002
    if triton_backend is None:
        return False
    target = AcceleratorTarget.from_device(input.device)
    return target.is_nvidia_cuda and target.cuda_capability_at_least(7, 5)


@torch.library.custom_op(
    "piper_kernels::minimax_h3_vae_convrot_int8_conv3d",
    mutates_args=(),
)
def _conv3d_op(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: float,
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    _validate_common(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride,
        symmetric_spatial_padding,
        right_spatial_padding,
        residual,
    )
    backend = (
        triton_backend if triton_backend is not None and _supports_triton(input) else reference
    )
    return backend.conv3d(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        cast(tuple[int, int, int], tuple(stride)),
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
        residual=residual,
    )


@_conv3d_op.register_fake
def _conv3d_fake(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    _input_scale: float,
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    _residual: torch.Tensor | None,
) -> torch.Tensor:
    shape = _output_shape(
        input,
        weight_qdata.shape[0],
        stride,
        symmetric_spatial_padding,
        right_spatial_padding,
    )
    return input.new_empty(shape)


@torch.library.custom_op(
    "piper_kernels::minimax_h3_vae_convrot_int8_group_norm_silu_conv3d",
    mutates_args=(),
)
def _group_norm_silu_conv3d_op(  # noqa: PLR0913, PLR0917
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
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    _validate_common(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride,
        symmetric_spatial_padding,
        right_spatial_padding,
        residual,
    )
    channels = input.shape[1]
    if norm_groups <= 0 or channels % norm_groups:
        raise ValueError(
            f"H3 VAE GroupNorm groups must divide {channels} channels, got {norm_groups}"
        )
    for name, tensor in (("weight", norm_weight), ("bias", norm_bias)):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(
                f"H3 VAE GroupNorm {name} must have shape ({channels},), got {tuple(tensor.shape)}"
            )
        if tensor.device != input.device or tensor.dtype not in (torch.float16, torch.float32):
            raise ValueError(f"H3 VAE GroupNorm {name} must be float16/float32 on {input.device}")
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError(
            f"H3 VAE GroupNorm epsilon must be finite and positive, got {norm_epsilon}"
        )
    if torch.is_grad_enabled() and (norm_weight.requires_grad or norm_bias.requires_grad):
        raise RuntimeError("H3 VAE ConvRot convolution is inference-only")

    backend = (
        triton_backend if triton_backend is not None and _supports_triton(input) else reference
    )
    return backend.group_norm_silu_conv3d(
        input,
        norm_weight,
        norm_bias,
        norm_groups,
        norm_epsilon,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        cast(tuple[int, int, int], tuple(stride)),
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
        residual=residual,
    )


@_group_norm_silu_conv3d_op.register_fake
def _group_norm_silu_conv3d_fake(
    input: torch.Tensor,  # noqa: A002
    _norm_weight: torch.Tensor,
    _norm_bias: torch.Tensor,
    _norm_groups: int,
    _norm_epsilon: float,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    _input_scale: float,
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    _residual: torch.Tensor | None,
) -> torch.Tensor:
    shape = _output_shape(
        input,
        weight_qdata.shape[0],
        stride,
        symmetric_spatial_padding,
        right_spatial_padding,
    )
    return input.new_empty(shape)


def _padding_flags(padding: SpatialPadding) -> tuple[bool, bool]:
    if padding not in ("none", "reflect", "reflect_right"):
        raise ValueError(f"unsupported H3 VAE ConvRot spatial padding mode {padding!r}")
    return padding == "reflect", padding == "reflect_right"


def conv3d(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: float,
    stride: tuple[int, int, int],
    *,
    padding: SpatialPadding,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one calibrated H3 encoder ConvRot INT8 3-D convolution."""
    symmetric, right = _padding_flags(padding)
    return _conv3d_op(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        list(stride),
        symmetric,
        right,
        residual,
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
    padding: SpatialPadding,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse isolated GroupNorm and SiLU into one calibrated H3 convolution boundary."""
    symmetric, right = _padding_flags(padding)
    return _group_norm_silu_conv3d_op(
        input,
        norm_weight,
        norm_bias,
        norm_groups,
        norm_epsilon,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        list(stride),
        symmetric,
        right,
        residual,
    )


__all__ = ["SpatialPadding", "conv3d", "group_norm_silu_conv3d", "prepare_weight"]
