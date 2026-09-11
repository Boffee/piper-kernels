"""Shared storage and shape contracts for eager and compiled ConvRot convolutions."""

import math
from collections.abc import Sequence

import torch

from piper_kernels.weights.convrot.int8._quantization import (
    validate_activation_scale,
    validate_storage,
)


def _output_shape(
    input_shape: Sequence[int],
    output_channels: int,
    stride: Sequence[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
) -> tuple[int, int, int, int, int]:
    batch, _, frames, height, width = input_shape
    spatial_padding = 2 if symmetric_spatial_padding else int(right_spatial_padding)
    return (
        batch,
        output_channels,
        (frames - 1) // stride[0] + 1,
        (height + spatial_padding - 3) // stride[1] + 1,
        (width + spatial_padding - 3) // stride[2] + 1,
    )


def _validate_weight(
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> None:
    if weight_qdata.ndim != 5:
        raise ValueError(f"ConvRot INT8 Conv3D qdata must be 5-D, got {tuple(weight_qdata.shape)}")
    validate_storage(weight_qdata, weight_scale, group_size, torch.float16)
    if bias is not None:
        if bias.layout is not torch.strided:
            raise ValueError("ConvRot INT8 Conv3D bias must use strided layout")
        if tuple(bias.shape) != (weight_qdata.shape[0],) or bias.dtype is not torch.float16:
            raise ValueError(
                "ConvRot INT8 Conv3D bias must be float16 with one value per output channel"
            )
        if bias.device != weight_qdata.device:
            raise ValueError("ConvRot INT8 Conv3D bias must share the weight device")


def _validate_config(
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
) -> None:
    if len(stride) != 3 or any(type(value) is not int or value <= 0 for value in stride):
        raise ValueError(
            f"ConvRot INT8 Conv3D stride must contain three positive integers, got {stride}"
        )
    if symmetric_spatial_padding and right_spatial_padding:
        raise ValueError("ConvRot INT8 Conv3D spatial padding modes are mutually exclusive")


def _validate_common(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if input.ndim != 5:
        raise ValueError(f"ConvRot INT8 Conv3D input must be NCTHW, got {tuple(input.shape)}")
    if input.dtype is not torch.float16:
        raise ValueError(f"ConvRot INT8 Conv3D input must be float16, got {input.dtype}")
    if input.layout is not torch.strided:
        raise ValueError("ConvRot INT8 Conv3D input must use strided layout")
    if any(size <= 0 for size in input.shape):
        raise ValueError("ConvRot INT8 Conv3D input dimensions must be positive")
    _validate_weight(weight_qdata, weight_scale, bias, group_size)
    if weight_qdata.shape[4] != input.shape[1]:
        raise ValueError(
            f"ConvRot INT8 Conv3D input has {input.shape[1]} channels, "
            f"but the weight expects {weight_qdata.shape[4]}"
        )
    validate_activation_scale(input_scale, weight_qdata.device)
    _validate_config(stride, symmetric_spatial_padding, right_spatial_padding)
    if (symmetric_spatial_padding or right_spatial_padding) and min(input.shape[3:]) <= 1:
        raise ValueError("ConvRot INT8 Conv3D reflection padding requires height and width > 1")
    if weight_qdata.device != input.device:
        raise ValueError("ConvRot INT8 Conv3D inputs and weight storage must share a device")

    expected_shape = _output_shape(
        input.shape,
        weight_qdata.shape[0],
        stride,
        symmetric_spatial_padding,
        right_spatial_padding,
    )
    if any(size <= 0 for size in expected_shape):
        raise ValueError("ConvRot INT8 Conv3D kernel does not fit the padded input")
    if residual is not None:
        if residual.layout is not torch.strided:
            raise ValueError("ConvRot INT8 Conv3D residual must use strided layout")
        if residual.device != input.device or residual.dtype is not input.dtype:
            raise ValueError("ConvRot INT8 Conv3D residual must match the input device and dtype")
        if tuple(residual.shape) != expected_shape:
            raise ValueError(
                f"ConvRot INT8 Conv3D residual must have shape {expected_shape}, "
                f"got {tuple(residual.shape)}"
            )
    return expected_shape


def _validate_norm(
    input: torch.Tensor,  # noqa: A002
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_groups: int,
    norm_epsilon: float,
) -> None:
    channels = input.shape[1]
    if norm_groups <= 0 or channels % norm_groups:
        raise ValueError(
            f"Framewise GroupNorm groups must divide {channels} channels, got {norm_groups}"
        )
    for name, tensor in (("weight", norm_weight), ("bias", norm_bias)):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(
                f"Framewise GroupNorm {name} must have shape ({channels},), "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.layout is not torch.strided:
            raise ValueError(f"Framewise GroupNorm {name} must use strided layout")
        if tensor.device != input.device or tensor.dtype not in (torch.float16, torch.float32):
            raise ValueError(
                f"Framewise GroupNorm {name} must be float16/float32 on {input.device}"
            )
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError(
            f"Framewise GroupNorm epsilon must be finite and positive, got {norm_epsilon}"
        )


def _validate_inference(tensors: Sequence[torch.Tensor | None]) -> None:
    """Reject autograd at the public boundary, before custom-op redispatch disables it."""
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    ):
        raise RuntimeError("ConvRot INT8 Conv3D is inference-only and does not support autograd")
