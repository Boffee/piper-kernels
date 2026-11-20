"""Stable ConvRot INT8 convolution schemas, validation, and runtime dispatch.

Custom operators remain portable because pre-grad padding fusion can emit them
on both CPU and GPU. Fake implementations share the metadata checks run at the
functional boundary, and Triton is loaded lazily only for supported accelerators.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from types import ModuleType
from typing import cast

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8._quantization import (
    validate_activation_scale,
    validate_storage,
)

from . import _policy, reference
from ._policy import Padding, output_shape


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
        if tuple(bias.shape) != (weight_qdata.shape[0],) or bias.dtype not in (
            torch.float16,
            torch.float32,
        ):
            raise ValueError(
                "ConvRot INT8 Conv3D bias must be float16/float32 with one value per output channel"
            )
        if bias.device != weight_qdata.device:
            raise ValueError("ConvRot INT8 Conv3D bias must share the weight device")


def _validate_config(
    stride: list[int],
    padding: Padding,
) -> None:
    if len(stride) != 3 or any(type(value) is not int or value <= 0 for value in stride):
        raise ValueError(
            f"ConvRot INT8 Conv3D stride must contain three positive integers, got {stride}"
        )
    if len(padding) != 6 or any(type(value) is not int or value < 0 for value in padding):
        raise ValueError("ConvRot INT8 Conv3D padding must contain six nonnegative integers")


def _validate_common(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: list[int],
    padding: Padding,
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if input.ndim != 5:
        raise ValueError(f"ConvRot INT8 Conv3D input must be NCTHW, got {tuple(input.shape)}")
    if input.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"ConvRot INT8 Conv3D input must be float16 or float32, got {input.dtype}")
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
    _validate_config(stride, padding)
    if spatial_reflection and (
        max(padding[:2]) >= input.shape[4] or max(padding[2:4]) >= input.shape[3]
    ):
        raise ValueError(
            "ConvRot INT8 Conv3D reflection padding must be smaller than height and width"
        )
    if weight_qdata.device != input.device:
        raise ValueError("ConvRot INT8 Conv3D inputs and weight storage must share a device")

    expected_shape = output_shape(
        input.shape,
        weight_qdata.shape[0],
        stride,
        padding,
    )
    if any(size <= 0 for size in expected_shape):
        raise ValueError("ConvRot INT8 Conv3D kernel does not fit the padded input")
    if residual is not None:
        if residual.layout is not torch.strided:
            raise ValueError("ConvRot INT8 Conv3D residual must use strided layout")
        if residual.device != input.device or residual.dtype is not torch.float16:
            raise ValueError("ConvRot INT8 Conv3D residual must be float16 on the input device")
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


def _select_native_backend(input: torch.Tensor) -> tuple[ModuleType, AcceleratorTarget] | None:  # noqa: A002
    """Return the shared Triton module and target when supported, else None for reference."""
    target = AcceleratorTarget.from_device(input.device)
    if not _policy.supports_target(target):
        return None
    try:
        backend = importlib.import_module(f"{__package__}.triton")
    except ModuleNotFoundError as error:
        if error.name != "triton":
            raise
        return None
    return backend, target


@torch.library.custom_op(
    "piper_kernels::convrot_int8_conv3d",
    mutates_args=(),
)
def _conv3d_op(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: list[int],
    padding: list[int],
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    stride_tuple = cast(tuple[int, int, int], tuple(stride))
    padding_tuple = cast(Padding, tuple(padding))
    native = _select_native_backend(input)
    if native is not None:
        backend, target = native
        plan = _policy.default_execution_plan(
            input,
            weight_qdata,
            stride_tuple,
            group_norm=False,
            padding=padding_tuple,
            target=target,
        )
        return backend.conv3d(
            input,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            input_scale,
            stride_tuple,
            execution_plan=plan,
            accelerator_backend=target.backend,
            padding=padding_tuple,
            spatial_reflection=spatial_reflection,
            residual=residual,
        )
    return reference.conv3d(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride_tuple,
        padding=padding_tuple,
        spatial_reflection=spatial_reflection,
        residual=residual,
    )


@_conv3d_op.register_fake
def _conv3d_fake(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: list[int],
    padding: list[int],
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    shape = _validate_common(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride,
        cast(Padding, tuple(padding)),
        spatial_reflection,
        residual,
    )
    return input.new_empty(shape, dtype=torch.float16)


@torch.library.custom_op(
    "piper_kernels::convrot_int8_group_norm_silu_conv3d",
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
    input_scale: torch.Tensor,
    stride: list[int],
    padding: list[int],
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    stride_tuple = cast(tuple[int, int, int], tuple(stride))
    padding_tuple = cast(Padding, tuple(padding))
    native = _select_native_backend(input)
    if native is not None:
        backend, target = native
        plan = _policy.default_execution_plan(
            input,
            weight_qdata,
            stride_tuple,
            group_norm=True,
            padding=padding_tuple,
            target=target,
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
            stride_tuple,
            execution_plan=plan,
            accelerator_backend=target.backend,
            padding=padding_tuple,
            spatial_reflection=spatial_reflection,
            residual=residual,
        )
    return reference.group_norm_silu_conv3d(
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
        stride_tuple,
        padding=padding_tuple,
        spatial_reflection=spatial_reflection,
        residual=residual,
    )


@_group_norm_silu_conv3d_op.register_fake
def _group_norm_silu_conv3d_fake(  # noqa: PLR0913, PLR0917
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
    stride: list[int],
    padding: list[int],
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    shape = _validate_common(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride,
        cast(Padding, tuple(padding)),
        spatial_reflection,
        residual,
    )
    _validate_norm(input, norm_weight, norm_bias, norm_groups, norm_epsilon)
    return input.new_empty(shape, dtype=torch.float16)
