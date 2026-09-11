"""Validated static-scale ConvRot INT8 causal convolution operations."""

from __future__ import annotations

from typing import Literal, cast

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

from . import reference
from ._validation import (
    _validate_common,
    _validate_inference,
    _validate_norm,
)

try:
    from . import triton as triton_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton_backend = None

type SpatialPadding = Literal["none", "reflect", "reflect_right"]


def _supports_triton(input: torch.Tensor) -> bool:  # noqa: A002
    target = AcceleratorTarget.from_device(input.device)
    return target.is_cuda_capability(12, 0)


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
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: list[int],
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
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
        symmetric_spatial_padding,
        right_spatial_padding,
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
    _validate_norm(input, norm_weight, norm_bias, norm_groups, norm_epsilon)

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
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
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
        symmetric_spatial_padding,
        right_spatial_padding,
        residual,
    )
    _validate_norm(input, norm_weight, norm_bias, norm_groups, norm_epsilon)
    return input.new_empty(shape, dtype=torch.float16)


def _padding_flags(padding: SpatialPadding) -> tuple[bool, bool]:
    if padding not in ("none", "reflect", "reflect_right"):
        raise ValueError(f"unsupported ConvRot INT8 Conv3D spatial padding mode {padding!r}")
    return padding == "reflect", padding == "reflect_right"


def _weight_storage(
    weight: ConvRotInt8Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    if not isinstance(weight, ConvRotInt8Tensor) or weight.ndim != 5:
        raise TypeError("ConvRot INT8 Conv3D requires a 5-D ConvRotInt8Tensor weight")
    if weight.act_per_tensor_scale is None:
        raise ValueError("ConvRot INT8 Conv3D requires a static activation scale on the weight")
    return weight.qdata, weight.scale, weight.group_size, weight.act_per_tensor_scale


def conv3d(
    input: torch.Tensor,  # noqa: A002
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: SpatialPadding,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a causal 3-D convolution with a static-scale ConvRot INT8 weight."""
    qdata, scale, group_size, input_scale = _weight_storage(weight)
    _validate_inference((input, weight, scale, input_scale, bias, residual))
    symmetric, right = _padding_flags(padding)
    return _conv3d_op(
        input, qdata, scale, bias, group_size, input_scale, list(stride), symmetric, right, residual
    )


def group_norm_silu_conv3d(
    input: torch.Tensor,  # noqa: A002
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_groups: int,
    norm_epsilon: float,
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: SpatialPadding,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse framewise GroupNorm and SiLU into a causal convolution boundary."""
    qdata, scale, group_size, input_scale = _weight_storage(weight)
    _validate_inference((input, norm_weight, norm_bias, weight, scale, input_scale, bias, residual))
    symmetric, right = _padding_flags(padding)
    return _group_norm_silu_conv3d_op(
        input,
        norm_weight,
        norm_bias,
        norm_groups,
        norm_epsilon,
        qdata,
        scale,
        bias,
        group_size,
        input_scale,
        list(stride),
        symmetric,
        right,
        residual,
    )


__all__ = ["SpatialPadding", "conv3d", "group_norm_silu_conv3d"]
