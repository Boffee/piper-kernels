"""Functional ConvRot INT8 convolutions and causal convenience APIs."""

from collections.abc import Callable
from typing import Any

import torch

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

from . import dispatch
from ._arguments import bind_conv3d_arguments, convolution_parameters
from ._padding import Padding, SpatialPadding, causal_padding
from ._validation import _validate_inference


def convrot_int8_conv3d(
    input: torch.Tensor,  # noqa: A002
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: Padding = (0, 0, 0, 0, 0, 0),
) -> torch.Tensor:
    """Execute a standard convolution with resolved zero padding in F.pad order.

    This typed internal entry point receives geometry resolved by the Python/ATen
    adapter. FP16 autocast converts FP32 inputs/bias without repacking weights;
    BF16 autocast is unsupported. Execution returns FP16 on every backend.
    """
    qdata, scale, group_size, input_scale = _weight_storage(weight)
    _validate_inference((weight,))
    device_type = input.device.type
    if torch.amp.autocast_mode.is_autocast_available(device_type) and torch.is_autocast_enabled(
        device_type
    ):
        dtype = torch.get_autocast_dtype(device_type)
        if dtype is not torch.float16:
            raise ValueError("ConvRot INT8 Conv3D supports only float16 autocast")
        if input.dtype is torch.float32:
            input = input.to(dtype=dtype)  # noqa: A001
        if bias is not None and bias.dtype is torch.float32:
            bias = bias.to(dtype=dtype)
    return dispatch.conv3d(
        input,
        qdata,
        scale,
        bias,
        group_size,
        input_scale,
        stride,
        padding=padding,
    )


def conv3d_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Adapt Python/ATen calls to the typed ConvRot INT8 functional boundary."""
    bound = bind_conv3d_arguments(func, args, kwargs)
    input, weight, bias = bound.input, bound.weight, bound.bias  # noqa: A001
    if not isinstance(input, torch.Tensor) or not isinstance(weight, ConvRotInt8Tensor):
        raise TypeError("ConvRot Conv3D requires a tensor input and ConvRotInt8Tensor weight")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("ConvRot Conv3D bias must be a tensor or None")
    stride, padding = convolution_parameters(bound)
    return convrot_int8_conv3d(input, weight, bias, stride=stride, padding=padding)


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
    _validate_inference((weight,))
    pads, reflection = causal_padding(padding)
    return dispatch.conv3d(
        input,
        qdata,
        scale,
        bias,
        group_size,
        input_scale,
        stride,
        padding=pads,
        spatial_reflection=reflection,
        residual=residual,
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
    _validate_inference((weight,))
    pads, reflection = causal_padding(padding)
    return dispatch.group_norm_silu_conv3d(
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
        stride,
        padding=pads,
        spatial_reflection=reflection,
        residual=residual,
    )
