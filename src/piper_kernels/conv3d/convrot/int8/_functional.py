"""Functional ConvRot INT8 convolutions, argument binding, and causal convenience APIs."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NamedTuple, cast

import torch

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

from . import _ops
from ._ops import _validate_common, _validate_inference, _validate_norm
from ._policy import Padding

type SpatialPadding = Literal["none", "reflect", "reflect_right"]


def causal_padding(padding: SpatialPadding) -> tuple[Padding, bool]:
    """Resolve causal convenience padding to explicit F.pad-order (W, H, T) padding."""
    if padding == "none":
        return (0, 0, 0, 0, 2, 0), False
    if padding == "reflect":
        return (1, 1, 1, 1, 2, 0), True
    if padding == "reflect_right":
        return (0, 1, 0, 1, 2, 0), True
    raise ValueError(f"unsupported ConvRot INT8 Conv3D spatial padding mode {padding!r}")


class Conv3dArguments(NamedTuple):
    """Unvalidated arguments; operands may be tensors or compiler graph nodes."""

    input: object
    weight: object
    bias: object
    stride: object
    padding: object
    dilation: object
    groups: object
    transposed: object
    output_padding: object


def _triple(value: object, name: str, *, minimum: int) -> tuple[int, int, int]:
    if type(value) is int:
        values = (value, value, value)
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
        if len(values) == 1:
            values = values * 3
    else:
        raise TypeError(f"ConvRot INT8 Conv3D {name} must be an integer or three integers")
    if len(values) != 3 or any(type(item) is not int or item < minimum for item in values):
        raise ValueError(f"ConvRot INT8 Conv3D {name} must contain three integers >= {minimum}")
    return cast(tuple[int, int, int], values)


def _bind(
    names: Sequence[str], args: tuple[object, ...], kwargs: Mapping[str, object]
) -> dict[str, object]:
    if len(args) > len(names):
        raise TypeError("conv3d() received too many positional arguments")
    bound = dict(zip(names, args, strict=False))
    for name, value in kwargs.items():
        if name not in names:
            raise TypeError(f"conv3d() got an unexpected keyword argument {name!r}")
        if name in bound:
            raise TypeError(f"conv3d() got multiple values for argument {name!r}")
        bound[name] = value
    for name in ("input", "weight"):
        if name not in bound:
            raise TypeError(f"conv3d() missing required argument {name!r}")
    return bound


def bind_conv3d_arguments(
    func: Callable[..., torch.Tensor], args: tuple[object, ...], kwargs: Mapping[str, object]
) -> Conv3dArguments:
    """Bind Python and ATen forms at the semantic operator boundary."""
    names = ("input", "weight", "bias", "stride", "padding", "dilation")
    convolution = func in (torch.ops.aten.convolution.default, torch.ops.aten._convolution.default)
    if convolution:
        names += ("transposed", "output_padding", "groups")
        if func is torch.ops.aten._convolution.default:
            names += ("benchmark", "deterministic", "cudnn_enabled", "allow_tf32")
    else:
        names += ("groups",)
    bound = _bind(names, args, kwargs)
    return Conv3dArguments(
        bound["input"],
        bound["weight"],
        bound.get("bias"),
        bound.get("stride", 1),
        bound.get("padding", 0),
        bound.get("dilation", 1),
        bound.get("groups", 1),
        bound.get("transposed", False),
        bound.get("output_padding", 0),
    )


def convolution_parameters(bound: Conv3dArguments) -> tuple[tuple[int, int, int], Padding]:
    """Validate supported configuration and resolve explicit zero padding."""
    if bound.transposed:
        raise NotImplementedError("ConvRot INT8 Conv3D does not support transposed convolution")
    if _triple(bound.output_padding, "output_padding", minimum=0) != (0, 0, 0):
        raise NotImplementedError("ConvRot INT8 Conv3D does not support output_padding")
    if bound.groups != 1:
        raise NotImplementedError("ConvRot INT8 Conv3D requires groups=1")
    if _triple(bound.dilation, "dilation", minimum=1) != (1, 1, 1):
        raise NotImplementedError("ConvRot INT8 Conv3D requires dilation=1")
    stride = _triple(bound.stride, "stride", minimum=1)
    padding = bound.padding
    if isinstance(padding, str):
        if padding == "valid":
            padding = 0
        elif padding == "same":
            if stride != (1, 1, 1):
                raise ValueError("ConvRot INT8 Conv3D padding='same' requires stride=1")
            padding = 1
        else:
            raise ValueError(f"unsupported ConvRot INT8 Conv3D padding {padding!r}")
    time, height, width = _triple(padding, "padding", minimum=0)
    return stride, (width, width, height, height, time, time)


def _weight_storage(
    weight: ConvRotInt8Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    if not isinstance(weight, ConvRotInt8Tensor) or weight.ndim != 5:
        raise TypeError("ConvRot INT8 Conv3D requires a 5-D ConvRotInt8Tensor weight")
    if weight.act_per_tensor_scale is None:
        raise ValueError("ConvRot INT8 Conv3D requires a static activation scale on the weight")
    return weight.qdata, weight.scale, weight.group_size, weight.act_per_tensor_scale


def convrot_int8_conv3d(
    input: torch.Tensor,  # noqa: A002
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: Padding = (0, 0, 0, 0, 0, 0),
) -> torch.Tensor:
    """Execute a standard convolution with resolved zero padding in F.pad order.

    FP16 autocast converts FP32 inputs/bias without repacking weights; BF16
    autocast is unsupported. Execution returns FP16 on every backend.
    """
    qdata, scale, group_size, input_scale = _weight_storage(weight)
    _validate_inference((weight, input, scale, input_scale, bias))
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
    stride_list = list(stride)
    _validate_common(
        input, qdata, scale, bias, group_size, input_scale, stride_list, padding, False, None
    )
    return _ops._conv3d_op(
        input, qdata, scale, bias, group_size, input_scale, stride_list, list(padding), False, None
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
    _validate_inference((weight, input, scale, input_scale, bias, residual))
    pads, reflection = causal_padding(padding)
    stride_list = list(stride)
    _validate_common(
        input,
        qdata,
        scale,
        bias,
        group_size,
        input_scale,
        stride_list,
        pads,
        reflection,
        residual,
    )
    return _ops._conv3d_op(
        input,
        qdata,
        scale,
        bias,
        group_size,
        input_scale,
        stride_list,
        list(pads),
        reflection,
        residual,
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
    _validate_inference((weight, input, norm_weight, norm_bias, scale, input_scale, bias, residual))
    pads, reflection = causal_padding(padding)
    stride_list = list(stride)
    _validate_common(
        input,
        qdata,
        scale,
        bias,
        group_size,
        input_scale,
        stride_list,
        pads,
        reflection,
        residual,
    )
    _validate_norm(input, norm_weight, norm_bias, norm_groups, norm_epsilon)
    return _ops._group_norm_silu_conv3d_op(
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
        stride_list,
        list(pads),
        reflection,
        residual,
    )
