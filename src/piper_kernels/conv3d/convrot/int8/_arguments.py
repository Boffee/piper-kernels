"""Bind Python/ATen arguments and resolve the supported Conv3D configuration."""

from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple, cast

import torch

from ._padding import Padding


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
