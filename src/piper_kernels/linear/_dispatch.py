"""Argument binding shared by linear tensor-subclass dispatchers."""

from __future__ import annotations

from typing import Any

import torch


def _explicit_to_copy_args(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[tuple[object, ...], dict[str, object]] | None:
    """Remove an explicit true ``copy`` before TorchAO parses ``Tensor.to``."""
    parsed_args = args
    parsed_kwargs = dict(kwargs)
    if "copy" in parsed_kwargs:
        copy = parsed_kwargs.pop("copy")
    else:
        copy_index = 2 if args and isinstance(args[0], (torch.Tensor, torch.dtype)) else 3
        if len(args) <= copy_index:
            return None
        copy = args[copy_index]
        parsed_args = (*args[:copy_index], *args[copy_index + 1 :])
    if copy is not True:
        return None
    return parsed_args, parsed_kwargs


def bind_linear_arguments(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """Bind ``linear(input, weight, bias=None)`` positional and keyword forms."""
    parameter_names = ("input", "weight", "bias")
    if len(args) > len(parameter_names):
        raise TypeError(f"linear() expected at most 3 positional arguments, got {len(args)}")

    bound = dict(zip(parameter_names, args, strict=False))
    for name, value in kwargs.items():
        if name not in parameter_names:
            raise TypeError(f"linear() got an unexpected keyword argument {name!r}")
        if name in bound:
            raise TypeError(f"linear() got multiple values for argument {name!r}")
        bound[name] = value

    missing = [name for name in parameter_names[:2] if name not in bound]
    if missing:
        names = " and ".join(repr(name) for name in missing)
        raise TypeError(f"linear() missing required argument: {names}")
    return bound["input"], bound["weight"], bound.get("bias")


def linear_autocast_dtype(input: torch.Tensor) -> torch.dtype | None:  # noqa: A002
    """Return the active autocast dtype for a linear input, if any."""
    device_type = input.device.type
    if not torch.amp.autocast_mode.is_autocast_available(device_type):
        return None
    if not torch.is_autocast_enabled(device_type):
        return None
    return torch.get_autocast_dtype(device_type)


def apply_linear_autocast(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply the active autocast dtype at a semantic linear boundary.

    PyTorch may invoke a public semantic function directly or redispatch ``linear``
    after converting its operands. Quantized wrappers implement ``_to_copy`` so either
    route changes only their logical compute dtype, not their quantized storage dtype.
    """
    dtype = linear_autocast_dtype(input)
    if dtype is None:
        return input, weight, bias

    device_type = input.device.type

    def cast_if_eligible(tensor: torch.Tensor) -> torch.Tensor:
        if (
            tensor.is_floating_point()
            and tensor.device.type == device_type
            and tensor.dtype is not torch.float64
        ):
            return tensor.to(dtype=dtype)
        return tensor

    return (
        cast_if_eligible(input),
        cast_if_eligible(weight),
        None if bias is None else cast_if_eligible(bias),
    )


__all__ = ["apply_linear_autocast", "bind_linear_arguments", "linear_autocast_dtype"]
