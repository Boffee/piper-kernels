"""Argument binding shared by linear tensor-subclass dispatchers."""

from __future__ import annotations

from typing import Any


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


__all__ = ["bind_linear_arguments"]
