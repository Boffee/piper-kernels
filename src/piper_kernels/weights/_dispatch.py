"""Shared dispatch helpers for quantized weight wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .convrot.int8.tensor import ConvRotInt8Tensor
    from .nvfp4.tensor import PiperNVFP4Tensor
type QuantizedWeight = ConvRotInt8Tensor | PiperNVFP4Tensor


def unsupported_operation_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    _args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> torch.Tensor:
    """Reject operations without an implementation that preserves quantized weights."""
    raise NotImplementedError(f"Piper quantized weights do not support {func}")


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
