"""Shared dispatch helpers for quantized weight wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torchao.utils import TorchAOBaseTensor, _dispatch__torch_dispatch__

if TYPE_CHECKING:
    from torch._ops import OpOverload

    from .convrot.int8.tensor import ConvRotInt8Tensor
    from .nvfp4.tensor import PiperNVFP4Tensor
type QuantizedWeight = ConvRotInt8Tensor | PiperNVFP4Tensor


def _torch_dispatch(
    cls: type[TorchAOBaseTensor],
    func: OpOverload,
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object:
    # object: an op returns whatever its schema declares, not only tensors.
    if func not in cls._ATEN_OP_TABLE[cls]:
        decomposed = func.decompose(*args, **kwargs)
        if decomposed is not NotImplemented:
            return decomposed
    elif func.is_view and torch.is_inference_mode_enabled() and not args[0].is_inference():
        with torch.inference_mode(False):
            return _dispatch__torch_dispatch__(cls, func, types, args, kwargs)
    return _dispatch__torch_dispatch__(cls, func, types, args, kwargs)


def _decompose_dispatch(
    func: OpOverload,
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object:
    return func.decompose(*args, **kwargs)


def register_inference_mode_dispatch(cls: type[TorchAOBaseTensor]) -> None:
    """Dispatch the same way under ``torch.inference_mode`` as under ``torch.no_grad``.

    Autograd dispatch runs CompositeImplicitAutograd decompositions, such as ``aten.to``
    returning ``self`` or lowering to ``_to_copy``, before an op reaches a wrapper.
    Inference mode skips autograd dispatch, so those ops arrive intact; running their
    decompositions here reaches the implementations every other grad mode reaches.
    Registered implementations, such as ``linear``, keep precedence.

    Inference mode still tracks views of a normal tensor, one created outside it, and
    that tracking requires the view to be a normal tensor too, as ATen's views are.
    A wrapper constructed in inference mode is an inference tensor, so registered view
    implementations of a normal weight run with inference mode disabled.
    """
    cls.__torch_dispatch__ = classmethod(_torch_dispatch)
    # Only inference mode reaches TorchAO's handler, which diverges from the
    # decomposition: it copies for an unindexed same device and rejects non_blocking.
    cls.implements(torch.ops.aten.to.dtype_layout)(_decompose_dispatch)


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
