"""Views and storage layout guards shared by quantized weight wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from torch._prims_common import infer_size
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import TorchAOBaseTensor

from ._dispatch import unsupported_operation_dispatch

if TYPE_CHECKING:
    from ._dispatch import QuantizedWeight


def require_untransposed(weight: QuantizedWeight, operation: str) -> None:
    """Reject operations that would interpret a transpose as canonical storage."""
    if weight.transposed:
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} does not support a transposed weight"
        )


def _transpose_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast("QuantizedWeight", args[0])
    if weight.ndim != 2:
        raise NotImplementedError(f"{type(weight).__name__} transpose requires a 2-D weight")
    swap = True
    if func is torch.ops.aten.transpose.int:
        dim0, dim1 = args[1:3]
        if dim0 not in (-2, -1, 0, 1) or dim1 not in (-2, -1, 0, 1):
            raise IndexError("dimension out of range for a 2-D quantized weight")
        swap = dim0 % 2 != dim1 % 2
    elif func is torch.ops.aten.permute.default:
        dims = args[1]
        if len(dims) != 2 or any(dim not in (-2, -1, 0, 1) for dim in dims):
            raise ValueError("quantized weight permute requires both matrix dimensions")
        dims = tuple(dim % 2 for dim in dims)
        if dims not in ((0, 1), (1, 0)):
            raise ValueError("quantized weight permute requires distinct dimensions")
        swap = dims == (1, 0)
    viewed = weight._transpose() if swap else weight._apply_fn_to_data(torch.ops.aten.alias.default)
    return cast(torch.Tensor, return_and_correct_aliasing(func, args, kwargs, viewed))


def _same_shape_view_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Create a fresh wrapper without changing its quantized storage layout."""
    tensor = cast(TorchAOBaseTensor, args[0])
    size = args[1].shape if func is torch.ops.aten.view_as.default else args[1]
    shape = infer_size(size, tensor.numel())
    if shape != tuple(tensor.shape):
        raise NotImplementedError(
            f"{type(tensor).__name__} only supports same-shape views; "
            f"cannot view {tuple(tensor.shape)} as {shape}"
        )
    return _alias(func, args, kwargs)


def _as_strided_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Support AOTAutograd replay of an alias or a matrix transpose."""
    tensor = cast(TorchAOBaseTensor, args[0])
    shape, strides = args[1:3]
    offset = args[3] if len(args) > 3 else kwargs.get("storage_offset")
    if (
        tensor.ndim == 2
        and tuple(shape) == tuple(reversed(tensor.shape))
        and tuple(strides) == tuple(reversed(tensor.stride()))
        and (offset is None or offset == tensor.storage_offset())
    ):
        viewed = cast("QuantizedWeight", tensor)._transpose()
        return cast(torch.Tensor, return_and_correct_aliasing(func, args, kwargs, viewed))
    if (
        tuple(shape) != tuple(tensor.shape)
        or tuple(strides) != tensor.stride()
        or (offset is not None and offset != tensor.storage_offset())
    ):
        raise NotImplementedError(
            f"{type(tensor).__name__} only supports as_strided for an unchanged layout "
            "or a matrix transpose, with unchanged storage offset"
        )
    return _alias(func, args, kwargs)


def _alias(
    func: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    tensor = cast(TorchAOBaseTensor, args[0])
    # DTensor needs a distinct wrapper for autograd metadata, sharing the weight
    # storage. Rebuild through the concrete class to retain all quantization data.
    viewed = tensor._apply_fn_to_data(torch.ops.aten.alias.default)
    return cast(torch.Tensor, return_and_correct_aliasing(func, args, kwargs, viewed))


def _contiguous_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast("QuantizedWeight", args[0])
    require_untransposed(weight, "contiguous")
    return weight._apply_fn_to_data(lambda value: func(value, *args[1:], **kwargs))


def _clone_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast("QuantizedWeight", args[0])
    if kwargs.get("memory_format") not in (None, torch.preserve_format):
        require_untransposed(weight, "clone with a different memory format")
    return weight._apply_fn_to_data(lambda value: func(value, **kwargs))


def register_view_ops(cls: type[TorchAOBaseTensor]) -> None:
    """Preserve wrappers through views and reject unsupported layout changes."""
    aten = torch.ops.aten
    cls.implements([aten.view.default, aten.view_as.default])(_same_shape_view_dispatch)
    cls.implements(aten.as_strided.default)(_as_strided_dispatch)
    cls.implements([aten.t.default, aten.transpose.int, aten.permute.default])(_transpose_dispatch)
    cls.implements(aten.clone.default)(_clone_dispatch)
    cls.implements(aten.contiguous.default)(_contiguous_dispatch)
    cls.implements_torch_function(torch.Tensor.contiguous)(_contiguous_dispatch)
    # TorchAO's inherited slicing handlers construct a base NVFP4Tensor, losing
    # interpretation metadata. Use shard_quantized_weight for owning partitions:
    # repacked scales cannot promise the aliasing semantics of these views.
    cls.implements([aten.slice.Tensor, aten.select.int])(unsupported_operation_dispatch)


__all__ = ["register_view_ops", "require_untransposed"]
