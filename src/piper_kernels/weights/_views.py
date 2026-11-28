"""Views and storage layout guards shared by quantized weight wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from torch._prims_common import infer_size
from torch.utils._python_dispatch import return_and_correct_aliasing

from piper_kernels.weights._dispatch import (
    QuantizedWeight,
    unsupported_operation_dispatch,
)
from piper_kernels.weights._dispatch import (
    # Retained for linear/* callers until Wave 2 migrates them to weights._dispatch.
    require_untransposed as require_untransposed,  # noqa: PLC0414 - shared export
)

if TYPE_CHECKING:
    from torch._ops import OpOverload


class _FlatQuantizedWeightView(torch.Tensor):
    """Opaque 1-D storage handle returned for ``weight.view(-1)`` during FSDP2 setup."""

    def __new__(cls, source: torch.Tensor) -> _FlatQuantizedWeightView:
        # Avoid holding a reference to source so state-dict swaps can reclaim stale shards.
        return cast(
            _FlatQuantizedWeightView,
            torch.Tensor._make_wrapper_subclass(
                cls,
                (source.numel(),),
                dtype=source.dtype,
                device=source.device,
                requires_grad=False,
            ),
        )

    def __repr__(self, *, tensor_contents: object = None) -> str:
        _ = tensor_contents
        return (
            f"{type(self).__name__}(shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, device={self.device})"
        )

    @classmethod
    def __torch_dispatch__(
        cls,
        func: OpOverload,
        _types: tuple[type, ...],
        _args: tuple[object, ...],
        _kwargs: dict[str, object] | None = None,
    ) -> object:
        raise NotImplementedError(
            f"1-D flat views of Piper quantized weights do not support {func}"
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
    tensor = cast(QuantizedWeight, args[0])
    size = args[1].shape if func is torch.ops.aten.view_as.default else args[1]
    shape = infer_size(size, tensor.numel())
    if shape == tuple(tensor.shape):
        return _alias(func, args, kwargs)
    if len(shape) == 1 and tensor._supports_flat_1d_view:
        if tensor.ndim != 2:
            raise NotImplementedError(
                f"{type(tensor).__name__} 1-D flat view requires a 2-D weight"
            )
        require_untransposed(tensor, "1-D flat view")
        return _FlatQuantizedWeightView(tensor)
    raise NotImplementedError(
        f"{type(tensor).__name__} only supports same-shape views; "
        f"cannot view {tuple(tensor.shape)} as {shape}"
    )


def _as_strided_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Support AOTAutograd replay of an alias or a matrix transpose."""
    tensor = cast(QuantizedWeight, args[0])
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
    tensor = cast(QuantizedWeight, args[0])
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


def _full_extent_slice_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Allow full-extent dim-0 aliases used by FSDP2 while rejecting partial slices."""
    weight = cast("QuantizedWeight", args[0])
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    start = args[2] if len(args) > 2 else kwargs.get("start")
    end = args[3] if len(args) > 3 else kwargs.get("end")
    step = args[4] if len(args) > 4 else kwargs.get("step", 1)
    if (
        weight.ndim == 2
        and not weight.transposed
        and type(dim) is int
        and dim in (0, -2)
        and (start is None or (type(start) is int and start == 0))
        and (end is None or (type(end) is int and end >= weight.shape[0]))
        and type(step) is int
        and step == 1
    ):
        return _alias(func, args, kwargs)
    raise NotImplementedError(f"Piper quantized weights do not support {func}")


def register_view_ops(cls: type[QuantizedWeight]) -> None:
    """Preserve wrappers through views and reject unsupported layout changes."""
    aten = torch.ops.aten
    cls.implements([aten.view.default, aten.view_as.default])(_same_shape_view_dispatch)
    cls.implements(aten.as_strided.default)(_as_strided_dispatch)
    cls.implements([aten.t.default, aten.transpose.int, aten.permute.default])(_transpose_dispatch)
    cls.implements(aten.clone.default)(_clone_dispatch)
    cls.implements(aten.contiguous.default)(_contiguous_dispatch)
    cls.implements_torch_function(torch.Tensor.contiguous)(_contiguous_dispatch)
    # Only full-extent dim-0 aliases (for FSDP2) are permitted here;
    # use shard_quantized_weight for owning partitions.
    cls.implements(aten.slice.Tensor)(_full_extent_slice_dispatch)
    cls.implements(aten.select.int)(unsupported_operation_dispatch)


__all__ = ["_FlatQuantizedWeightView", "register_view_ops", "require_untransposed"]
