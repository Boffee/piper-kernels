"""Aliasing views shared by quantized weight wrappers."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from torch._prims_common import infer_size
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import TorchAOBaseTensor

if TYPE_CHECKING:
    from ._tensor_matmul import QuantizedWeight


def same_shape_view(
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


def same_layout_as_strided(
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
            f"{type(tensor).__name__} only supports as_strided with unchanged "
            "shape, strides, and storage offset"
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
