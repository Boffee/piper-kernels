"""Matrix views and products that retain quantized linear semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import TorchAOBaseTensor

if TYPE_CHECKING:
    from .convrot.int8.tensor import ConvRotInt8Tensor
    from .nvfp4.tensor import PiperNVFP4Tensor

    type QuantizedWeight = ConvRotInt8Tensor | PiperNVFP4Tensor


def require_untransposed(weight: QuantizedWeight, operation: str) -> None:
    """Reject operations that would interpret a transpose as canonical storage."""
    if weight.transposed:
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} does not support a transposed weight"
        )


def _transpose(
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


def _matrix_product(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    has_bias = func is torch.ops.aten.addmm.default
    activation, weight = args[1:3] if has_bias else args[:2]
    if (
        not isinstance(activation, torch.Tensor)
        or isinstance(activation, TorchAOBaseTensor)
        or not isinstance(weight, TorchAOBaseTensor)
        or weight.ndim != 2
        or not getattr(weight, "transposed", False)
    ):
        raise NotImplementedError(
            f"{func} requires a dense activation and a transposed Piper quantized weight"
        )
    if activation.ndim != 2 and func is not torch.ops.aten.matmul.default:
        raise ValueError(f"{func} requires a 2-D activation")
    canonical = cast("QuantizedWeight", weight)._transpose()
    if not has_bias:
        return torch.nn.functional.linear(activation, canonical)

    bias = args[0]
    alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 1)
    if isinstance(alpha, complex) or isinstance(beta, complex):
        raise NotImplementedError("quantized addmm requires real alpha and beta")
    output_shape = (activation.shape[0], canonical.shape[0])
    if torch.broadcast_shapes(bias.shape, output_shape) != output_shape:
        raise ValueError("quantized addmm bias must broadcast to the matrix product shape")
    if alpha == 1 and beta == 1 and tuple(bias.shape) == (canonical.shape[0],):
        return torch.nn.functional.linear(activation, canonical, bias)
    result = torch.nn.functional.linear(activation, canonical)
    if alpha != 1:
        result = result * alpha
    # In particular, beta=0 must not propagate a NaN or infinity in the bias.
    return result if beta == 0 else result + bias * beta


def _unsupported(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    _args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> torch.Tensor:
    raise NotImplementedError(f"Piper quantized weights do not support {func}")


def _matmul(
    input: torch.Tensor,  # noqa: A002 - match torch.matmul keyword arguments
    other: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        raise NotImplementedError("Piper quantized matmul does not support out")
    return _matrix_product(torch.ops.aten.matmul.default, (), (input, other), {})


def _torch_matmul(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    # Dynamo's generic batched matmul decomposition would expand the quantized
    # weight into a batch. Preserve a single canonical linear instead.
    return _matmul(*args, **kwargs)


def _contiguous(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast("QuantizedWeight", args[0])
    require_untransposed(weight, "contiguous")
    return weight._apply_fn_to_data(lambda value: func(value, *args[1:], **kwargs))


def _clone(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast("QuantizedWeight", args[0])
    if kwargs.get("memory_format") not in (None, torch.preserve_format):
        require_untransposed(weight, "clone with a different memory format")
    return weight._apply_fn_to_data(lambda value: func(value, **kwargs))


def register_matrix_ops(cls: type[TorchAOBaseTensor]) -> None:
    """Keep DTensor's transpose/mm/addmm decomposition on Piper's linear path."""
    aten = torch.ops.aten
    cls.implements([aten.t.default, aten.transpose.int, aten.permute.default])(_transpose)
    cls.implements([aten.mm.default, aten.matmul.default, aten.addmm.default])(_matrix_product)
    cls.implements_torch_function([torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__])(
        _torch_matmul
    )
    cls.implements(aten.clone.default)(_clone)
    cls.implements(aten.contiguous.default)(_contiguous)
    cls.implements_torch_function(torch.Tensor.contiguous)(_contiguous)
    # TorchAO's inherited slicing handlers construct a base NVFP4Tensor. Until
    # there is a packing/rotation-aware implementation, fail before losing data
    # interpretation (including during a DTensor redistribution).
    cls.implements(
        [
            aten.slice.Tensor,
            aten.select.int,
            aten.bmm.default,
            aten._grouped_mm.default,
        ]
    )(_unsupported)
