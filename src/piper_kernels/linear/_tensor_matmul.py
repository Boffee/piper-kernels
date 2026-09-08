"""Matrix products that retain quantized linear semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
from torchao.utils import TorchAOBaseTensor

from ._tensor_views import unsupported_operation

if TYPE_CHECKING:
    from ._tensor_views import QuantizedWeight


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
    if alpha != 1 or beta != 1 or isinstance(alpha, complex) or isinstance(beta, complex):
        raise NotImplementedError("quantized addmm requires alpha=1 and beta=1")
    if tuple(bias.shape) != (canonical.shape[0],):
        raise NotImplementedError("quantized addmm requires one bias value per output feature")
    return torch.nn.functional.linear(activation, canonical, bias)


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


def register_matrix_ops(cls: type[TorchAOBaseTensor]) -> None:
    """Keep DTensor's transpose/mm/addmm decomposition on Piper's linear path."""
    aten = torch.ops.aten
    cls.implements([aten.mm.default, aten.matmul.default, aten.addmm.default])(_matrix_product)
    cls.implements_torch_function([torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__])(
        _torch_matmul
    )
    cls.implements([aten.bmm.default, aten._grouped_mm.default])(unsupported_operation)
