"""Validation and dispatch shared by in-place quantized-weight updates."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from torchao.utils import TorchAOBaseTensor

from piper_kernels.stochastic_quantization import signed_seed
from piper_kernels.weights._views import require_untransposed

if TYPE_CHECKING:
    from piper_kernels.weights._dispatch import QuantizedWeight

_WeightT = TypeVar("_WeightT", bound="QuantizedWeight")


def validate_real_scalar(
    value: float,
    name: str,
    *,
    operation: str,
) -> float:
    """Normalize a real coefficient and reject invalid runtime inputs at the boundary."""
    if value is None or isinstance(value, complex):
        raise TypeError(f"{operation} {name} must be a real number, got {value}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{operation} {name} must be finite, got {value}")
    return converted


def validate_rounding_seed(
    rounding_seed: int | None,
    *,
    operation: str,
) -> None:
    """Validate an optional seed accepted by PyTorch device generators."""
    if rounding_seed is None:
        return
    if isinstance(rounding_seed, bool) or not isinstance(rounding_seed, int):
        raise TypeError(f"{operation} rounding_seed must be an unsigned 64-bit integer")
    if not 0 <= rounding_seed < (1 << 64):
        raise ValueError(f"{operation} rounding_seed must be an unsigned 64-bit integer")


def narrow_rounding_seed(
    rounding_seed: int | None,
    *,
    operation: str,
) -> int | None:
    """Validate a seed and narrow it to the int64 an operator schema carries."""
    validate_rounding_seed(rounding_seed, operation=operation)
    if rounding_seed is None:
        return None
    return signed_seed(rounding_seed)


def validate_update_operands(
    operands: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    differentiable_storage: tuple[torch.Tensor | None, ...],
    operation: str,
) -> None:
    """Validate operands shared by in-place quantized-weight updates."""
    if any(operand.device != device for operand in operands):
        devices = "/".join((str(device), *(str(operand.device) for operand in operands)))
        raise ValueError(f"{operation} weight and inputs must share a device, got {devices}")
    if any(operand.dtype is not dtype for operand in operands):
        dtypes = "/".join(str(operand.dtype) for operand in operands)
        raise ValueError(
            f"{operation} inputs must match the weight's logical dtype, got {dtype}/{dtypes}"
        )
    if any(operand.layout is not torch.strided for operand in operands):
        raise ValueError(f"{operation} inputs must use strided layout")
    differentiable = (*differentiable_storage, *operands)
    if torch.is_grad_enabled() and any(
        value is not None and value.requires_grad for value in differentiable
    ):
        raise RuntimeError(
            f"{operation} does not support autograd; detach its inputs or use no_grad"
        )


def validate_addmm_operands(
    weight_shape: tuple[int, ...],
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    differentiable_storage: tuple[torch.Tensor | None, ...],
    operation: str,
) -> None:
    """Validate matrix dimensions and shared operand metadata for ``addmm_``."""
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError(
            f"{operation} matrices must be 2-D, "
            f"got shapes {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    expected_mat1 = (weight_shape[0], mat2.shape[0])
    expected_mat2 = (mat1.shape[1], weight_shape[1])
    if tuple(mat1.shape) != expected_mat1 or tuple(mat2.shape) != expected_mat2:
        raise ValueError(
            f"{operation} shape mismatch: expected "
            f"mat1 {expected_mat1} and mat2 {expected_mat2} for weight "
            f"{weight_shape}, got {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    validate_update_operands(
        (mat1, mat2),
        device=device,
        dtype=dtype,
        differentiable_storage=differentiable_storage,
        operation=operation,
    )


def validate_add_operands(
    weight_shape: tuple[int, ...],
    update: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    differentiable_storage: tuple[torch.Tensor | None, ...],
    operation: str,
) -> None:
    """Validate update shape and shared operand metadata for ``add_``."""
    if tuple(update.shape) != weight_shape:
        raise ValueError(
            f"{operation} shape mismatch: expected update {weight_shape}, got {tuple(update.shape)}"
        )
    validate_update_operands(
        (update,),
        device=device,
        dtype=dtype,
        differentiable_storage=differentiable_storage,
        operation=operation,
    )


def run_addmm_(
    weight: _WeightT,
    mat1: object,
    mat2: object,
    *,
    operation: str,
    apply: Callable[[torch.Tensor, torch.Tensor], None],
) -> _WeightT:
    """Validate boundary preconditions for ``Tensor.addmm_`` and run ``apply``."""
    require_untransposed(weight, "addmm_")
    if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
        raise TypeError(f"{operation} matrices must be tensors")
    apply(mat1, mat2)
    return weight


def run_add_(
    weight: _WeightT,
    other: object,
    *,
    operation: str,
    apply: Callable[[torch.Tensor], None],
) -> _WeightT:
    """Validate boundary preconditions for ``Tensor.add_`` and run ``apply``."""
    require_untransposed(weight, "add_")
    if not isinstance(other, torch.Tensor):
        raise TypeError(f"{operation} update must be a tensor")
    apply(other)
    return weight


def register_update_ops(cls: type[TorchAOBaseTensor], *, format_name: str) -> None:
    """Register ``aten.addmm_.default`` and ``aten.add_.Tensor`` to delegate to methods."""

    def _addmm_dispatch(
        _func: Callable[..., torch.Tensor],
        _types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        weight, mat1, mat2 = args
        if not isinstance(weight, cls):
            raise TypeError(
                f"{format_name} addmm_ weight must be {cls.__name__}, got {type(weight)}"
            )
        return cast("QuantizedWeight", weight).addmm_(
            mat1, mat2, beta=kwargs.get("beta", 1), alpha=kwargs.get("alpha", 1)
        )

    def _add_dispatch(
        _func: Callable[..., torch.Tensor],
        _types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        weight, update = args
        if not isinstance(weight, cls):
            raise TypeError(f"{format_name} add_ weight must be {cls.__name__}, got {type(weight)}")
        return cast("QuantizedWeight", weight).add_(update, alpha=kwargs.get("alpha", 1))

    cls.implements(torch.ops.aten.addmm_.default)(_addmm_dispatch)
    cls.implements(torch.ops.aten.add_.Tensor)(_add_dispatch)


__all__ = [
    "narrow_rounding_seed",
    "register_update_ops",
    "run_add_",
    "run_addmm_",
    "validate_add_operands",
    "validate_addmm_operands",
    "validate_real_scalar",
    "validate_rounding_seed",
    "validate_update_operands",
]
