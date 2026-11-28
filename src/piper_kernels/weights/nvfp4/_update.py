"""Validation and backend routing for plain and ConvRot NVFP4 updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights._update import (
    narrow_rounding_seed,
    validate_add_operands,
    validate_addmm_operands,
    validate_real_scalar,
)
from piper_kernels.weights.nvfp4 import _layout

from . import _update_reference as reference

try:
    from . import triton as _triton_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _triton_backend = None


def _supports_triton(input: torch.Tensor) -> bool:  # noqa: A002
    """Return whether the fused NVFP4 update backend can execute on this device."""
    return _triton_backend is not None and AcceleratorTarget.from_device(
        input.device
    ).cuda_capability_at_least(10)


if TYPE_CHECKING:
    from piper_kernels.weights.nvfp4.tensor import PiperNVFP4Tensor


def _operation(weight: PiperNVFP4Tensor, name: str) -> str:
    prefix = "ConvRot NVFP4" if weight._update_group_size() else "NVFP4"
    return f"{prefix} {name}"


def _validate_storage(weight: PiperNVFP4Tensor, *, operation: str) -> None:
    _layout.validate_nvfp4_weight(
        weight,
        allowed_ndims=(2,),
        allow_empty=False,
        allow_meta=False,
        prefix=operation,
    )


def _validate_addmm(
    weight: PiperNVFP4Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    operation = _operation(weight, "addmm_")
    _validate_storage(weight, operation=operation)
    validate_addmm_operands(
        tuple(weight.shape),
        mat1,
        mat2,
        device=weight.device,
        dtype=weight.dtype,
        differentiable_storage=(weight.scale, weight.per_tensor_scale),
        operation=operation,
    )


def _validate_add(
    weight: PiperNVFP4Tensor,
    update: torch.Tensor,
) -> None:
    operation = _operation(weight, "add_")
    _validate_storage(weight, operation=operation)
    validate_add_operands(
        tuple(weight.shape),
        update,
        device=weight.device,
        dtype=weight.dtype,
        differentiable_storage=(weight.scale, weight.per_tensor_scale),
        operation=operation,
    )


def addmm_(
    weight: PiperNVFP4Tensor,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
    rounding_seed: int | None = None,
) -> None:
    """Apply a logical addmm update and refill the existing packed storage."""
    _validate_addmm(weight, mat1, mat2)
    operation = _operation(weight, "addmm_")
    beta_float = validate_real_scalar(beta, "beta", operation=operation)
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    seed = narrow_rounding_seed(rounding_seed, operation=operation)
    if beta_float == 1 and alpha_float == 0:
        return

    if _supports_triton(weight.qdata):
        assert _triton_backend is not None
        _triton_backend.addmm_(
            weight.qdata,
            weight.scale,
            weight.per_tensor_scale,
            mat1,
            mat2,
            group_size,
            beta_float,
            alpha_float,
            weight.is_swizzled_scales,
            weight.high_first,
            seed,
        )
        return

    reference.addmm_(
        weight,
        group_size,
        mat1,
        mat2,
        beta=beta_float,
        alpha=alpha_float,
        rounding_seed=rounding_seed,
    )


def add_(
    weight: PiperNVFP4Tensor,
    group_size: int,
    update: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
    rounding_seed: int | None = None,
) -> None:
    """Apply a logical dense update and refill the existing packed storage."""
    _validate_add(weight, update)
    operation = _operation(weight, "add_")
    beta_float = validate_real_scalar(beta, "beta", operation=operation)
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    seed = narrow_rounding_seed(rounding_seed, operation=operation)
    if beta_float == 1 and alpha_float == 0:
        return

    if _supports_triton(weight.qdata):
        assert _triton_backend is not None
        _triton_backend.add_(
            weight.qdata,
            weight.scale,
            weight.per_tensor_scale,
            update,
            group_size,
            alpha_float,
            weight.is_swizzled_scales,
            weight.high_first,
            seed,
            beta=beta_float,
        )
        return

    reference.add_(
        weight,
        group_size,
        update,
        beta=beta_float,
        alpha=alpha_float,
        rounding_seed=rounding_seed,
    )


__all__ = ["add_", "addmm_"]
