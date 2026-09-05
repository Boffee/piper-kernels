"""Validation and backend routing for plain and ConvRot NVFP4 updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from piper_kernels.linear.convrot._update import (
    validate_real_scalar,
    validate_rounding_seed,
    validate_update_operands,
)

from . import _layout, reference
from ._backend import supports_triton, triton_backend

if TYPE_CHECKING:
    from .tensor import PiperNVFP4Tensor


def _operation(weight: PiperNVFP4Tensor, name: str) -> str:
    prefix = "ConvRot NVFP4" if weight._update_group_size() else "NVFP4"
    return f"{prefix} {name}"


def _validate_storage(weight: PiperNVFP4Tensor, *, operation: str) -> None:
    if weight.device.type == "meta":
        raise ValueError(f"{operation} cannot update a meta tensor without values")
    if weight.ndim != 2:
        raise ValueError(f"{operation} requires a 2-D weight")
    rows, features = weight.shape
    if rows == 0 or features == 0 or features % _layout.BLOCK_SIZE:
        raise ValueError(f"{operation} requires nonempty block-16 storage")
    expected_qdata_shape = _layout.qdata_shape(rows, features)
    if weight.qdata.dtype is not torch.uint8 or tuple(weight.qdata.shape) != expected_qdata_shape:
        raise ValueError(
            f"{operation} requires packed uint8 qdata with shape "
            f"{expected_qdata_shape}, got {weight.qdata.dtype} {tuple(weight.qdata.shape)}"
        )
    if weight.block_size != _layout.BLOCK_SIZE:
        raise ValueError(
            f"{operation} requires block size {_layout.BLOCK_SIZE}, got {weight.block_size}"
        )
    expected_scale_shape = (
        _layout.scale_shape(rows, features)
        if weight.is_swizzled_scales
        else (rows, features // weight.block_size)
    )
    if (
        weight.scale.dtype is not torch.float8_e4m3fn
        or tuple(weight.scale.shape) != expected_scale_shape
    ):
        raise ValueError(
            f"{operation} requires canonical block-16 FP8 scales with shape "
            f"{expected_scale_shape}, got {weight.scale.dtype} {tuple(weight.scale.shape)}"
        )
    if not weight.qdata.is_contiguous() or not weight.scale.is_contiguous():
        raise ValueError(
            f"{operation} requires contiguous packed storage; "
            "a transposed weight cannot be updated in place"
        )
    if weight.per_tensor_scale is not None and weight.per_tensor_scale.numel() != 1:
        raise ValueError(f"{operation} requires a scalar per-tensor weight scale")
    storage = (
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
    )
    if any(value is not None and value.device != weight.device for value in storage):
        raise ValueError(f"{operation} storage tensors must share a device")


def _validate_addmm(
    weight: PiperNVFP4Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    operation = _operation(weight, "addmm_")
    _validate_storage(weight, operation=operation)
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError(
            f"{operation} matrices must be 2-D, "
            f"got shapes {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    expected_mat1 = (weight.shape[0], mat2.shape[0])
    expected_mat2 = (mat1.shape[1], weight.shape[1])
    if tuple(mat1.shape) != expected_mat1 or tuple(mat2.shape) != expected_mat2:
        raise ValueError(
            f"{operation} shape mismatch: expected "
            f"mat1 {expected_mat1} and mat2 {expected_mat2} for weight "
            f"{tuple(weight.shape)}, got {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    validate_update_operands(
        (mat1, mat2),
        device=weight.device,
        dtype=weight.orig_dtype,
        differentiable_storage=(weight.scale, weight.per_tensor_scale),
        operation=operation,
    )


def _validate_add(
    weight: PiperNVFP4Tensor,
    update: torch.Tensor,
) -> None:
    operation = _operation(weight, "add_")
    _validate_storage(weight, operation=operation)
    if tuple(update.shape) != tuple(weight.shape):
        raise ValueError(
            f"{operation} shape mismatch: expected update {tuple(weight.shape)}, "
            f"got {tuple(update.shape)}"
        )
    validate_update_operands(
        (update,),
        device=weight.device,
        dtype=weight.orig_dtype,
        differentiable_storage=(weight.scale, weight.per_tensor_scale),
        operation=operation,
    )


def addmm_(
    weight: PiperNVFP4Tensor,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: int | float | complex = 1,
    alpha: int | float | complex = 1,
    rounding_seed: int | None = None,
) -> None:
    """Apply a logical addmm update and refill the existing packed storage."""
    _validate_addmm(weight, mat1, mat2)
    operation = _operation(weight, "addmm_")
    beta_float = validate_real_scalar(beta, "beta", operation=operation)
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    validate_rounding_seed(rounding_seed, operation=operation)
    if beta_float == 1 and alpha_float == 0:
        return

    if supports_triton(weight.qdata):
        assert triton_backend is not None
        triton_backend.addmm_(
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
            _seed_argument(rounding_seed),
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
    alpha: int | float | complex = 1,
    rounding_seed: int | None = None,
) -> None:
    """Apply a logical dense update and refill the existing packed storage."""
    _validate_add(weight, update)
    operation = _operation(weight, "add_")
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    validate_rounding_seed(rounding_seed, operation=operation)
    if alpha_float == 0:
        return

    if supports_triton(weight.qdata):
        assert triton_backend is not None
        triton_backend.add_(
            weight.qdata,
            weight.scale,
            weight.per_tensor_scale,
            update,
            group_size,
            alpha_float,
            weight.is_swizzled_scales,
            weight.high_first,
            _seed_argument(rounding_seed),
        )
        return

    reference.add_(
        weight,
        group_size,
        update,
        alpha=alpha_float,
        rounding_seed=rounding_seed,
    )


def _seed_argument(seed: int | None) -> int | None:
    return seed if seed is None or seed < (1 << 63) else seed - (1 << 64)


__all__ = ["add_", "addmm_"]
