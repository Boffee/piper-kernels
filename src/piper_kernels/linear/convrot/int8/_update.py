"""Validated in-place updates for ConvRot INT8 weights."""

import torch

from piper_kernels.linear.convrot._update import (
    validate_real_scalar,
    validate_rounding_seed,
    validate_update_operands,
)

from . import _backend, _ops, reference


def _validate_storage(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    *,
    operation: str,
) -> None:
    reference.validate_storage(qdata, scale, group_size, dtype)
    if qdata.device.type == "meta":
        raise ValueError(f"{operation} cannot update a meta tensor without values")


def _validate_addmm(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    operation = "ConvRot INT8 addmm_"
    _validate_storage(qdata, scale, dtype, group_size, operation=operation)
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError(
            "ConvRot INT8 addmm_ matrices must be 2-D, "
            f"got shapes {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    expected_mat1 = (qdata.shape[0], mat2.shape[0])
    expected_mat2 = (mat1.shape[1], qdata.shape[1])
    if tuple(mat1.shape) != expected_mat1 or tuple(mat2.shape) != expected_mat2:
        raise ValueError(
            "ConvRot INT8 addmm_ shape mismatch: expected "
            f"mat1 {expected_mat1} and mat2 {expected_mat2} for weight {tuple(qdata.shape)}, "
            f"got {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    validate_update_operands(
        (mat1, mat2),
        device=qdata.device,
        dtype=dtype,
        differentiable_storage=(scale,),
        operation=operation,
    )


def _validate_add(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    update: torch.Tensor,
) -> None:
    operation = "ConvRot INT8 add_"
    _validate_storage(qdata, scale, dtype, group_size, operation=operation)
    if tuple(update.shape) != tuple(qdata.shape):
        raise ValueError(
            f"{operation} shape mismatch: expected update {tuple(qdata.shape)}, "
            f"got {tuple(update.shape)}"
        )
    validate_update_operands(
        (update,),
        device=qdata.device,
        dtype=dtype,
        differentiable_storage=(scale,),
        operation=operation,
    )


def _seed_argument(rounding_seed: int | None) -> int | None:
    return (
        rounding_seed
        if rounding_seed is None or rounding_seed < (1 << 63)
        else rounding_seed - (1 << 64)
    )


def add_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    update: torch.Tensor,
    *,
    alpha: int | float | complex = 1,
    rounding_seed: int | None = None,
) -> None:
    """Apply the logical ``weight + alpha * update`` operation in place."""
    _validate_add(qdata, scale, dtype, group_size, update)
    operation = "ConvRot INT8 add_"
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    validate_rounding_seed(rounding_seed, operation=operation)
    if alpha_float == 0:
        return
    seed = _seed_argument(rounding_seed)
    if _backend.select_add(qdata) is not None:
        _ops.add_(
            qdata,
            scale,
            update,
            group_size,
            alpha_float,
            seed,
        )
    else:
        reference.add_(
            qdata,
            scale,
            update,
            group_size,
            alpha_float,
            seed,
        )


def addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: int | float | complex = 1,
    alpha: int | float | complex = 1,
    rounding_seed: int | None = None,
) -> None:
    """Apply the logical ``beta * weight + alpha * mat1 @ mat2`` update in place."""
    _validate_addmm(qdata, scale, dtype, group_size, mat1, mat2)
    operation = "ConvRot INT8 addmm_"
    beta_float = validate_real_scalar(beta, "beta", operation=operation)
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    validate_rounding_seed(rounding_seed, operation=operation)
    if beta_float == 1 and alpha_float == 0:
        return
    seed = _seed_argument(rounding_seed)
    if _backend.select_addmm(qdata) is not None:
        _ops.addmm_(
            qdata,
            scale,
            mat1,
            mat2,
            group_size,
            beta_float,
            alpha_float,
            seed,
        )
    else:
        reference.addmm_(
            qdata,
            scale,
            mat1,
            mat2,
            group_size,
            beta_float,
            alpha_float,
            seed,
        )


__all__ = ["add_", "addmm_"]
