"""Validated in-place updates and backend routing for ConvRot INT8 weights."""

import torch

from piper_kernels.weights._update import (
    narrow_rounding_seed,
    validate_add_operands,
    validate_addmm_operands,
    validate_real_scalar,
)
from piper_kernels.weights.convrot.int8._quantization import validate_storage

from . import _update_reference as reference

try:
    from . import triton as _triton_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _triton_backend = None


def _use_triton(value: torch.Tensor) -> bool:
    """Route bounded accelerator rows to Triton; keep CPU and wide rows on PyTorch."""
    return (
        _triton_backend is not None
        and value.device.type not in ("cpu", "meta")
        and value.numel() != 0
        and value.shape[-1] <= 16384
    )


def _validate_storage(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    *,
    operation: str,
) -> None:
    if qdata.ndim != 2:
        raise ValueError(f"{operation} requires 2-D qdata")
    validate_storage(qdata, scale, group_size, dtype)
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
    validate_addmm_operands(
        tuple(qdata.shape),
        mat1,
        mat2,
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
    validate_add_operands(
        tuple(qdata.shape),
        update,
        device=qdata.device,
        dtype=dtype,
        differentiable_storage=(scale,),
        operation=operation,
    )


def add_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    update: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
    rounding_seed: int | None = None,
) -> None:
    """Apply the logical ``beta * weight + alpha * update`` operation in place."""
    _validate_add(qdata, scale, dtype, group_size, update)
    operation = "ConvRot INT8 add_"
    beta_float = validate_real_scalar(beta, "beta", operation=operation)
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    seed = narrow_rounding_seed(rounding_seed, operation=operation)
    if (beta_float == 1 and alpha_float == 0) or qdata.numel() == 0:
        return
    if _use_triton(qdata):
        assert _triton_backend is not None
        _triton_backend.add_(
            qdata,
            scale,
            update,
            group_size,
            alpha_float,
            seed,
            beta=beta_float,
        )
    else:
        reference.add_(
            qdata,
            scale,
            update,
            group_size,
            beta_float,
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
    beta: float = 1.0,
    alpha: float = 1.0,
    rounding_seed: int | None = None,
) -> None:
    """Apply the logical ``beta * weight + alpha * mat1 @ mat2`` update in place."""
    _validate_addmm(qdata, scale, dtype, group_size, mat1, mat2)
    operation = "ConvRot INT8 addmm_"
    beta_float = validate_real_scalar(beta, "beta", operation=operation)
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    seed = narrow_rounding_seed(rounding_seed, operation=operation)
    if (beta_float == 1 and alpha_float == 0) or qdata.numel() == 0:
        return
    if _use_triton(qdata):
        assert _triton_backend is not None
        _triton_backend.addmm_(
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
