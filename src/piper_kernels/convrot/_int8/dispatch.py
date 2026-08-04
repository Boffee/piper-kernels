"""Internal backend selection for INT8 ConvRot operators."""

import math

import torch

from .reference import reference_addmm_, reference_linear, validate_storage

try:
    from .backends.triton import (
        triton_convrot_int8_addmm_ as _triton_addmm_,
    )
    from .backends.triton import (
        triton_convrot_int8_linear as _triton_linear,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_addmm_ = None
    _triton_linear = None


def _can_use_triton(activation: torch.Tensor, qdata: torch.Tensor) -> bool:
    return (
        _triton_linear is not None
        and activation.device.type == "cuda"
        and activation.device == qdata.device
        and activation.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and torch.cuda.get_device_capability(activation.device) >= (7, 5)
    )


def _validate_scalar(value: int | float | complex, name: str) -> float:
    if isinstance(value, complex):
        raise TypeError(f"ConvRot INT8 addmm_ {name} must be a real number, got {value}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"ConvRot INT8 addmm_ {name} must be finite, got {value}")
    return converted


def _validate_addmm(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    validate_storage(qdata, scale, group_size, dtype)
    if qdata.device.type == "meta":
        raise ValueError("ConvRot INT8 addmm_ cannot update a meta tensor without values")
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
    if mat1.device != qdata.device or mat2.device != qdata.device:
        raise ValueError(
            "ConvRot INT8 addmm_ weight and matrices must share a device, "
            f"got {qdata.device}/{mat1.device}/{mat2.device}"
        )
    if mat1.dtype is not dtype or mat2.dtype is not dtype:
        raise ValueError(
            "ConvRot INT8 addmm_ matrices must match the weight's logical dtype, "
            f"got {dtype}/{mat1.dtype}/{mat2.dtype}"
        )
    if mat1.layout is not torch.strided or mat2.layout is not torch.strided:
        raise ValueError("ConvRot INT8 addmm_ matrices must use strided layout")
    if torch.is_grad_enabled() and (mat1.requires_grad or mat2.requires_grad):
        raise RuntimeError(
            "ConvRot INT8 addmm_ does not support autograd; detach the matrices or use no_grad"
        )


def _can_use_triton_addmm(qdata: torch.Tensor, mat1: torch.Tensor) -> bool:
    return (
        _triton_addmm_ is not None
        and qdata.device.type == "cuda"
        and mat1.device == qdata.device
        and torch.cuda.get_device_capability(qdata.device) >= (7, 5)
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_addmm_",
    mutates_args=("qdata", "scale"),
)
def _convrot_int8_addmm_op(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
) -> None:
    if _can_use_triton_addmm(qdata, mat1):
        assert _triton_addmm_ is not None
        _triton_addmm_(qdata, scale, mat1, mat2, group_size, beta, alpha)
        return
    reference_addmm_(qdata, scale, mat1, mat2, group_size, beta, alpha)


@_convrot_int8_addmm_op.register_fake
def _convrot_int8_addmm_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _mat1: torch.Tensor,
    _mat2: torch.Tensor,
    _group_size: int,
    _beta: float,
    _alpha: float,
) -> None:
    return None


def _addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: int | float | complex = 1,
    alpha: int | float | complex = 1,
) -> None:
    """Apply the logical ``beta * weight + alpha * mat1 @ mat2`` update in place."""
    _validate_addmm(qdata, scale, dtype, group_size, mat1, mat2)
    beta_float = _validate_scalar(beta, "beta")
    alpha_float = _validate_scalar(alpha, "alpha")
    if beta_float == 1 and alpha_float == 0:
        return
    _convrot_int8_addmm_op(
        qdata,
        scale,
        mat1,
        mat2,
        group_size,
        beta_float,
        alpha_float,
    )


def _linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply raw ConvRot INT8 storage to a floating-point activation.

    ``qdata`` stores the two-dimensional INT8 weight in its rotated basis,
    and ``scale`` contains one float32 value per output channel. This is the
    internal storage-level ABI. Consumers should call
    :func:`torch.nn.functional.linear` with a ``ConvRotInt8Tensor`` weight.
    """
    validate_storage(qdata, scale, group_size, activation.dtype)
    if activation.ndim == 0 or activation.shape[-1] != qdata.shape[1]:
        actual = 0 if activation.ndim == 0 else activation.shape[-1]
        raise ValueError(f"ConvRot linear input has {actual} features, expected {qdata.shape[1]}")
    if activation.device != qdata.device:
        raise ValueError(
            "ConvRot activation and weight must share a device, "
            f"got {activation.device}/{qdata.device}"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(f"ConvRot linear bias must be a tensor or None, got {type(bias).__name__}")
    if _can_use_triton(activation, qdata):
        assert _triton_linear is not None
        return _triton_linear(activation, qdata, scale, bias, group_size)
    return reference_linear(activation, qdata, scale, group_size, bias)
