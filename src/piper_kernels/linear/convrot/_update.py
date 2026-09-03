"""Validation shared by in-place ConvRot quantized-weight updates."""

import math

import torch


def validate_real_scalar(
    value: int | float | complex,
    name: str,
    *,
    operation: str,
) -> float:
    """Convert a finite real update scalar while preserving useful errors."""
    if isinstance(value, complex):
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


__all__ = [
    "validate_real_scalar",
    "validate_rounding_seed",
    "validate_update_operands",
]
