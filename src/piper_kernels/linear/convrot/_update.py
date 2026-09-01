"""Validation shared by in-place ConvRot quantized-weight updates."""

import math


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


__all__ = ["validate_real_scalar", "validate_rounding_seed"]
