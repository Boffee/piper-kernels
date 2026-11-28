"""Compile-safe scalar validation helpers shared across Piper Kernels.

Keep this module strictly free of Triton imports so portable CPU
and optional-dependency suites can import it anywhere.
"""

from __future__ import annotations

import decimal
import math
import numbers
from typing import cast

import torch

_MAX_FINITE_FLOAT_INT = (1 << 1024) - (1 << 970) - 1
_INT64_MAX = (1 << 63) - 1


def _require_real_scalar(
    value: object,
    message: str,
    *,
    type_message: str | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, numbers.Real, decimal.Decimal, torch.SymInt, torch.SymFloat)
    ):
        error_prefix = message if type_message is None else type_message
        got = (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
            if isinstance(value, torch.Tensor)
            else type(value).__name__
        )
        raise TypeError(f"{error_prefix}, got {got}")


def _to_compile_float(value: object, message: str) -> float:
    # Check integer bounds before torch.sym_float so huge int literals fail with
    # ValueError rather than OverflowError during Dynamo FX fake evaluation.
    if isinstance(value, int):
        torch._check(value >= -_MAX_FINITE_FLOAT_INT, lambda: message)
        torch._check(value <= _MAX_FINITE_FLOAT_INT, lambda: message)
        return cast(float, torch.sym_float(value))
    if isinstance(value, (float, torch.SymInt, torch.SymFloat)):
        return cast(float, torch.sym_float(value))
    return _to_eager_float(value, message)


def _to_eager_float(value: object, message: str) -> float:
    try:
        return float(cast(float, value))
    except OverflowError as exc:
        raise ValueError(f"{message}, got overflowing {type(value).__name__}") from exc


def validate_finite_float(
    value: float,
    message: str,
    *,
    type_message: str | None = None,
) -> float:
    """Validate and normalize a finite real scalar across eager and compiled runs.

    ``message`` must remain a static Python string (without interpolating
    ``value``) so Dynamo keeps ``SymFloat`` inputs symbolic instead of
    specializing on a formatted message string.
    """
    _require_real_scalar(value, message, type_message=type_message)
    if torch.compiler.is_compiling() or isinstance(value, (torch.SymFloat, torch.SymInt)):
        converted = _to_compile_float(value, message)
        torch._check(converted > -math.inf, lambda: message)
        torch._check(converted < math.inf, lambda: message)
        return converted
    converted = _to_eager_float(value, message)
    if not math.isfinite(converted):
        raise ValueError(f"{message}, got {converted}")
    return converted


def validate_positive_finite_float(
    value: float,
    message: str,
    *,
    type_message: str | None = None,
) -> float:
    """Validate and normalize a finite positive real scalar across eager and compiled runs.

    ``message`` must remain a static Python string (without interpolating
    ``value``) so Dynamo keeps ``SymFloat`` inputs symbolic instead of
    specializing on a formatted message string.
    """
    _require_real_scalar(value, message, type_message=type_message)
    if torch.compiler.is_compiling() or isinstance(value, (torch.SymFloat, torch.SymInt)):
        converted = _to_compile_float(value, message)
        torch._check(converted > 0.0, lambda: message)
        torch._check(converted < math.inf, lambda: message)
        return converted
    converted = _to_eager_float(value, message)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{message}, got {converted}")
    return converted


def validate_nonnegative_int(
    value: int,
    message: str,
    *,
    type_message: str | None = None,
) -> int:
    """Validate a nonnegative integer or SymInt across eager and compiled runs."""
    if isinstance(value, bool) or not isinstance(value, (int, torch.SymInt)):
        error_prefix = message if type_message is None else type_message
        got = (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
            if isinstance(value, torch.Tensor)
            else type(value).__name__
        )
        raise TypeError(f"{error_prefix}, got {got}")
    if torch.compiler.is_compiling() or isinstance(value, torch.SymInt):
        torch._check(value >= 0, lambda: message)
        torch._check(value <= _INT64_MAX, lambda: message)
        return cast(int, value)
    if value < 0 or value > _INT64_MAX:
        raise ValueError(f"{message}, got {value}")
    return value


def static_positive_finite_float(value: object) -> float | None:
    """Return a finite positive static Python float, or None for non-static/invalid values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    return converted if math.isfinite(converted) and converted > 0.0 else None


__all__ = [
    "static_positive_finite_float",
    "validate_finite_float",
    "validate_nonnegative_int",
    "validate_positive_finite_float",
]
