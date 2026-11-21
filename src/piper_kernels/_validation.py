"""Compile-safe scalar validation helpers shared across Piper Kernels.

Keep this module strictly free of Triton and TorchAO imports so portable CPU
and optional-dependency suites can import it anywhere.
"""

from __future__ import annotations

import math
from typing import cast

import torch


def _reject_non_real_scalar(
    value: object,
    message: str,
    *,
    type_message: str | None = None,
) -> None:
    if value is None or isinstance(value, (bool, complex)):
        error_prefix = message if type_message is None else type_message
        raise TypeError(f"{error_prefix}, got {value}")


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
    _reject_non_real_scalar(value, message, type_message=type_message)
    if torch.compiler.is_compiling() or isinstance(value, (torch.SymFloat, torch.SymInt)):
        converted = cast(float, torch.sym_float(value))
        torch._check(converted > -math.inf, lambda: message)
        torch._check(converted < math.inf, lambda: message)
        return converted
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{message}, got {value}")
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
    _reject_non_real_scalar(value, message, type_message=type_message)
    if torch.compiler.is_compiling() or isinstance(value, (torch.SymFloat, torch.SymInt)):
        converted = cast(float, torch.sym_float(value))
        torch._check(converted > 0.0, lambda: message)
        torch._check(converted < math.inf, lambda: message)
        return converted
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{message}, got {value}")
    return converted


def static_positive_finite_float(value: object) -> float | None:
    """Return a finite positive static Python float, or None for non-static/invalid values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted > 0.0 else None


__all__ = [
    "static_positive_finite_float",
    "validate_finite_float",
    "validate_positive_finite_float",
]
