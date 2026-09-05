"""Bias dtype policy shared by Piper linear operators and fusions."""

from __future__ import annotations

import torch

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def is_supported_dtype(dtype: torch.dtype) -> bool:
    """Return whether a bias dtype is supported by Piper linear epilogues."""
    return dtype in SUPPORTED_DTYPES


def validate_dtype(bias: torch.Tensor, name: str) -> None:
    """Require a floating bias dtype supported by every Piper linear epilogue."""
    if not is_supported_dtype(bias.dtype):
        raise ValueError(f"{name} bias dtype must be FP16, BF16, or FP32, got {bias.dtype}")


__all__ = ["SUPPORTED_DTYPES", "is_supported_dtype", "validate_dtype"]
