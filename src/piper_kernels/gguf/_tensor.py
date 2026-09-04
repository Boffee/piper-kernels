"""Normalize packed GGUF tensors for conversion kernels."""

from __future__ import annotations

import torch

from ._types import GGUFQuantizationType, logical_shape, normalize_quant_type


def prepare_packed_matrix(
    data: torch.Tensor,
    quant_type: int | None,
) -> tuple[torch.Tensor, GGUFQuantizationType, int, int]:
    """Return contiguous bytes, normalized type, and the logical matrix shape."""
    if quant_type is None:
        quant_type = getattr(data, "quant_type", None)
    if quant_type is None:
        raise TypeError("GGUF conversion requires quant_type or data.quant_type")
    if data.ndim != 2:
        raise ValueError(f"GGUF conversion requires a matrix, got {data.ndim} dimensions")

    normalized = normalize_quant_type(quant_type)
    source = data.as_subclass(torch.Tensor) if type(data) is not torch.Tensor else data
    raw = source.detach().contiguous().view(torch.uint8)
    rows, features = logical_shape(tuple(raw.shape), normalized)
    return raw, normalized, rows, features


__all__ = ["prepare_packed_matrix"]
