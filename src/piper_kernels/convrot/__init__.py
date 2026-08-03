"""Rotated quantized weights with transparent PyTorch linear dispatch."""

from ._int8.tensor import ConvRotInt8Tensor

__all__ = ["ConvRotInt8Tensor"]
