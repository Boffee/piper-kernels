"""Finite packed GGUF fixtures independent of a GGUF parser."""

from __future__ import annotations

import gguf
import numpy as np
import torch

from piper_kernels.gguf import GGUF_QUANT_SIZES, GGUFQuantizationType, logical_shape

_SCALE_OFFSETS = {
    GGUFQuantizationType.Q4_0: (0,),
    GGUFQuantizationType.Q4_1: (0, 2),
    GGUFQuantizationType.Q5_0: (0,),
    GGUFQuantizationType.Q5_1: (0, 2),
    GGUFQuantizationType.Q8_0: (0,),
    GGUFQuantizationType.Q2_K: (80, 82),
    GGUFQuantizationType.Q3_K: (108,),
    GGUFQuantizationType.Q4_K: (0, 2),
    GGUFQuantizationType.Q5_K: (0, 2),
    GGUFQuantizationType.Q6_K: (208,),
    GGUFQuantizationType.IQ4_NL: (0,),
    GGUFQuantizationType.IQ4_XS: (0,),
}


def finite_packed(
    quant_type: GGUFQuantizationType,
    *,
    rows: int = 2,
    features: int = 256,
) -> torch.Tensor:
    """Build valid-layout packed bytes whose floating scales are finite."""
    if quant_type in (
        GGUFQuantizationType.F32,
        GGUFQuantizationType.F16,
        GGUFQuantizationType.BF16,
    ):
        dtype = {
            GGUFQuantizationType.F32: torch.float32,
            GGUFQuantizationType.F16: torch.float16,
            GGUFQuantizationType.BF16: torch.bfloat16,
        }[quant_type]
        return torch.randn(rows, features, dtype=dtype).view(torch.uint8)

    block_size, type_size = GGUF_QUANT_SIZES[quant_type]
    raw = torch.randint(
        0,
        256,
        (rows, features // block_size * type_size),
        dtype=torch.uint8,
    )
    one = torch.tensor([0, 60], dtype=torch.uint8)
    for row in range(rows):
        for block in range(0, raw.shape[1], type_size):
            for offset in _SCALE_OFFSETS[quant_type]:
                raw[row, block + offset : block + offset + 2] = one
    return raw


def dequantize_reference(
    packed: torch.Tensor,
    quant_type: GGUFQuantizationType,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode a fixture with the official GGUF implementation."""
    dense = gguf.dequantize(
        packed.numpy(),
        gguf.GGMLQuantizationType(int(quant_type)),
    )
    return (
        torch.from_numpy(np.asarray(dense).copy())
        .reshape(logical_shape(tuple(packed.shape), quant_type))
        .to(dtype)
    )


__all__ = ["dequantize_reference", "finite_packed"]
