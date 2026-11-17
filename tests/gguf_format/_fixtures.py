"""Finite packed GGUF inputs and independently decoded reference tensors."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import torch

from piper_kernels.gguf import GGUF_QUANT_SIZES, GGUFQuantizationType

REFERENCE_PATH = Path(__file__).with_name("reference.zip")

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


def reference_key(quant_type: GGUFQuantizationType, *, rows: int, features: int, seed: int) -> str:
    return f"{quant_type.name}-{rows}x{features}-seed{seed}"


def reference_case(
    quant_type: GGUFQuantizationType, *, seed: int, rows: int = 2, features: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load packed rows and their upstream-decoded FP32 values."""
    key = reference_key(quant_type, rows=rows, features=features, seed=seed)
    with ZipFile(REFERENCE_PATH) as archive:
        packed = torch.frombuffer(bytearray(archive.read(f"{key}/packed")), dtype=torch.uint8)
        dense = torch.frombuffer(
            bytearray(archive.read(f"{key}/decoded-fp32")), dtype=torch.float32
        )
    return packed.reshape(rows, -1), dense.reshape(rows, features)
