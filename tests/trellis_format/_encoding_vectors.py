"""Encoder vectors: portable outlier-bearing inputs and SQNR quality floors.

``build_rows`` makes ConvRot-like rows from integer hashing, including 1% 4x
outliers and row magnitudes spanning two octaves, so the inputs are identical across
devices and PyTorch versions. ``min_sqnr_db`` bounds the post-decode rotated-basis
SQNR achieved with peak-anchored Stage 1 seeding across bit rates and block sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_MASK = 0xFFFFFFFF


def _hash(index: torch.Tensor, salt: int) -> torch.Tensor:
    """A 32-bit integer hash of ``index`` (int64, below 2^31); products stay below 2^63."""
    h = (index * 0x9E3779B1 + salt) & _MASK
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & _MASK
    h ^= h >> 12
    h = (h * 0x297A2D39) & _MASK
    return h ^ (h >> 15)


def build_rows(rows: int, cols: int, seed: int) -> torch.Tensor:
    """FP32 ``[rows, cols]`` rows like ConvRot-rotated weights: a sum of four uniforms
    per weight, 1% of weights four times larger, and row magnitudes over two octaves."""
    index = torch.arange(rows * cols, dtype=torch.int64)
    total = torch.zeros(rows * cols, dtype=torch.float64)
    for draw in range(4):
        total += _hash(index, seed * 7919 + draw).double() / 2**32
    values = (total - 2.0) * 3**0.5
    outlier = _hash(index, seed * 7919 + 4) % 100 == 0
    values = torch.where(outlier, 4 * values, values).view(rows, cols)
    magnitude = 0.02 * 2.0 ** ((torch.arange(rows, dtype=torch.float64) % 5 - 2) / 2)
    return (values * magnitude[:, None]).float()


@dataclass(frozen=True)
class EncoderVector:
    rows: int
    cols: int
    block_bits: int
    seed: int
    min_sqnr_db: float


# Cases, as (rows, cols, block_bits, seed, min_sqnr_db); the block size is the format's
# choice for cols: 4/5-bit steps at 4.5 bits per weight with scales, then 224-weight
# blocks at that rate; 4-bit steps; 3/4-bit steps at 3.5; 3-bit steps; 5/6-bit steps at
# 5.5; 6-bit steps; and 112-weight blocks, whose final overlap spans the whole sequence.
ENCODER_VECTORS = (
    EncoderVector(6, 512, 1144, 1, 22.5),
    EncoderVector(6, 448, 1000, 2, 24.0),
    EncoderVector(6, 512, 1024, 3, 22.0),
    EncoderVector(6, 512, 888, 4, 17.5),
    EncoderVector(6, 512, 768, 5, 15.1),
    EncoderVector(6, 512, 1400, 6, 30.0),
    EncoderVector(6, 512, 1536, 7, 30.5),
    EncoderVector(4, 336, 496, 8, 24.8),
)
