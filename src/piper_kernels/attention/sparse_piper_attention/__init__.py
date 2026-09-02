"""Sparse Piper Attention for NVIDIA SM120 inference."""

from .coarse import (
    apply_coarse_attention_residual,
    coarse_attention,
    coarse_attention_residual,
    mean_pool_block_values,
)
from .dispatch import SparsePiperAttention
from .dsa import dsa_coarse_residual
from .mean_pool import mean_pool_coarse_residual

__all__ = [
    "SparsePiperAttention",
    "apply_coarse_attention_residual",
    "coarse_attention",
    "coarse_attention_residual",
    "dsa_coarse_residual",
    "mean_pool_block_values",
    "mean_pool_coarse_residual",
]
