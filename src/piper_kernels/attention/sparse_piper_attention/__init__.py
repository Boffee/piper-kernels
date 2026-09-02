"""Sparse Piper Attention for NVIDIA SM120 inference."""

from .coarse import (
    apply_coarse_attention_residual,
    coarse_attention,
    coarse_attention_residual,
    mean_pool_block_values,
)
from .dispatch import SparsePiperAttention

__all__ = [
    "SparsePiperAttention",
    "apply_coarse_attention_residual",
    "coarse_attention",
    "coarse_attention_residual",
    "mean_pool_block_values",
]
