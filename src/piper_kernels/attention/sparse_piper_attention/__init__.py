"""Sparse Piper Attention with NVIDIA, AMD, and portable inference backends."""

from .coarse import (
    apply_coarse_attention_residual,
    coarse_attention,
    coarse_attention_residual,
    mean_pool_block_values,
    sparse_piper_coarse_residual,
)
from .dispatch import SparsePiperAttention

__all__ = [
    "SparsePiperAttention",
    "apply_coarse_attention_residual",
    "coarse_attention",
    "coarse_attention_residual",
    "mean_pool_block_values",
    "sparse_piper_coarse_residual",
]
