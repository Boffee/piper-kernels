"""Sparse Piper Attention for NVIDIA SM120 inference."""

from . import _coarse_dispatch as _coarse_dispatch
from .coarse import (
    apply_coarse_attention_residual,
    coarse_attention,
    coarse_attention_residual,
    mean_pool_block_values,
)
from .dispatch import SparsePiperAttention
from .residual import sparse_piper_coarse_residual

__all__ = [
    "SparsePiperAttention",
    "apply_coarse_attention_residual",
    "coarse_attention",
    "coarse_attention_residual",
    "mean_pool_block_values",
    "sparse_piper_coarse_residual",
]
