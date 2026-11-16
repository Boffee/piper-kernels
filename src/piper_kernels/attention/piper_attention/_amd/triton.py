"""Prepare dense per-token V1024-centered values in RDNA4's K64 WMMA layout."""

import torch

from .._quantization import _quantize_value_k1024_kernel, prepare_chunk_centered_value

# Re-exported for compiler/provider diagnostics.
_prepare_value_kernel = _quantize_value_k1024_kernel


def prepare_value(
    value: torch.Tensor,
    *,
    storage_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center V per 1024-token chunk, quantize per token, and pack RDNA4 WMMA tiles."""
    return prepare_chunk_centered_value(
        value,
        storage_key_length=storage_length,
        store_log_scale=True,
        scale_storage_length=storage_length,
        pack_wmma=True,
        log_scale_dtype=torch.float32,
    )
