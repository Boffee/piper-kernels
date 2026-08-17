"""Portable signed-Hadamard rotation for attention heads."""

import math

import torch

SUPPORTED_HEAD_DIMS = (64, 128)

# First 128 bits of SHA-256("piper-kernels/signed-hadamard/v1"), frozen as
# four compact bit masks. D64 uses the first two words. This makes the
# transform deterministic without allocating a sign tensor.
SIGNED_HADAMARD_MASK = (0xF62B39F3, 0x802576E6, 0x910D4CFD, 0x4F877B5A)


def rotate_signed_hadamard_heads(value: torch.Tensor) -> torch.Tensor:
    """Apply a fixed signed, normalized Hadamard to D64 or D128 heads."""
    head_dim = value.shape[-1]
    if head_dim not in SUPPORTED_HEAD_DIMS:
        supported = ", ".join(map(str, SUPPORTED_HEAD_DIMS))
        raise ValueError(
            f"signed Hadamard head dimension must be one of {supported}, got {head_dim}"
        )

    offsets = torch.arange(head_dim, device=value.device)
    words = torch.tensor(SIGNED_HADAMARD_MASK, device=value.device, dtype=torch.int64)
    signs = 2 * ((words[offsets // 32] >> (offsets % 32)) & 1) - 1
    rotated = value.float() * signs
    butterfly_distance = 1
    while butterfly_distance < head_dim:
        grouped = rotated.reshape(
            *rotated.shape[:-1],
            -1,
            2,
            butterfly_distance,
        )
        low, high = grouped.unbind(dim=-2)
        rotated = torch.stack((low + high, low - high), dim=-2).flatten(-3)
        butterfly_distance *= 2
    return rotated * (1.0 / math.sqrt(head_dim))
