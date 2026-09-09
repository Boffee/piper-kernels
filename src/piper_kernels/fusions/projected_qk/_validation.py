"""Head-width validation shared by projected Q/K operators."""

import torch


def resolve_head_dim(norm_weight: torch.Tensor | None, head_dim: int | None) -> int:
    """Infer the head width from an affine norm, or require it for a weightless norm."""
    if head_dim is None:
        if norm_weight is None:
            raise ValueError("Weightless RMSNorm requires an explicit head_dim")
        head_dim = norm_weight.shape[0] if norm_weight.ndim == 1 else 0
    if (
        isinstance(head_dim, bool)
        or not isinstance(head_dim, (int, torch.SymInt))
        or head_dim not in (64, 128)
    ):
        raise ValueError("Q/K projection requires head_dim=64 or 128")
    if norm_weight is not None and norm_weight.shape != (head_dim,):
        raise ValueError("Q/K projection RMSNorm weight must match head_dim")
    return head_dim
