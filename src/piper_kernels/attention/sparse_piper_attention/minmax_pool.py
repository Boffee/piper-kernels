"""Min/max-pool public helpers for sparse Piper Attention."""

from __future__ import annotations

import torch

from ._routes import _MINMAX_ROUTING
from ._routing import coarse_residual


def minmax_pool_coarse_residual(
    fine_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compression_gate: torch.Tensor,
    *,
    sparse_key_blocks: int,
    coarse_key_blocks: int | None = None,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a min/max-pool coarse branch over a K64 prefix.

    All token tensors use ``[batch,sequence,heads,features]``. Compact storage
    may end in one partial block. With ``block_lengths``, storage instead holds
    complete physical K64 blocks whose valid rows occupy each block's prefix.
    ``coarse_key_blocks`` defaults to ``sparse_key_blocks`` and may extend the
    coarse branch across the following dense-key blocks.
    """
    return coarse_residual(
        fine_output,
        query,
        key,
        value,
        compression_gate,
        sparse_key_blocks=sparse_key_blocks,
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        routing_mode=_MINMAX_ROUTING,
        block_lengths=block_lengths,
    )


__all__ = ["minmax_pool_coarse_residual"]
