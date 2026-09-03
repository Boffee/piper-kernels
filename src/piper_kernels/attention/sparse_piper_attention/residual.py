"""Public coarse-residual orchestration for sparse Piper Attention."""

from __future__ import annotations

from typing import Literal

import torch

from ._routes import routing_mode_from_name
from ._routing import _coarse_residual_from_mode


def sparse_piper_coarse_residual(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coarse_gate: torch.Tensor,
    *,
    routing: Literal["mean", "minmax"],
    coarse_key_blocks: int | None = None,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a gated coarse branch using the selected block-summary policy.

    All token tensors use ``[batch,sequence,heads,features]``. Compact storage
    may end in one partial block. With ``block_lengths``, storage instead holds
    complete physical K64 blocks whose valid rows occupy each block's prefix.
    ``coarse_key_blocks`` defaults to every available K64 block.
    """
    return _coarse_residual_from_mode(
        query,
        key,
        value,
        coarse_gate,
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        routing_mode=routing_mode_from_name(routing),
        block_lengths=block_lengths,
    )


__all__ = ["sparse_piper_coarse_residual"]
