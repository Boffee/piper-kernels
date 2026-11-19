"""Public coarse-residual orchestration for sparse Piper Attention."""

from __future__ import annotations

from typing import Literal

import torch

from ._routing import _coarse_residual_from_mode
from ._routing_modes import (
    routing_mode_from_name,
)


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
    global_row_offset: int = 0,
) -> torch.Tensor:
    """Return a gated coarse branch using the selected block-summary policy.

    All token tensors use ``[batch,sequence,heads,features]``. Compact storage
    may end in one partial block. With ``block_lengths``, storage instead holds
    complete physical K64 blocks whose valid rows occupy each block's prefix.
    ``coarse_key_blocks`` defaults to every available K64 block.

    With ``global_row_offset``, the call computes only the slice of query rows that
    starts at that global row, placed as in ``SparsePiperAttention.forward``: Q and
    the gate hold the slice's rows, K/V and ``block_lengths`` describe the whole
    sequence, and the result has Q's rows.
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
        global_row_offset=global_row_offset,
    )


__all__ = ["sparse_piper_coarse_residual"]
