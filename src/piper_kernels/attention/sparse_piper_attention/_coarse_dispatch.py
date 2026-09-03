"""Compiler-visible semantic dispatch for Q/K/V-derived coarse residuals."""

from __future__ import annotations

import torch

from ._routes import _ROUTING_NAME_BY_MODE, validate_routing_mode
from ._routing import coarse_residual_impl
from .coarse import validate_coarse_residual_inputs


def _routing_label(routing_mode: int) -> str:
    validate_routing_mode(routing_mode)
    return _ROUTING_NAME_BY_MODE[routing_mode]


@torch.library.custom_op(
    "piper_kernels::sparse_piper_coarse_residual",
    mutates_args=(),
)
def _sparse_piper_coarse_residual_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compression_gate: torch.Tensor,
    coarse_key_blocks: int,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve a Q/K/V-derived coarse residual as one semantic operation."""
    return coarse_residual_impl(
        query,
        key,
        value,
        compression_gate,
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        routing_mode=routing_mode,
        block_lengths=block_lengths,
    )


@_sparse_piper_coarse_residual_op.register_fake
def _sparse_piper_coarse_residual_op_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compression_gate: torch.Tensor,
    coarse_key_blocks: int,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_coarse_residual_inputs(
        query,
        key,
        value,
        compression_gate,
        coarse_key_blocks,
        coarse_scale,
        block_lengths,
        routing_label=_routing_label(routing_mode),
    )
    return torch.empty_like(query, memory_format=torch.contiguous_format)
