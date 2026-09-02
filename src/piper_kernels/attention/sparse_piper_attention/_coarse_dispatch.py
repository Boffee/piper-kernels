"""Compiler-visible semantic dispatch for Q/K/V-derived coarse residuals."""

from __future__ import annotations

import torch

from ._routes import _DSA_ROUTING, _MEAN_POOL_ROUTING
from .coarse import validate_coarse_residual_inputs
from .dsa import _dsa_coarse_residual_impl
from .mean_pool import _mean_pool_coarse_residual_impl


def _routing_label(routing_mode: int) -> str:
    if routing_mode == _MEAN_POOL_ROUTING:
        return "mean-pool"
    if routing_mode == _DSA_ROUTING:
        return "DSA"
    raise ValueError(f"unsupported sparse Piper routing mode {routing_mode}")


@torch.library.custom_op(
    "piper_kernels::sparse_piper_coarse_residual",
    mutates_args=(),
)
def _sparse_piper_coarse_residual_op(
    fine_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compression_gate: torch.Tensor,
    sparse_key_blocks: int,
    coarse_key_blocks: int,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve a Q/K/V-derived coarse residual as one semantic operation."""
    _routing_label(routing_mode)
    implementation = (
        _mean_pool_coarse_residual_impl
        if routing_mode == _MEAN_POOL_ROUTING
        else _dsa_coarse_residual_impl
    )
    return implementation(
        fine_output,
        query,
        key,
        value,
        compression_gate,
        sparse_key_blocks=sparse_key_blocks,
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        block_lengths=block_lengths,
    )


@_sparse_piper_coarse_residual_op.register_fake
def _sparse_piper_coarse_residual_op_fake(
    fine_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compression_gate: torch.Tensor,
    sparse_key_blocks: int,
    coarse_key_blocks: int,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_coarse_residual_inputs(
        fine_output,
        query,
        key,
        value,
        compression_gate,
        sparse_key_blocks,
        coarse_key_blocks,
        coarse_scale,
        block_lengths,
        routing_label=_routing_label(routing_mode),
    )
    return torch.empty_like(fine_output, memory_format=torch.contiguous_format)
