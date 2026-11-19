"""Distributed FX rewrites for fused ConvRot INT8 dense Piper attention."""

from __future__ import annotations

import torch

from piper_kernels.fusions.attention import _distributed_fold as distributed_fold

from . import (
    distributed_output as distributed_output_operators,  # noqa: F401 - registers distributed output ops
)

_ops = torch.ops.piper_kernels
_PROJECTED_QUERY_OUTPUT = _ops.convrot_int8_piper_projected_query_attention_output.default
_PROJECTED_QUERY_RS_RESIDUAL_ADD = (
    _ops.convrot_int8_piper_projected_query_attention_reduce_scatter_residual_add_.default
)
_PROJECTED_QUERY_RS = _ops.convrot_int8_piper_projected_query_attention_reduce_scatter.default


def fold_distributed_attention_output(graph: torch.fx.Graph) -> bool:
    """Fold ``projected_query_attention_output -> reduce_scatter -> (optional) residual_add_``."""
    return distributed_fold.fold_distributed_projected_output(
        graph,
        output_op=_PROJECTED_QUERY_OUTPUT,
        reduce_scatter_op=_PROJECTED_QUERY_RS,
        reduce_scatter_residual_op=_PROJECTED_QUERY_RS_RESIDUAL_ADD,
    )


__all__ = ["fold_distributed_attention_output"]
