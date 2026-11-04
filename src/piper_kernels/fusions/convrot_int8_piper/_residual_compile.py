"""Fold a dense attention output into the residual add that consumes it."""

from __future__ import annotations

import torch

from piper_kernels.fusions.residual_updates._graph import ResidualAdd, bind_arguments, call_by_name

from . import output

_kernels = torch.ops.piper_kernels
_ATTENTION_OUTPUT = _kernels.convrot_int8_piper_projected_query_attention_output.default
_ATTENTION_RESIDUAL_ADD = (
    _kernels.convrot_int8_piper_projected_query_attention_residual_add_.default
)


def source_files() -> tuple[str, ...]:
    """Files whose operator contracts this fold depends on."""
    return (__file__, output.__file__)


def fold_residual_add(graph: torch.fx.Graph, add: ResidualAdd) -> bool:
    """Add an exclusive dense attention output into the residual as it is projected.

    The in-place operator takes the same attention operands. The residual add already
    requires its update's dtype to match ``base``, so ``base`` supplies the output dtype.
    """
    attention = add.update
    if attention.target is not _ATTENTION_OUTPUT or set(attention.users) != {add.node}:
        return False
    operands = {
        name: value for name, value in bind_arguments(attention).items() if name != "output_dtype"
    }
    with graph.inserting_before(add.node):
        call_by_name(
            graph,
            _ATTENTION_RESIDUAL_ADD,
            {
                **operands,
                "base": add.base,
                "gate": add.gate,
                "gate_indices": add.gate_indices,
                "python_indexing": add.python_indexing,
            },
        )
    return True


__all__ = ["fold_residual_add", "source_files"]
