"""Compute a fused attention output inside the residual add that reads it."""

from __future__ import annotations

import torch
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates import _graph as residual_graph

from . import _head_slices as head_slices

SOURCE_FILES = (__file__, head_slices.__file__, residual_graph.__file__)


def fold_residual_adds(
    graph: torch.fx.Graph,
    output: torch._ops.OpOverload,
    residual_add: torch._ops.OpOverload,
    head_slicing: head_slices.HeadSlicing | None = None,
) -> bool:
    """Compute each ``output`` call that only a residual add reads inside that add.

    ``residual_add`` takes the add's ``base``, ``gate``, ``gate_indices``, and
    ``python_indexing``, then ``output``'s arguments by name without ``output_dtype``,
    which is ``base``'s. With ``head_slicing``, an add whose K and V feed nothing else runs
    in head slices; see ``_head_slices``.
    """

    def compute(
        graph: torch.fx.Graph, update: torch.fx.Node, add: dict[str, Argument]
    ) -> torch.fx.Node:
        operands = residual_graph.bind_arguments(update)
        values = {
            **add,
            **{name: value for name, value in operands.items() if name != "output_dtype"},
        }
        sliced = (
            None
            if head_slicing is None
            else head_slices.add_in_head_slices(graph, residual_add, update, values, head_slicing)
        )
        return sliced or residual_graph.call_functional(graph, residual_add, values)

    return residual_graph.compute_inside_adds(graph, output, compute) > 0


__all__ = ["SOURCE_FILES", "fold_residual_adds"]
