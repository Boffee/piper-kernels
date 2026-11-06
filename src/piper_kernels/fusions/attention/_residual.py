"""Compute a fused attention output inside the residual add that reads it."""

from __future__ import annotations

import torch
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates import _graph as residual_graph

from . import _head_groups as head_groups

SOURCE_FILES = (__file__, head_groups.__file__, residual_graph.__file__)


def fold_residual_adds(
    graph: torch.fx.Graph,
    output: torch._ops.OpOverload,
    residual_add: torch._ops.OpOverload,
    head_split: head_groups.HeadSplit | None = None,
) -> bool:
    """Compute each ``output`` call that only a residual add reads inside that add.

    ``residual_add`` takes the add's ``base``, ``gate``, ``gate_indices``, and
    ``python_indexing``, then ``output``'s arguments by name without ``output_dtype``,
    which is ``base``'s. With ``head_split``, an add whose K and V feed nothing else runs
    in head groups; see ``_head_groups``.
    """

    def compute(
        graph: torch.fx.Graph, update: torch.fx.Node, add: dict[str, Argument]
    ) -> torch.fx.Node:
        operands = residual_graph.bind_arguments(update)
        values = {
            **add,
            **{name: value for name, value in operands.items() if name != "output_dtype"},
        }
        grouped = (
            None
            if head_split is None
            else head_groups.add_in_head_groups(graph, residual_add, update, values, head_split)
        )
        return grouped or residual_graph.call_functional(graph, residual_add, values)

    return residual_graph.compute_inside_adds(graph, output, compute) > 0


__all__ = ["SOURCE_FILES", "fold_residual_adds"]
