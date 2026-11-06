"""Compute a fused attention output inside the residual add that reads it."""

from __future__ import annotations

import torch
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates import _graph as residual_graph

SOURCE_FILES = (__file__, residual_graph.__file__)


def _arguments(operands: dict[str, Argument], add: dict[str, Argument]) -> dict[str, Argument]:
    # The add accepted the output as an update of ``base``'s dtype, so ``base`` supplies it.
    return {**add, **{name: value for name, value in operands.items() if name != "output_dtype"}}


def fold_residual_adds(
    graph: torch.fx.Graph,
    output: torch._ops.OpOverload,
    residual_add: torch._ops.OpOverload,
) -> bool:
    """Compute each ``output`` call that only a residual add reads inside that add.

    ``residual_add`` takes the add's ``base``, ``gate``, ``gate_indices``, and
    ``python_indexing``, then ``output``'s arguments by name without ``output_dtype``.
    """
    return residual_graph.compute_inside_adds(graph, output, residual_add, _arguments) > 0


__all__ = ["SOURCE_FILES", "fold_residual_adds"]
