"""Fold a dense attention output into the residual add that consumes it."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates._graph import bind_arguments, call_functional

from . import output

_ATTENTION_RESIDUAL_ADD = (
    torch.ops.piper_kernels.convrot_int8_piper_projected_query_attention_residual_add_.default
)


def source_files() -> tuple[str, ...]:
    """Files whose operator contracts this fold depends on."""
    return (__file__, output.__file__)


def fold_residual_add(
    graph: torch.fx.Graph,
    attention: torch.fx.Node,
    residual_add: Mapping[str, Argument],
) -> torch.fx.Node:
    """Compute a dense attention output inside the residual add that is its only consumer.

    ``residual_add`` holds the add's ``base``, ``gate``, ``gate_indices``, and
    ``python_indexing``. Emits the attention residual-add operator, which takes the
    output's operands, at the graph's insertion point and returns the new ``base``. The
    FFN fusion matched the output and ``base`` with one dtype, so ``base`` supplies the
    output dtype.
    """
    operands = {
        name: value for name, value in bind_arguments(attention).items() if name != "output_dtype"
    }
    return call_functional(graph, _ATTENTION_RESIDUAL_ADD, {**operands, **residual_add})


__all__ = ["fold_residual_add", "source_files"]
