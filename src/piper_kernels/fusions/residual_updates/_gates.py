"""How traced graphs select one gate row per input row.

``gate[indices]`` traces to ``aten.index.Tensor``, which wraps negative indices as
Python does; ``gate.index_select(0, indices)`` traces to ``aten.index_select.default``,
which doesn't. The residual-update rewrite and the FFN fusions' two-update patterns both
match gated updates through this table, so they accept the same forms.
"""

from __future__ import annotations

from typing import cast

import torch
from torch.fx.node import Argument

_aten = torch.ops.aten

# Each traced gate-row selection and whether its indices wrap as Python indexing does.
GATE_SELECTIONS: dict[torch._ops.OpOverload, bool] = {
    _aten.index.Tensor: True,
    _aten.index_select.default: False,
}


def selection_arguments(
    selection: torch._ops.OpOverload, gate: object, indices: object
) -> tuple[object, ...]:
    """The arguments with which ``selection`` picks ``gate``'s rows by ``indices``."""
    return (gate, [indices]) if selection is _aten.index.Tensor else (gate, 0, indices)


def selected_gate(node: Argument) -> tuple[torch.fx.Node, torch.fx.Node, bool] | None:
    """Return ``(gate, indices, python_indexing)`` if ``node`` selects one gate row per index."""
    if (
        not isinstance(node, torch.fx.Node)
        or node.target not in GATE_SELECTIONS
        or node.kwargs
        or not node.args
    ):
        return None
    selection = cast(torch._ops.OpOverload, node.target)
    gate, last = node.args[0], node.args[-1]
    indices = last[0] if isinstance(last, (list, tuple)) and len(last) == 1 else last
    if (
        not isinstance(gate, torch.fx.Node)
        or not isinstance(indices, torch.fx.Node)
        or tuple(node.args) != selection_arguments(selection, gate, indices)
    ):
        return None
    return gate, indices, GATE_SELECTIONS[selection]


__all__ = ["GATE_SELECTIONS", "selected_gate", "selection_arguments"]
