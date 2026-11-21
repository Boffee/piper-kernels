"""How traced graphs select one gate row per input row.

``gate[indices]`` traces to ``aten.index.Tensor``, which wraps negative indices as
Python does; ``gate.index_select(0, indices)`` traces to ``aten.index_select.default``,
which doesn't. The residual-update rewrite and the FFN fusions' two-update matchers
both match gated updates through this table, so they accept the same forms.
"""

from __future__ import annotations

import torch
from torch.fx.node import Argument

from piper_kernels.linear import _compile_fx as linear_fx

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


def selected_gate(node: Argument) -> tuple[torch.fx.Node, torch.fx.Node, bool] | None:  # noqa: PLR0911
    """Return ``(gate, indices, python_indexing)`` if ``node`` selects one gate row per index."""
    if not isinstance(node, torch.fx.Node):
        return None
    if node.target is _aten.index.Tensor:
        bound = linear_fx.bind_call(node, _aten.index.Tensor)
        if bound is None:
            return None
        gate = bound["self"]
        index_list = bound["indices"]
        if (
            not isinstance(gate, torch.fx.Node)
            or not isinstance(index_list, (list, tuple))
            or len(index_list) != 1
            or not isinstance(index_list[0], torch.fx.Node)
        ):
            return None
        return gate, index_list[0], True

    if node.target is _aten.index_select.default:
        bound = linear_fx.bind_call(node, _aten.index_select.default)
        if bound is None or bound["dim"] != 0:
            return None
        gate = bound["self"]
        indices = bound["index"]
        if not isinstance(gate, torch.fx.Node) or not isinstance(indices, torch.fx.Node):
            return None
        return gate, indices, False

    return None


__all__ = ["GATE_SELECTIONS", "selected_gate", "selection_arguments"]
