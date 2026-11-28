"""Rewrite a copy-back block's residual sums as adds into the input it overwrites.

A compiled block whose program ends with ``hidden.copy_(result)`` declares that the
caller no longer needs ``hidden``. When ``result`` is built from ``hidden`` by gated
residual sums, as in::

    added = hidden + gate[indices] * update
    result = added + ffn_gate[indices] * ffn_update
    hidden.copy_(result)

the rewrite expresses each sum as an add into its base, emitted as a functional call
so the graph stays functional::

    added = residual_add(hidden, update, gate, indices)
    result = residual_add(added, ffn_update, ffn_gate, indices)
    hidden.copy_(result)

Inductor's reinplacing pass, which runs after every custom pass and tracks every view
of ``hidden``, then adds into ``hidden`` in place when nothing reads its original value
afterward; otherwise it copies ``hidden`` first. Once every add is in place, the block's
new full-size outputs are gone.

The rewrite runs as Phase 0 of the unified post-grad pass and names no fusion. A fusion
whose operator has a residual-add counterpart finds the adds that alone read its output
with ``_graph.compute_inside_adds`` and computes that output inside the add, which also
drops the full-size update. Unmatched graphs, including those on devices Triton doesn't
support, are unchanged. Debug logging here and in ``_graph`` reports what was rewritten,
and Inductor's reinplacing log reports any input it had to copy.
"""

from __future__ import annotations

import logging

import torch
from torch.fx.node import Argument

from piper_kernels.linear import _compile_fx as linear_fx

from . import _gates, _ops
from ._graph import RESIDUAL_ADD, call_functional

_log = logging.getLogger(__name__)

_aten = torch.ops.aten
_COPY = _aten.copy_.default


def _gated_sum(node: torch.fx.Node) -> dict[str, Argument] | None:
    """Return ``residual_add_``'s arguments if ``node`` is ``base + gate[indices] * update``."""
    add_args = linear_fx.binary_args(node, _aten.add.Tensor)
    if add_args is None:
        return None
    base, product = add_args
    mul_args = linear_fx.binary_args(product, _aten.mul.Tensor)
    if mul_args is None:
        return None
    for selection, update in (mul_args, mul_args[::-1]):
        selected = _gates.selected_gate(selection)
        if selected is None:
            continue
        gate, indices, python_indexing = selected
        operands = (
            linear_fx.node_tensor_metadata(base),
            linear_fx.node_tensor_metadata(update),
            linear_fx.node_tensor_metadata(gate),
            linear_fx.node_tensor_metadata(indices),
        )
        base_value, update_value, gate_value, indices_value = operands
        if (
            base_value is None
            or update_value is None
            or gate_value is None
            or indices_value is None
            or not _ops.supports_residual_add(base_value, update_value, gate_value, indices_value)
        ):
            return None
        return {
            "base": base,
            "update": update,
            "gate": gate,
            "gate_indices": indices,
            "python_indexing": python_indexing,
        }
    return None


def _rewrite_copy_back(graph: torch.fx.Graph, copy: torch.fx.Node) -> bool:
    """Rewrite the gated sums that build a copied-back result from the input it overwrites."""
    bound = linear_fx.bind_call(copy, _COPY)
    if bound is None or bound["non_blocking"] is not False:
        return False
    hidden, result = bound["self"], bound["src"]
    sums: list[tuple[torch.fx.Node, dict[str, Argument]]] = []
    value = result
    while value is not hidden:
        if not isinstance(value, torch.fx.Node):
            return False
        values = _gated_sum(value)
        if values is None:
            return False
        sums.append((value, values))
        value = values["base"]
    if not sums:
        return False
    added = hidden
    for node, values in reversed(sums):
        with graph.inserting_before(node):
            added = call_functional(graph, RESIDUAL_ADD, {**values, "base": added})
        node.replace_all_uses_with(added)
    _log.debug("%s: %d residual sums add into %s", copy.name, len(sums), hidden)
    return True


def fold_residual_updates(graph: torch.fx.Graph) -> bool:
    """Rewrite eligible copy-back gated residual sums into functional in-place adds."""
    changed = False
    for node in list(graph.nodes):
        if node.target is _COPY:
            changed = _rewrite_copy_back(graph, node) or changed
    if changed:
        # Drop the replaced sums and their gated products.
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__ = ["fold_residual_updates"]
