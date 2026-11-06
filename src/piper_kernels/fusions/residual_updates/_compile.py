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

The rewrite runs before every other custom pass and names no fusion. A fusion whose
operator has a residual-add counterpart finds the adds that alone read its output with
``_graph.compute_inside_adds`` and computes that output inside the add, which also drops
the full-size update. Unmatched graphs, including those on devices Triton doesn't
support, are unchanged. Debug logging here and in ``_graph`` reports what was
rewritten, and Inductor's reinplacing log reports any input it had to copy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass, get_hash_for_files
from torch.fx.node import Argument

from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _graph, _ops, triton
from ._graph import RESIDUAL_ADD, call_functional

_log = logging.getLogger(__name__)
_COMPILE_PASS_VERSION = "residual-updates-compile-v4"

_aten = torch.ops.aten
_COPY = _aten.copy_.default
# Each gate-row selection and whether it wraps negative indices, as Python indexing does.
_GATE_SELECTIONS: dict[object, bool] = {
    _aten.index.Tensor: True,
    _aten.index_select.default: False,
}


def _source_files() -> tuple[str, ...]:
    """Invalidate cached rewrites when matched or emitted operator contracts change."""
    return tuple(
        path
        for path in (
            __file__,
            triton.__file__,
            _graph.__file__,
            _ops.__file__,
            preparation_sharing.__file__,
        )
        if path is not None
    )


def _metadata(value: Argument) -> torch.Tensor | None:
    return preparation_sharing.tensor_metadata(value) if isinstance(value, torch.fx.Node) else None


def _selected_gate(node: Argument) -> tuple[torch.fx.Node, torch.fx.Node, bool] | None:
    """Return ``(gate, indices, python_indexing)`` if ``node`` selects one gate row per index."""
    if not isinstance(node, torch.fx.Node):
        return None
    python_indexing = _GATE_SELECTIONS.get(node.target)
    if python_indexing is None:
        return None
    if python_indexing:
        gate, indices = node.args
        if not isinstance(indices, (list, tuple)) or len(indices) != 1:
            return None
        (indices,) = indices
    else:
        gate, dimension, indices = node.args
        if dimension != 0:
            return None
    if not isinstance(gate, torch.fx.Node) or not isinstance(indices, torch.fx.Node):
        return None
    return gate, indices, python_indexing


def _gated_sum(node: torch.fx.Node) -> dict[str, Argument] | None:
    """Return ``residual_add_``'s arguments if ``node`` is ``base + gate[indices] * update``."""
    if node.target is not _aten.add.Tensor or len(node.args) != 2 or node.kwargs:
        return None
    base, product = node.args
    if not isinstance(product, torch.fx.Node) or product.target is not _aten.mul.Tensor:
        return None
    if len(product.args) != 2 or product.kwargs:
        return None
    for selection, update in (product.args, product.args[::-1]):
        selected = _selected_gate(selection)
        if selected is None:
            continue
        gate, indices, python_indexing = selected
        operands = (_metadata(base), _metadata(update), _metadata(gate), _metadata(indices))
        base_value, update_value, gate_value, indices_value = operands
        if (
            base_value is None
            or update_value is None
            or gate_value is None
            or indices_value is None
            or not _graph.supports_residual_add(base_value, update_value, gate_value, indices_value)
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
    hidden, result = copy.args
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


class _ResidualUpdatePass(CustomInferenceAwareGraphPass):
    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        changed = False
        for node in list(graph.nodes):
            if node.target is _COPY:
                changed = _rewrite_copy_back(graph, node) or changed
        if changed:
            # Drop the replaced sums and their gated products.
            graph.eliminate_dead_code()
            graph.lint()

    def uuid(self) -> bytes:
        return get_hash_for_files(_source_files(), extra=_COMPILE_PASS_VERSION)


_compile_pass = _ResidualUpdatePass()


def residual_updates_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install the in-place residual rewrite ahead of every other post-grad pass.

    See the package helper for the caller contract. Buffer reuse is disabled because
    Inductor's planner would otherwise keep a dead full-size buffer allocated across
    the in-place attention to reuse it afterward.
    """
    combined = preparation_sharing.add_first_post_grad_pass(options, _compile_pass)
    combined["allow_buffer_reuse"] = False
    return combined


__all__ = ["residual_updates_compile_options"]
