"""Rewrite copy-back residual blocks so their updates can add into their input in place.

A compiled block whose program ends with ``hidden.copy_(result)`` declares that the
caller no longer needs ``hidden``. The rewrite expresses the block's residual updates
as operators that add into a ``base``, emitted as functional calls so the graph stays
functional. Inductor's reinplacing pass, which runs after every custom pass and tracks
every view of ``hidden``, then updates ``hidden`` in place when nothing reads its
original value afterward and the copy back permits it; otherwise it copies ``hidden``
first. Two independent rewrites follow, joined only by the residual:

1. FFN residual adds. After FFN fusion, a block contains::

       residual = hidden + gate[indices] * update        # feeds the FFN input
       ffn_gated_updates_(..., base=hidden, reusable_update=update, update_gate=gate, ...)
       hidden.copy_(update)

   which becomes::

       added = residual_add(hidden, update, gate, indices)  # hidden + gate[indices] * update
       result = ffn_residual_add(..., base=added, ...)      # added + ffn_gate[indices] * ffn
       hidden.copy_(result)                                 # (the FFN input reads ``added``)

   Any producer of ``update`` qualifies. Once both adds are in place, the block's new
   output buffer is gone.

2. Attention residual adds. A registered attention output consumed only by a residual
   add folds into its family's residual-add operator, removing the full-size output.

Each FFN family registers its residual-add operator and each attention family its fold.
Unmatched graphs are unchanged; debug logging reports what was rewritten, and
Inductor's reinplacing log reports any input it had to copy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import cast

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass, get_hash_for_files

from piper_kernels.fusions.convrot_int8_piper import _residual_compile as dense_attention
from piper_kernels.fusions.convrot_int8_swiglu_ffn import triton as convrot_int8_swiglu_ffn
from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _graph, _ops, triton
from ._graph import ResidualAdd, bind_arguments, call_functional, functional_target

type AttentionFold = Callable[[torch.fx.Graph, ResidualAdd], torch.fx.Node | None]

_log = logging.getLogger(__name__)
_COMPILE_PASS_VERSION = "residual-updates-compile-v3"

_kernels = torch.ops.piper_kernels
_RESIDUAL_ADD = _kernels.residual_add_.default
_COPY = torch.ops.aten.copy_.default

# Each FFN two-update operator and its in-place residual-add counterpart. A
# counterpart takes the two-update operator's arguments, by name, without the
# first update's ``reusable_update`` and ``update_gate``. The FFN fusion replaces
# ``base + update_gate[i] * reusable_update + ffn_gate[i] * ffn(...)`` with the
# two-update operator and keeps ``base + update_gate[i] * reusable_update`` as the
# FFN input's source, so that gated product is ``reusable_update``'s other reader.
_FFN_RESIDUAL_ADDS: dict[object, torch._ops.OpOverload] = {
    _kernels.convrot_int8_swiglu_ffn_gated_updates_.default: (
        _kernels.convrot_int8_swiglu_ffn_residual_add_.default
    ),
}
# Each attention output operator and the fold into its residual-add operator.
_ATTENTION_FOLDS: dict[object, AttentionFold] = {
    _kernels.convrot_int8_piper_projected_query_attention_output.default: (
        dense_attention.fold_residual_add
    ),
}
_FIRST_UPDATE_ARGUMENTS = ("reusable_update", "update_gate")


def _source_files() -> tuple[str, ...]:
    """Invalidate cached rewrites when matched or emitted operator contracts change."""
    return tuple(
        path
        for path in (
            __file__,
            triton.__file__,
            _graph.__file__,
            _ops.__file__,
            convrot_int8_swiglu_ffn.__file__,
            *dense_attention.source_files(),
            preparation_sharing.__file__,
        )
        if path is not None
    )


def _rewrite_ffn_updates(
    graph: torch.fx.Graph,
    ffn: torch.fx.Node,
    residual_add: torch._ops.OpOverload,
) -> bool:
    values = bind_arguments(ffn)
    # Tensor operands of the two-update operator are graph nodes.
    base = cast(torch.fx.Node, values["base"])
    update = cast(torch.fx.Node, values["reusable_update"])
    # Without a copy back into ``base``, Inductor would copy it rather than update it in
    # place, so the rewrite would only add work.
    copies = [user for user in update.users if user.target is _COPY]
    if len(copies) != 1 or copies[0].args != (base, update):
        return False
    (copy,) = copies
    # Another reader would expect ``update`` to hold the block result, which the
    # rewrite no longer stores there.
    readers = set(update.users) - {ffn, copy}
    if len(readers) != 1:
        _log.debug("%s: update %s has other consumers", ffn.name, update.name)
        return False
    (product,) = readers
    (residual,) = product.users

    with graph.inserting_before(residual):
        updated = call_functional(
            graph,
            _RESIDUAL_ADD,
            {
                "base": base,
                "update": update,
                "gate": values["update_gate"],
                "gate_indices": values["gate_indices"],
                "python_indexing": values["python_indexing"],
            },
        )
    residual.replace_all_uses_with(updated)
    with graph.inserting_before(ffn):
        result = call_functional(
            graph,
            residual_add,
            {
                **{
                    name: value
                    for name, value in values.items()
                    if name not in _FIRST_UPDATE_ARGUMENTS
                },
                "base": updated,
            },
        )
    copy.args = (base, result)
    graph.erase_node(ffn)
    _log.debug("%s: FFN adds into %s", ffn.name, base.name)
    return True


def _fold_attention(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    # Step 1 emitted this call by name, with node operands and a constant index mode.
    values = node.kwargs
    add = ResidualAdd(
        node,
        cast(torch.fx.Node, values["base"]),
        cast(torch.fx.Node, values["update"]),
        values["gate"],
        values["gate_indices"],
        cast(bool, values["python_indexing"]),
    )
    fold = _ATTENTION_FOLDS.get(add.update.target)
    folded = None if fold is None else fold(graph, add)
    if folded is None:
        _log.debug("%s: %s stays materialized before its add", node.name, add.update.target)
        return
    (added,) = node.users
    added.replace_all_uses_with(folded)
    graph.erase_node(added)
    graph.erase_node(node)
    _log.debug("%s: %s adds into %s", node.name, add.update.target, add.base.name)


class _ResidualUpdatePass(CustomInferenceAwareGraphPass):
    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        changed = False
        for node in list(graph.nodes):
            residual_add = _FFN_RESIDUAL_ADDS.get(node.target)
            if residual_add is not None:
                changed = _rewrite_ffn_updates(graph, node, residual_add) or changed
        if not changed:
            return
        # Drop the replaced residual sums so each update's only consumer is its add.
        graph.eliminate_dead_code()
        for node in list(graph.nodes):
            if functional_target(node) is _RESIDUAL_ADD:
                _fold_attention(graph, node)
        graph.eliminate_dead_code()
        graph.lint()

    def uuid(self) -> bytes:
        return get_hash_for_files(_source_files(), extra=_COMPILE_PASS_VERSION)


_compile_pass = _ResidualUpdatePass()


def residual_updates_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install the in-place residual rewrite after existing attention and FFN passes.

    See the package helper for the caller contract. Buffer reuse is disabled because
    Inductor's planner would otherwise keep a dead full-size buffer allocated across
    the in-place attention to reuse it afterward.
    """
    combined = preparation_sharing.add_ordered_post_grad_passes(options, (_compile_pass,))
    combined["allow_buffer_reuse"] = False
    return combined


__all__ = ["residual_updates_compile_options"]
