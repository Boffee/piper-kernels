"""Rewrite copy-back residual blocks to add their updates into their input in place.

A compiled block whose program ends with ``hidden.copy_(result)`` declares that the
caller no longer needs ``hidden``; Inductor then permits writes to it. Two independent
rewrites follow, joined only by the residual:

1. FFN residual adds. After FFN fusion, a block contains::

       residual = hidden + gate[indices] * update        # feeds the FFN input
       ffn_gated_updates_(..., base=hidden, reusable_update=update, update_gate=gate, ...)
       hidden.copy_(update)

   which becomes::

       residual_add_(hidden, update, gate, indices)      # hidden += gate[indices] * update
       ffn_residual_add_(..., base=hidden, ...)          # hidden += ffn_gate[indices] * ffn
                                                         # (the FFN input reads ``hidden``)

   Any producer of ``update`` qualifies; the block's new output buffer is gone.

2. Attention residual adds. A registered attention output consumed only by a residual
   add folds into its family's in-place operator, removing the full-size output.

Each FFN family registers its residual-add operator and each attention family its fold.
Unmatched graphs are unchanged; debug logging reports what was rewritten.
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
from ._graph import ResidualAdd, bind_arguments, call_by_name

type AttentionFold = Callable[[torch.fx.Graph, ResidualAdd], bool]

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
# Each attention output operator and the fold into its in-place residual add.
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
    if base.op != "placeholder":
        return False
    copies = [user for user in update.users if user.target is _COPY]
    if len(copies) != 1 or copies[0].args != (base, update):
        return False
    (copy,) = copies
    # Another reader would expect ``update`` to hold the block result, which the
    # rewrite writes into ``base`` instead.
    readers = set(update.users) - {ffn, copy}
    if len(readers) != 1:
        _log.debug("%s: update %s has other consumers", ffn.name, update.name)
        return False
    (product,) = readers
    (residual,) = product.users
    # Every other reader of the original input must precede the in-place add.
    order = {node: index for index, node in enumerate(graph.nodes)}
    if any(
        user not in (residual, ffn, copy) and order[user] > order[residual] for user in base.users
    ):
        _log.debug("%s: %s is read after the residual update", ffn.name, base.name)
        return False

    with graph.inserting_before(residual):
        graph.call_function(
            _RESIDUAL_ADD,
            (
                base,
                update,
                values["update_gate"],
                values["gate_indices"],
                values["python_indexing"],
            ),
        )
    residual.replace_all_uses_with(base)
    with graph.inserting_before(ffn):
        call_by_name(
            graph,
            residual_add,
            {name: value for name, value in values.items() if name not in _FIRST_UPDATE_ARGUMENTS},
        )
    copy.replace_all_uses_with(base)
    graph.erase_node(copy)
    graph.erase_node(ffn)
    _log.debug("%s: FFN updates %s in place", ffn.name, base.name)
    return True


def _fold_attention(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    values = bind_arguments(node)
    # Step 1 emitted this add with node operands and a constant index mode.
    add = ResidualAdd(
        node,
        cast(torch.fx.Node, values["base"]),
        cast(torch.fx.Node, values["update"]),
        values["gate"],
        values["gate_indices"],
        cast(bool, values["python_indexing"]),
    )
    fold = _ATTENTION_FOLDS.get(add.update.target)
    if fold is None or not fold(graph, add):
        _log.debug("%s: %s stays materialized before its add", node.name, add.update.target)
        return
    graph.erase_node(node)
    _log.debug("%s: %s adds into %s in place", node.name, add.update.target, add.base.name)


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
            if node.target is _RESIDUAL_ADD:
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
