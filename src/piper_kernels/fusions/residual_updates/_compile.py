"""Rewrite copy-back residual blocks so their updates can add into their input in place.

A compiled block whose program ends with ``hidden.copy_(result)`` declares that the
caller no longer needs ``hidden``. The rewrite expresses the block's residual updates
as operators that add into a ``base``, emitted as functional calls so the graph stays
functional. Inductor's reinplacing pass, which runs after every custom pass and tracks
every view of ``hidden``, then updates ``hidden`` in place when nothing reads its
original value afterward and the copy back permits it; otherwise it copies ``hidden``
first.

After FFN fusion, a block contains::

    residual = hidden + gate[indices] * update        # feeds the FFN input
    ffn_gated_updates_(..., base=hidden, reusable_update=update, update_gate=gate, ...)
    hidden.copy_(update)

which becomes::

    added = residual_add(hidden, update, gate, indices)  # hidden + gate[indices] * update
    result = ffn_residual_add(..., base=added, ...)      # added + ffn_gate[indices] * ffn
    hidden.copy_(result)                                 # (the FFN input reads ``added``)

Any producer of ``update`` qualifies; once both adds are in place, the block's new
output buffer is gone. The rewrite leaves ``update`` with no other consumer, so when it
is a registered attention output, the add calls that output's residual-add counterpart
instead, which computes the output inside the add, and the full-size attention output
is gone too.

Each FFN two-update operator and each attention output operator registers its
residual-add counterpart. Unmatched graphs are unchanged; debug logging reports what
was rewritten, and Inductor's reinplacing log reports any input it had to copy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass, get_hash_for_files
from torch.fx.node import Argument

from piper_kernels.fusions.convrot_int8_piper import output as dense_attention
from piper_kernels.fusions.convrot_int8_sparse_piper import output as sparse_attention
from piper_kernels.fusions.convrot_int8_swiglu_ffn import triton as convrot_int8_swiglu_ffn
from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _graph, _ops, triton
from ._graph import bind_arguments, call_functional

_log = logging.getLogger(__name__)
_COMPILE_PASS_VERSION = "residual-updates-compile-v3"

_kernels = torch.ops.piper_kernels
_RESIDUAL_ADD = _kernels.residual_add_.default
_COPY = torch.ops.aten.copy_.default

# Each FFN two-update operator and its in-place residual-add counterpart. A
# counterpart takes the two-update operator's arguments, by name, without
# ``reusable_update`` and its ``update_gate``. The FFN fusion replaces
# ``base + update_gate[i] * reusable_update + ffn_gate[i] * ffn(...)`` with the
# two-update operator and keeps ``base + update_gate[i] * reusable_update`` as the
# FFN input's source, so that gated product is ``reusable_update``'s other reader.
_FFN_RESIDUAL_ADDS: dict[object, torch._ops.OpOverload] = {
    _kernels.convrot_int8_swiglu_ffn_gated_updates_.default: (
        _kernels.convrot_int8_swiglu_ffn_residual_add_.default
    ),
}
# Each attention output operator and its residual-add counterpart, which computes the
# output inside ``base + gate[i] * output``. A counterpart takes the add's ``base``,
# ``gate``, ``gate_indices``, and ``python_indexing``, then the output operator's
# arguments, by name, without ``output_dtype``: the FFN fusion matched the output and
# ``base`` with one dtype, so ``base`` supplies it.
_ATTENTION_RESIDUAL_ADDS: dict[object, torch._ops.OpOverload] = {
    _kernels.convrot_int8_piper_projected_query_attention_output.default: (
        _kernels.convrot_int8_piper_projected_query_attention_residual_add_.default
    ),
    _kernels.convrot_int8_sparse_piper_projected_query_attention_output.default: (
        _kernels.convrot_int8_sparse_piper_projected_query_attention_residual_add_.default
    ),
}
_REUSABLE_UPDATE_ARGUMENTS = ("reusable_update", "update_gate")


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
            dense_attention.__file__,
            sparse_attention.__file__,
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

    update_add: dict[str, Argument] = {
        "base": base,
        "gate": values["update_gate"],
        "gate_indices": values["gate_indices"],
        "python_indexing": values["python_indexing"],
    }
    attention_add = _ATTENTION_RESIDUAL_ADDS.get(update.target)
    with graph.inserting_before(residual):
        if attention_add is None:
            added = call_functional(graph, _RESIDUAL_ADD, {**update_add, "update": update})
        else:
            operands = {
                name: value
                for name, value in bind_arguments(update).items()
                if name != "output_dtype"
            }
            added = call_functional(graph, attention_add, {**update_add, **operands})
    residual.replace_all_uses_with(added)
    ffn_add = {
        name: value for name, value in values.items() if name not in _REUSABLE_UPDATE_ARGUMENTS
    }
    with graph.inserting_before(ffn):
        result = call_functional(graph, residual_add, {**ffn_add, "base": added})
    copy.args = (base, result)
    graph.erase_node(ffn)
    _log.debug(
        "%s: %s %s its add, and the FFN adds into %s",
        ffn.name,
        update.target,
        "is computed inside" if attention_add is not None else "stays materialized before",
        base.name,
    )
    return True


class _ResidualUpdatePass(CustomInferenceAwareGraphPass):
    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        changed = False
        for node in list(graph.nodes):
            residual_add = _FFN_RESIDUAL_ADDS.get(node.target)
            if residual_add is not None:
                changed = _rewrite_ffn_updates(graph, node, residual_add) or changed
        if changed:
            # Drop the replaced residual sums and any attention output computed in its add.
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
