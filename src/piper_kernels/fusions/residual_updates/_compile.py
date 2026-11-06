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

With attention head groups, an attention residual add that registers a head-group fold,
and whose K and V producers feed only it, becomes the fold's grouped add, which takes
the producers' inputs and projects each group's K and V itself. One operator keeps the
groups in order: separate per-group nodes would let Inductor schedule every group's K
and V before the first attention. Dense Piper registers a fold; sparse doesn't yet.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping
from typing import cast

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass, get_hash_for_files
from torch.fx.node import Argument

from piper_kernels.fusions.convrot_int8_piper import _head_groups as dense_head_groups
from piper_kernels.fusions.convrot_int8_piper import output as dense_attention
from piper_kernels.fusions.convrot_int8_sparse_piper import output as sparse_attention
from piper_kernels.fusions.convrot_int8_swiglu_ffn import triton as convrot_int8_swiglu_ffn
from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _graph, _ops, triton
from ._graph import bind_arguments, call_functional

_log = logging.getLogger(__name__)
_COMPILE_PASS_VERSION = "residual-updates-compile-v4"

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

# Attention residual adds that can run in head groups. Each fold names the K and V
# producers it absorbs and maps their arguments onto its grouped add, so only one group's
# K and V are ever allocated.
_ATTENTION_HEAD_GROUP_FOLDS: dict[object, dense_head_groups.HeadGroupFold] = {
    _kernels.convrot_int8_piper_projected_query_attention_residual_add_.default: (
        dense_head_groups.HEAD_GROUP_FOLD
    ),
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
            convrot_int8_swiglu_ffn.__file__,
            dense_attention.__file__,
            dense_head_groups.__file__,
            sparse_attention.__file__,
            preparation_sharing.__file__,
        )
        if path is not None
    )


def _exclusive_producer(
    operands: Mapping[str, Argument],
    names: tuple[str, ...],
    target: torch._ops.OpOverload,
    consumer: torch.fx.Node,
) -> torch.fx.Node | None:
    """Return the ``target`` node whose outputs ``names`` are, if only ``consumer`` reads it."""
    producers = set()
    for index, name in enumerate(names):
        item = operands[name]
        if (
            not isinstance(item, torch.fx.Node)
            or item.target is not operator.getitem
            or item.args[1] != index
        ):
            return None
        producers.add(item.args[0])
    if len(producers) != 1:
        return None
    (producer,) = producers
    if not isinstance(producer, torch.fx.Node) or producer.target is not target:
        return None
    if any(set(item.users) != {consumer} for item in producer.users):
        return None
    return producer


def _fold_head_groups(
    update: torch.fx.Node,
    attention_add: torch._ops.OpOverload,
    operands: dict[str, Argument],
    head_groups: int,
) -> tuple[torch._ops.OpOverload, dict[str, Argument]] | None:
    """Return the grouped add and its arguments for ``attention_add``, if it can group."""
    fold = _ATTENTION_HEAD_GROUP_FOLDS.get(attention_add)
    if fold is None:
        return None
    key = _exclusive_producer(operands, fold.key_outputs, fold.key_producer, update)
    value = _exclusive_producer(operands, fold.value_outputs, fold.value_producer, update)
    if key is None or value is None:
        _log.debug("%s: K or V has other consumers; attention keeps one group", update.name)
        return None
    try:
        arguments = fold.build_arguments(
            operands, bind_arguments(key), bind_arguments(value), head_groups
        )
    except ValueError as error:
        # The requested split doesn't fit this attention; it still compiles ungrouped.
        _log.warning("%s: %s; attention keeps one group", update.name, error)
        return None
    return fold.grouped_add, arguments


def _rewrite_ffn_updates(
    graph: torch.fx.Graph,
    ffn: torch.fx.Node,
    residual_add: torch._ops.OpOverload,
    attention_head_groups: int,
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
    grouped = None
    with graph.inserting_before(residual):
        if attention_add is None:
            added = call_functional(graph, _RESIDUAL_ADD, {**update_add, "update": update})
        else:
            operands = {
                name: value
                for name, value in bind_arguments(update).items()
                if name != "output_dtype"
            }
            if attention_head_groups > 1:
                grouped = _fold_head_groups(update, attention_add, operands, attention_head_groups)
            if grouped is None:
                added = call_functional(graph, attention_add, {**update_add, **operands})
            else:
                grouped_add, arguments = grouped
                added = call_functional(graph, grouped_add, {**update_add, **arguments})
    residual.replace_all_uses_with(added)
    ffn_add = {
        name: value for name, value in values.items() if name not in _REUSABLE_UPDATE_ARGUMENTS
    }
    with graph.inserting_before(ffn):
        result = call_functional(graph, residual_add, {**ffn_add, "base": added})
    copy.args = (base, result)
    graph.erase_node(ffn)
    if attention_add is None:
        placement = "stays materialized before its add"
    elif grouped is not None:
        placement = f"is computed inside its add in {attention_head_groups} head groups"
    else:
        placement = "is computed inside its add"
    _log.debug("%s: %s %s, and the FFN adds into %s", ffn.name, update.target, placement, base.name)
    return True


class _ResidualUpdatePass(CustomInferenceAwareGraphPass):
    def __init__(self, attention_head_groups: int) -> None:
        super().__init__()
        self._attention_head_groups = attention_head_groups

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        changed = False
        for node in list(graph.nodes):
            residual_add = _FFN_RESIDUAL_ADDS.get(node.target)
            if residual_add is not None:
                changed = (
                    _rewrite_ffn_updates(graph, node, residual_add, self._attention_head_groups)
                    or changed
                )
        if changed:
            # Drop the replaced residual sums and any attention output computed in its add.
            graph.eliminate_dead_code()
            graph.lint()

    def uuid(self) -> bytes:
        return get_hash_for_files(
            _source_files(),
            extra=f"{_COMPILE_PASS_VERSION}:head-groups={self._attention_head_groups}",
        )


def residual_updates_compile_options(
    options: Mapping[str, object] | None = None,
    *,
    attention_head_groups: int = 1,
) -> dict[str, object]:
    """Install the in-place residual rewrite after existing attention and FFN passes.

    See the package helper for the caller contract. Buffer reuse is disabled because
    Inductor's planner would otherwise keep a dead full-size buffer allocated across
    the in-place attention to reuse it afterward.
    """
    if not isinstance(attention_head_groups, int) or isinstance(attention_head_groups, bool):
        raise TypeError("attention_head_groups must be an integer")
    if attention_head_groups < 1:
        raise ValueError("attention_head_groups must be positive")
    # Passes combine by identity; the earlier rewrite would otherwise run first and leave
    # nothing for this one to group.
    combined = preparation_sharing.add_ordered_post_grad_passes(
        options,
        (_ResidualUpdatePass(attention_head_groups),),
        replaces=lambda compiler_pass: isinstance(compiler_pass, _ResidualUpdatePass),
    )
    combined["allow_buffer_reuse"] = False
    return combined


__all__ = ["residual_updates_compile_options"]
