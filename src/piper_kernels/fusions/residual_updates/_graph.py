"""FX helpers for functional residual adds, shared by the rewrite and the fusions.

The residual-update rewrite emits each residual sum as a functional ``residual_add_``
call. A fusion whose operator has a residual-add counterpart then uses
:func:`compute_inside_adds` to compute its output inside the add that reads it.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable, Mapping

import torch
from torch.fx.node import Argument

from piper_kernels.linear import _compile_fx as linear_fx

from . import _ops  # noqa: F401 - registers residual_add_

_log = logging.getLogger(__name__)
_AUTO_FUNCTIONALIZED = torch.ops.higher_order.auto_functionalized
RESIDUAL_ADD = torch.ops.piper_kernels.residual_add_.default


def call_functional(
    graph: torch.fx.Graph,
    target: torch._ops.OpOverload,
    values: Mapping[str, Argument],
) -> torch.fx.Node:
    """Emit ``target``, which mutates one tensor argument, as a functional call.

    Arguments are taken by schema name; omitted ones use their defaults. The graph
    stays functional, so Inductor's reinplacing pass decides whether the mutated
    operand is updated in place or copied first. The new value carries that operand's
    metadata, so passes that run later can match on it.
    """
    mutated = next(
        argument.name
        for argument in target._schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    )
    call = graph.call_function(_AUTO_FUNCTIONALIZED, (target,), dict(values))
    # The call returns the operator's result (None) followed by the mutated tensor.
    result = graph.call_function(operator.getitem, (call, 1))
    value = linear_fx.node_tensor_metadata(values[mutated])
    if value is not None:
        result.meta["val"] = value.clone()
        call.meta["val"] = (None, result.meta["val"])
    return result


type ComputeInside = Callable[[torch.fx.Graph, torch.fx.Node, dict[str, Argument]], torch.fx.Node]


def compute_inside_adds(
    graph: torch.fx.Graph,
    producer: torch._ops.OpOverload,
    compute: ComputeInside,
) -> int:
    """Compute each ``producer`` output that only a residual add reads inside that add.

    For each such add, ``compute(graph, update, add)`` emits functional calls that add
    the ``producer`` call ``update``'s output into ``base`` themselves, given the add's
    ``base``, ``gate``, ``gate_indices``, and ``python_indexing`` by name, and returns
    the new ``base``; the add and ``update`` are then removed. Returns how many adds it
    replaced, and logs each at debug level.
    """
    replaced = 0
    for call in list(graph.nodes):
        if call.target is not _AUTO_FUNCTIONALIZED or call.args[0] is not RESIDUAL_ADD:
            continue
        add = dict(call.kwargs)
        update = add.pop("update")
        if (
            not isinstance(update, torch.fx.Node)
            or update.target is not producer
            or set(update.users) != {call}
            or any(user.target is not operator.getitem or user.args[1] != 1 for user in call.users)
        ):
            continue
        with graph.inserting_before(call):
            result = compute(graph, update, add)
        for user in list(call.users):
            user.replace_all_uses_with(result)
            graph.erase_node(user)
        graph.erase_node(call)
        graph.erase_node(update)
        _log.debug("%s is computed inside its add into %s", producer, add["base"])
        replaced += 1
    return replaced


__all__ = [
    "RESIDUAL_ADD",
    "ComputeInside",
    "call_functional",
    "compute_inside_adds",
]
