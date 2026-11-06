"""FX helpers for functional residual adds, shared by the rewrite and the fusions.

The residual-update rewrite emits each residual sum as a functional ``residual_add_``
call. A fusion whose operator has a residual-add counterpart then uses
:func:`compute_inside_adds` to compute its output inside the add that reads it.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import cast

import torch
from torch.fx.node import Argument

from . import _ops  # noqa: F401 - registers residual_add_

_AUTO_FUNCTIONALIZED = torch.ops.higher_order.auto_functionalized
RESIDUAL_ADD = torch.ops.piper_kernels.residual_add_.default


def bind_arguments(node: torch.fx.Node) -> dict[str, Argument]:
    """Name an operator call's arguments by its schema, independent of position.

    ``node`` must call an ``OpOverload``.
    """
    schema = cast(torch._ops.OpOverload, node.target)._schema
    values: dict[str, Argument] = {
        argument.name: value for argument, value in zip(schema.arguments, node.args, strict=False)
    }
    values.update(node.kwargs)
    return values


def call_functional(
    graph: torch.fx.Graph,
    target: torch._ops.OpOverload,
    values: Mapping[str, Argument],
) -> torch.fx.Node:
    """Emit ``target``, which mutates only ``base``, as a functional call; return the new value.

    Arguments are taken by schema name; omitted ones use their defaults. The graph
    stays functional, so Inductor's reinplacing pass decides whether ``base`` is
    updated in place or copied first. The new value carries ``base``'s metadata, so
    passes that run later can match on it.
    """
    call = graph.call_function(_AUTO_FUNCTIONALIZED, (target,), dict(values))
    # The call returns the operator's result (None) followed by the new ``base``.
    result = graph.call_function(operator.getitem, (call, 1))
    base = values["base"]
    value = base.meta.get("val") if isinstance(base, torch.fx.Node) else None
    if isinstance(value, torch.Tensor):
        result.meta["val"] = value.clone()
        call.meta["val"] = (None, result.meta["val"])
    return result


def compute_inside_adds(
    graph: torch.fx.Graph,
    producer: torch._ops.OpOverload,
    residual_add: torch._ops.OpOverload,
    arguments: Callable[[dict[str, Argument], dict[str, Argument]], dict[str, Argument]],
) -> int:
    """Compute each ``producer`` output that only a residual add reads inside that add.

    ``residual_add`` replaces the add; ``arguments(operands, add)`` maps the producer's
    arguments and the add's ``base``, ``gate``, ``gate_indices``, and
    ``python_indexing``, all by name, to its arguments. Returns how many adds it
    replaced.
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
            result = call_functional(graph, residual_add, arguments(bind_arguments(update), add))
        for user in list(call.users):
            user.replace_all_uses_with(result)
            graph.erase_node(user)
        graph.erase_node(call)
        graph.erase_node(update)
        replaced += 1
    return replaced


__all__ = ["RESIDUAL_ADD", "bind_arguments", "call_functional", "compute_inside_adds"]
