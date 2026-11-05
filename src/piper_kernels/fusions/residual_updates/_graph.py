"""FX helpers shared by residual-update rewrites, independent of operator families."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import cast

import torch
from torch.fx.node import Argument

_AUTO_FUNCTIONALIZED = torch.ops.higher_order.auto_functionalized


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
    updated in place or copied first.
    """
    call = graph.call_function(_AUTO_FUNCTIONALIZED, (target,), dict(values))
    # The call returns the operator's result (None) followed by the new ``base``.
    return graph.call_function(operator.getitem, (call, 1))


__all__ = ["bind_arguments", "call_functional"]
