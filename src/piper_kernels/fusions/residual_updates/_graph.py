"""FX helpers shared by residual-update rewrites, independent of operator families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument


@dataclass(frozen=True, slots=True)
class ResidualAdd:
    """A ``residual_add_(base, update, gate, gate_indices, python_indexing)`` call."""

    node: torch.fx.Node
    base: torch.fx.Node
    update: torch.fx.Node
    gate: Argument
    gate_indices: Argument
    python_indexing: bool


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


def call_by_name(
    graph: torch.fx.Graph,
    target: torch._ops.OpOverload,
    values: Mapping[str, Argument],
) -> torch.fx.Node:
    """Emit ``target`` with arguments taken by schema name; omitted ones use defaults."""
    arguments: list[Argument] = []
    kwargs: dict[str, Argument] = {}
    deferred_defaults: list[Argument] = []
    for argument in target._schema.arguments:
        if argument.name in values:
            if argument.kwarg_only:
                kwargs[argument.name] = values[argument.name]
            else:
                arguments.extend(deferred_defaults)
                deferred_defaults.clear()
                arguments.append(values[argument.name])
        elif not argument.has_default_value():
            raise ValueError(f"{target} requires {argument.name}")
        elif not argument.kwarg_only:
            deferred_defaults.append(argument.default_value)
    return graph.call_function(target, tuple(arguments), kwargs)


__all__ = ["ResidualAdd", "bind_arguments", "call_by_name"]
