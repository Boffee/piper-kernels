"""Run an attention residual add in two head groups, so only one group's K and V exist.

A fused attention residual add reads K and V for every head, projected before it runs.
When those projections feed nothing else, the attention's fold emits the three calls
once per head group instead: each group projects its own K and V rows, then adds
attention over its heads into ``base`` through its share of the output projection's
input columns. Inductor frees a group's K and V after its add, before the next group
projects, and its peak-memory scheduling keeps that order.

The calls are the attention's own operators on head-sliced weights, so an attention
joins by describing which of its arguments split by head in a :class:`HeadSplit`,
kept beside its operators. Each group quantizes its own output-projection input per
token and ``base`` rounds once per group, so results differ slightly from one group.
Each group's query windows hold twice the rows, so every attention launch keeps its
ungrouped rows times heads.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates._graph import bind_arguments, call_functional

# Two groups halve K and V at no measured speed cost; past that, other buffers dominate.
HEAD_GROUPS = 2


@dataclass(frozen=True)
class HeadSplit:
    """Which arguments of an attention residual add, and of its producers, split by head.

    ``producers`` pairs each K or V producer with the add arguments it supplies, in output
    order; its ``producer_rows`` arguments hold one block of rows per head. In the add,
    ``rows`` hold one block of rows per query head, ``columns`` names the output
    projection's weight, whose input columns follow the query heads, and ``once`` is
    added by the first group only. Each group's columns must be a multiple of the add's
    ``alignment`` argument, such as a ConvRot group size.
    """

    producers: tuple[tuple[torch._ops.OpOverload, tuple[str, ...]], ...]
    producer_rows: tuple[str, ...]
    rows: tuple[str, ...]
    columns: str
    once: tuple[str, ...]
    alignment: str


def _exclusive_producer(
    operands: Mapping[str, Argument],
    names: tuple[str, ...],
    target: torch._ops.OpOverload,
    consumer: torch.fx.Node,
) -> torch.fx.Node | None:
    """Return the ``target`` node whose outputs ``names`` are, if only ``consumer`` reads it."""
    items = [operands.get(name) for name in names]
    nodes = [item for item in items if isinstance(item, torch.fx.Node)]
    if len(nodes) != len(names) or any(
        node.target is not operator.getitem or node.args[1] != index
        for index, node in enumerate(nodes)
    ):
        return None
    producer = nodes[0].args[0]
    if (
        not isinstance(producer, torch.fx.Node)
        or producer.target is not target
        or any(node.args[0] is not producer for node in nodes)
    ):
        return None
    if any(set(item.users) != {consumer} for item in producer.users):
        return None
    return producer


def _metadata(value: Argument) -> object:
    return value.meta.get("val") if isinstance(value, torch.fx.Node) else value


def _rows(value: Argument) -> int | None:
    metadata = _metadata(value)
    return metadata.shape[0] if isinstance(metadata, torch.Tensor) else None


def _call(
    graph: torch.fx.Graph, target: torch._ops.OpOverload, values: Mapping[str, Argument]
) -> torch.fx.Node:
    """Emit ``target`` with its fake result as the node's metadata.

    Arguments go positionally in schema order, as in traced graphs, since Inductor passes
    read some operators' arguments by position.
    """
    arguments = target._schema.arguments
    args = tuple(values[a.name] for a in arguments if not a.kwarg_only and a.name in values)
    kwargs = {a.name: values[a.name] for a in arguments if a.kwarg_only and a.name in values}
    node = graph.call_function(target, args, kwargs)
    node.meta["val"] = target(
        *(_metadata(value) for value in args),
        **{name: _metadata(value) for name, value in kwargs.items()},
    )
    return node


def _group_slice(graph: torch.fx.Graph, value: Argument, group: int, dim: int) -> Argument:
    """Slice one head group's share of ``value`` along ``dim``; None stays None."""
    metadata = _metadata(value)
    if not isinstance(metadata, torch.Tensor):
        return None
    size = metadata.shape[dim] // HEAD_GROUPS
    bounds = {"dim": dim, "start": group * size, "end": (group + 1) * size}
    return _call(graph, torch.ops.aten.slice.Tensor, {"self": value, **bounds})


def add_in_head_groups(
    graph: torch.fx.Graph,
    residual_add: torch._ops.OpOverload,
    update: torch.fx.Node,
    values: dict[str, Argument],
    split: HeadSplit,
) -> torch.fx.Node | None:
    """Emit ``residual_add`` once per head group and return the updated ``base``.

    ``values`` are the add's arguments by name, and ``update`` is the attention output it
    replaces, whose K and V operands must feed nothing else. Emits nothing and returns
    None when K or V has other consumers or the heads don't split evenly into aligned
    groups.
    """
    producers: list[tuple[torch._ops.OpOverload, dict[str, Argument], tuple[str, ...]]] = []
    for target, names in split.producers:
        producer = _exclusive_producer(values, names, target, update)
        if producer is None:
            return None
        producers.append((target, bind_arguments(producer), names))

    head_rows = HEAD_GROUPS * cast(int, values["head_dim"])
    row_counts = [
        _rows(values[split.rows[0]]),
        *(_rows(arguments[split.producer_rows[0]]) for _, arguments, _ in producers),
    ]
    if any(rows is None or rows % head_rows for rows in row_counts):
        return None
    group_columns = cast(int, row_counts[0]) // HEAD_GROUPS
    if group_columns % cast(int, values[split.alignment]):
        return None

    chunk_rows = values.get("query_chunk_rows")
    if chunk_rows is None:
        (chunk_rows,) = (
            argument.default_value
            for argument in residual_add._schema.arguments
            if argument.name == "query_chunk_rows"
        )
    base = cast(torch.fx.Node, values["base"])
    for group in range(HEAD_GROUPS):
        group_values = {
            **values,
            "base": base,
            "query_chunk_rows": cast(int, chunk_rows) * HEAD_GROUPS,
            **{name: _group_slice(graph, values[name], group, 0) for name in split.rows},
            # The output GEMM needs each group's input columns contiguous.
            split.columns: _call(
                graph,
                torch.ops.aten.clone.default,
                {
                    "self": _group_slice(graph, values[split.columns], group, 1),
                    "memory_format": torch.contiguous_format,
                },
            ),
        }
        if group:
            group_values.update(dict.fromkeys(split.once))
        for target, arguments, names in producers:
            sliced = {
                name: _group_slice(graph, arguments[name], group, 0) for name in split.producer_rows
            }
            produced = _call(graph, target, {**arguments, **sliced})
            for index, name in enumerate(names):
                item = graph.call_function(operator.getitem, (produced, index))
                item.meta["val"] = produced.meta["val"][index]
                group_values[name] = item
        base = call_functional(graph, residual_add, group_values)
    return base


__all__ = ["HEAD_GROUPS", "HeadSplit", "add_in_head_groups"]
