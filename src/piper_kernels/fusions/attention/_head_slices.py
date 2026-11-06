"""Run an attention residual add in two head slices, so only one slice's K and V exist.

A fused attention residual add reads K and V for every head, projected before it runs.
When those projections feed nothing else, the attention's fold emits the three calls
once per head slice instead: each slice projects its own K and V rows, then adds
attention over its heads into ``base`` through its share of the output projection's
input columns. Inductor frees a slice's K and V after its add, before the next slice
projects, and its peak-memory scheduling keeps that order.

The calls are the attention's own operators on head-sliced weights, so an attention
joins by describing its arguments in a :class:`HeadSlicing`, kept beside its operators.
Each slice quantizes its own output-projection input per token and ``base`` rounds once
per slice, so results differ slightly from one slice.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates._graph import bind_arguments, call_functional

# Two slices halve K and V for at most 1.4% of attention time in measurements; past
# that, other buffers dominate.
NUM_HEAD_SLICES = 2

type Producer = Mapping[torch._ops.OpOverload, tuple[str, ...]]
"""Operators that can supply some of an add's K or V arguments, with the add arguments
each supplies, in output order."""


@dataclass(frozen=True)
class HeadSlicing:
    """Which arguments of an attention residual add, and of its producers, split by head.

    Each of ``producers`` supplies some of the add's K and V arguments; its operators'
    ``producer_rows`` arguments hold one block of rows per head. In the add, ``rows``
    hold one block of rows per query head, ``columns`` names the output projection's
    weight, whose input columns follow the query heads, and ``once`` is added by the
    first slice only. ``rows`` and ``producer_rows`` start with the query and K or V
    weights, which set the head counts. Each slice's columns must be a multiple of the add's
    ``alignment`` argument, such as a ConvRot group size. An add given any of
    ``unsliced``, which hold heads in some other layout, runs as one slice.

    ``window_rows`` names the add's query-window row cap, which each slice multiplies
    by ``NUM_HEAD_SLICES`` so its attention launches keep their unsliced rows times
    heads. With None, each slice keeps the cap, so its launches and their window
    buffers cover half as many heads. ``position`` names the add's arguments for a
    slice's index and the number of slices, for an add that resolves some state
    across every head, such as sparse Piper's keep budgets.

    Each slice adds into the previous slice's ``base``. The add or one of its producers
    must take ``head_dim``.
    """

    producers: tuple[Producer, ...]
    producer_rows: tuple[str, ...]
    rows: tuple[str, ...]
    columns: str
    once: tuple[str, ...]
    alignment: str
    unsliced: tuple[str, ...]
    window_rows: str | None
    position: tuple[str, str] | None


def _find_exclusive_producer(
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


def _find_producer(
    operands: Mapping[str, Argument], producer: Producer, consumer: torch.fx.Node
) -> tuple[torch._ops.OpOverload, dict[str, Argument], tuple[str, ...]] | None:
    """Return the operator, arguments, and outputs of the call that supplies ``producer``."""
    for target, names in producer.items():
        node = _find_exclusive_producer(operands, names, target, consumer)
        if node is not None:
            return target, bind_arguments(node), names
    return None


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
    read some operators' arguments by position. Those after an omitted argument go by
    name, so they keep their meaning.
    """
    args: list[Argument] = []
    kwargs: dict[str, Argument] = {}
    for index, argument in enumerate(target._schema.arguments):
        if argument.name not in values:
            continue
        if argument.kwarg_only or len(args) < index:
            kwargs[argument.name] = values[argument.name]
        else:
            args.append(values[argument.name])
    node = graph.call_function(target, tuple(args), kwargs)
    node.meta["val"] = target(
        *(_metadata(value) for value in args),
        **{name: _metadata(value) for name, value in kwargs.items()},
    )
    return node


def _slice_share(graph: torch.fx.Graph, value: Argument, slice_index: int, dim: int) -> Argument:
    """Slice one head slice's share of ``value`` along ``dim``; None stays None."""
    metadata = _metadata(value)
    if not isinstance(metadata, torch.Tensor):
        return None
    size = metadata.shape[dim] // NUM_HEAD_SLICES
    bounds = {"dim": dim, "start": slice_index * size, "end": (slice_index + 1) * size}
    return _call(graph, torch.ops.aten.slice.Tensor, {"self": value, **bounds})


def _head_dim(operands: Iterable[Mapping[str, Argument]]) -> int | None:
    """Return the first ``head_dim`` among the add's and its producers' arguments."""
    for values in operands:
        head_dim = values.get("head_dim")
        if head_dim is not None:
            return head_dim if isinstance(head_dim, int) else None
    return None


def _scaled_window_rows(
    residual_add: torch._ops.OpOverload, values: Mapping[str, Argument], name: str | None
) -> dict[str, Argument]:
    """Return each slice's query-window row cap, when the slicing scales argument ``name``."""
    if name is None:
        return {}
    rows = values.get(name)
    if rows is None:
        (rows,) = (
            argument.default_value
            for argument in residual_add._schema.arguments
            if argument.name == name
        )
    return {name: cast(int, rows) * NUM_HEAD_SLICES}


def add_in_head_slices(
    graph: torch.fx.Graph,
    residual_add: torch._ops.OpOverload,
    update: torch.fx.Node,
    values: dict[str, Argument],
    slicing: HeadSlicing,
) -> torch.fx.Node | None:
    """Emit ``residual_add`` once per head slice and return the updated ``base``.

    ``values`` are the add's arguments by name, and ``update`` is the attention output it
    replaces, whose K and V operands must feed nothing else. Emits nothing and returns
    None when K or V has other consumers, the add takes an ``unsliced`` argument, or the
    heads don't split evenly into aligned slices.
    """
    if any(values.get(name) is not None for name in slicing.unsliced):
        return None
    producers: list[tuple[torch._ops.OpOverload, dict[str, Argument], tuple[str, ...]]] = []
    for producer in slicing.producers:
        found = _find_producer(values, producer, update)
        if found is None:
            return None
        producers.append(found)

    head_dim = _head_dim((values, *(arguments for _, arguments, _ in producers)))
    if head_dim is None:
        return None
    row_counts = [
        _rows(values[slicing.rows[0]]),
        *(_rows(arguments[slicing.producer_rows[0]]) for _, arguments, _ in producers),
    ]
    # Each slice takes whole query, K, and V heads.
    if any(rows is None or rows % (NUM_HEAD_SLICES * head_dim) for rows in row_counts):
        return None
    slice_columns = cast(int, row_counts[0]) // NUM_HEAD_SLICES
    if slice_columns % cast(int, values[slicing.alignment]):
        return None

    window_rows = _scaled_window_rows(residual_add, values, slicing.window_rows)
    base = cast(torch.fx.Node, values["base"])
    for slice_index in range(NUM_HEAD_SLICES):
        slice_values = {
            **values,
            "base": base,
            **window_rows,
            **{
                name: _slice_share(graph, values[name], slice_index, 0)
                for name in slicing.rows
                if name in values
            },
            # The output GEMM needs each slice's input columns contiguous.
            slicing.columns: _call(
                graph,
                torch.ops.aten.clone.default,
                {
                    "self": _slice_share(graph, values[slicing.columns], slice_index, 1),
                    "memory_format": torch.contiguous_format,
                },
            ),
        }
        if slice_index:
            slice_values.update(dict.fromkeys(slicing.once))
        if slicing.position is not None:
            index_name, count_name = slicing.position
            slice_values.update({index_name: slice_index, count_name: NUM_HEAD_SLICES})
        for target, arguments, names in producers:
            sliced = {
                name: _slice_share(graph, arguments[name], slice_index, 0)
                for name in slicing.producer_rows
                if name in arguments
            }
            produced = _call(graph, target, {**arguments, **sliced})
            for index, name in enumerate(names):
                item = graph.call_function(operator.getitem, (produced, index))
                item.meta["val"] = produced.meta["val"][index]
                slice_values[name] = item
        base = call_functional(graph, residual_add, slice_values)
    return base


__all__ = ["NUM_HEAD_SLICES", "HeadSlicing", "Producer", "add_in_head_slices"]
