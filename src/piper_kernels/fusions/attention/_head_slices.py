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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates._graph import bind_arguments, call_functional

# Two slices halve K and V at no measured speed cost; past that, other buffers dominate.
NUM_HEAD_SLICES = 2


@dataclass(frozen=True)
class HeadSlicing:
    """Which arguments of an attention residual add, and of its producers, split by head.

    ``producers`` pairs each K or V producer with the add arguments it supplies, in output
    order; its ``producer_rows`` arguments hold one block of rows per head. In the add,
    ``rows`` hold one block of rows per query head, ``columns`` names the output
    projection's weight, whose input columns follow the query heads, and ``once`` is
    added by the first slice only. Each slice's columns must be a multiple of the add's
    ``alignment`` argument, such as a ConvRot group size.

    The add must also take ``base``, ``head_dim``, and ``query_chunk_rows`` as dense
    Piper's does. Each slice adds into the previous slice's ``base``, and its
    ``query_chunk_rows`` is ``NUM_HEAD_SLICES`` times the add's, so each attention launch
    keeps its unsliced rows times heads.
    """

    producers: tuple[tuple[torch._ops.OpOverload, tuple[str, ...]], ...]
    producer_rows: tuple[str, ...]
    rows: tuple[str, ...]
    columns: str
    once: tuple[str, ...]
    alignment: str


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


def _slice_share(graph: torch.fx.Graph, value: Argument, slice_index: int, dim: int) -> Argument:
    """Slice one head slice's share of ``value`` along ``dim``; None stays None."""
    metadata = _metadata(value)
    if not isinstance(metadata, torch.Tensor):
        return None
    size = metadata.shape[dim] // NUM_HEAD_SLICES
    bounds = {"dim": dim, "start": slice_index * size, "end": (slice_index + 1) * size}
    return _call(graph, torch.ops.aten.slice.Tensor, {"self": value, **bounds})


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
    None when K or V has other consumers or the heads don't split evenly into aligned
    slices.
    """
    producers: list[tuple[torch._ops.OpOverload, dict[str, Argument], tuple[str, ...]]] = []
    for target, names in slicing.producers:
        producer = _find_exclusive_producer(values, names, target, update)
        if producer is None:
            return None
        producers.append((target, bind_arguments(producer), names))

    head_dim = cast(int, values["head_dim"])
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

    chunk_rows = values.get("query_chunk_rows")
    if chunk_rows is None:
        (chunk_rows,) = (
            argument.default_value
            for argument in residual_add._schema.arguments
            if argument.name == "query_chunk_rows"
        )
    base = cast(torch.fx.Node, values["base"])
    for slice_index in range(NUM_HEAD_SLICES):
        slice_values = {
            **values,
            "base": base,
            "query_chunk_rows": cast(int, chunk_rows) * NUM_HEAD_SLICES,
            **{name: _slice_share(graph, values[name], slice_index, 0) for name in slicing.rows},
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
        for target, arguments, names in producers:
            sliced = {
                name: _slice_share(graph, arguments[name], slice_index, 0)
                for name in slicing.producer_rows
            }
            produced = _call(graph, target, {**arguments, **sliced})
            for index, name in enumerate(names):
                item = graph.call_function(operator.getitem, (produced, index))
                item.meta["val"] = produced.meta["val"][index]
                slice_values[name] = item
        base = call_functional(graph, residual_add, slice_values)
    return base


__all__ = ["NUM_HEAD_SLICES", "HeadSlicing", "add_in_head_slices"]
