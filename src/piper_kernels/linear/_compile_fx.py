"""FX graph-emission helpers shared by quantized linear compilers."""

from __future__ import annotations

import operator
from typing import cast

import torch


def emit_tuple_result(
    graph: torch.fx.Graph,
    target: object,
    args: tuple[object, ...],
    values: tuple[torch.Tensor, ...],
    *,
    kwargs: dict[str, object] | None = None,
) -> tuple[torch.fx.Node, ...]:
    """Emit a tuple-returning custom op and metadata-bearing getitems."""
    result = graph.call_function(target, args=args, kwargs=kwargs)  # pyright: ignore[reportArgumentType]
    result.meta["val"] = values
    outputs = []
    for index, value in enumerate(values):
        output = graph.call_function(operator.getitem, args=(result, index))
        output.meta["val"] = value
        outputs.append(output)
    return tuple(outputs)


def ordered_tuple_output_producer(
    outputs: tuple[object, ...],
    target: object,
) -> torch.fx.Node | None:
    """Return the common producer of ordered ``getitem`` tuple outputs."""
    if not outputs or not all(isinstance(output, torch.fx.Node) for output in outputs):
        return None
    nodes = cast(tuple[torch.fx.Node, ...], outputs)
    if any(node.target is not operator.getitem or len(node.args) != 2 for node in nodes):
        return None
    producer = nodes[0].args[0]
    if (
        not isinstance(producer, torch.fx.Node)
        or producer.target is not target
        or any(node.args[0] is not producer for node in nodes)
        or tuple(node.args[1] for node in nodes) != tuple(range(len(nodes)))
    ):
        return None
    return producer


__all__ = ["emit_tuple_result", "ordered_tuple_output_producer"]
