"""FX graph-emission helpers shared by quantized linear compilers."""

from __future__ import annotations

import operator

import torch


def emit_tuple_result(
    graph: torch.fx.Graph,
    target: object,
    args: tuple[object, ...],
    values: tuple[torch.Tensor, ...],
) -> tuple[torch.fx.Node, ...]:
    """Emit a tuple-returning custom op and metadata-bearing getitems."""
    result = graph.call_function(target, args=args)  # pyright: ignore[reportArgumentType]
    result.meta["val"] = values
    outputs = []
    for index, value in enumerate(values):
        output = graph.call_function(operator.getitem, args=(result, index))
        output.meta["val"] = value
        outputs.append(output)
    return tuple(outputs)


__all__ = ["emit_tuple_result"]
