"""FX graph-emission helpers shared by ConvRot compiler passes."""

from __future__ import annotations

import operator
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from piper_kernels.linear import _preparation_sharing as preparation_sharing

type PreparedInputNodes = tuple[torch.fx.Node, torch.fx.Node, torch.dtype]


@contextmanager
def canonical_linear_calls(graph: torch.fx.Graph) -> Iterator[None]:
    """Match optional defaults while preserving surviving calls' arity and rewritten inputs."""
    original_arity = {}
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.piper_kernels.convrot_int8_linear.default
            and not node.kwargs
            and len(node.args) in (5, 6)
        ):
            original_arity[node] = len(node.args)
            node.args = node.args + (None,) * (7 - len(node.args))
    try:
        yield
    finally:
        live_nodes = set(graph.nodes)
        for node, arity in original_arity.items():
            if node in live_nodes:
                node.args = node.args[:arity]


def valid_input_scale(node: object, device: torch.device) -> bool:
    """Check static scale metadata before consuming a semantic linear in a fusion."""
    if node is None:
        return True
    if not isinstance(node, torch.fx.Node):
        return False
    value = preparation_sharing.tensor_metadata(node)
    return bool(
        value is not None
        and value.ndim == 0
        and value.dtype is torch.float32
        and value.device == device
        and value.layout is torch.strided
        and not value.requires_grad
    )


def emit_prepared_input(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    group_size: int,
    activation_fn: str | None,
    prepared_shape: tuple[int | torch.SymInt, ...],
    input_scale: torch.fx.Node | None = None,
) -> PreparedInputNodes:
    """Emit shared ConvRot activation preparation with complete fake metadata."""
    input_value = preparation_sharing.tensor_metadata(input_node)
    assert input_value is not None
    input_qdata_value = input_value.new_empty(prepared_shape, dtype=torch.int8)
    row_scales_value = input_value.new_empty(prepared_shape[:-1], dtype=torch.float32)
    prepared = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_prepare_input.default,
        args=(input_node, group_size, activation_fn, input_scale),
    )
    prepared.meta["val"] = (input_qdata_value, row_scales_value)
    input_qdata = graph.call_function(operator.getitem, args=(prepared, 0))
    input_qdata.meta["val"] = input_qdata_value
    row_scales = graph.call_function(operator.getitem, args=(prepared, 1))
    row_scales.meta["val"] = row_scales_value
    return input_qdata, row_scales, input_value.dtype


__all__ = [
    "PreparedInputNodes",
    "canonical_linear_calls",
    "emit_prepared_input",
    "valid_input_scale",
]
