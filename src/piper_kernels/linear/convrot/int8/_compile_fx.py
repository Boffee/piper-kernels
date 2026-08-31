"""FX graph-emission helpers shared by ConvRot compiler passes."""

from __future__ import annotations

import operator

import torch

from piper_kernels.linear import _preparation_sharing as preparation_sharing

type PreparedInputNodes = tuple[torch.fx.Node, torch.fx.Node, torch.dtype]


def emit_prepared_input(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    group_size: int,
    activation_fn: str | None,
    prepared_shape: tuple[int | torch.SymInt, ...],
) -> PreparedInputNodes:
    """Emit shared ConvRot activation preparation with complete fake metadata."""
    input_value = preparation_sharing.tensor_metadata(input_node)
    assert input_value is not None
    input_qdata_value = input_value.new_empty(prepared_shape, dtype=torch.int8)
    input_scale_value = input_value.new_empty(prepared_shape[:-1], dtype=torch.float32)
    prepared = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_prepare_input.default,
        args=(input_node, group_size, activation_fn),
    )
    prepared.meta["val"] = (input_qdata_value, input_scale_value)
    input_qdata = graph.call_function(operator.getitem, args=(prepared, 0))
    input_qdata.meta["val"] = input_qdata_value
    input_scale = graph.call_function(operator.getitem, args=(prepared, 1))
    input_scale.meta["val"] = input_scale_value
    return input_qdata, input_scale, input_value.dtype


__all__ = ["PreparedInputNodes", "emit_prepared_input"]
