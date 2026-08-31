"""FX emission helpers shared by NVFP4 compiler passes."""

from __future__ import annotations

import operator

import torch

from piper_kernels.linear import _preparation_sharing as preparation_sharing

type PreparedInputNodes = tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    tuple[int | torch.SymInt, ...],
]


def emit_prepared_input(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    activation_per_tensor_scale: torch.fx.Node | None,
    dynamic_activation_scale: bool,
) -> PreparedInputNodes:
    """Emit shared NVFP4 activation preparation with complete fake metadata."""
    input_value = preparation_sharing.tensor_metadata(input_node)
    assert input_value is not None
    rows = input_value.numel() // input_value.shape[-1]
    features = input_value.shape[-1]
    scale_rows = ((rows + 127) // 128) * 32
    scale_columns = ((features + 63) // 64) * 16
    values = (
        input_value.new_empty((rows, features // 2), dtype=torch.uint8),
        input_value.new_empty((scale_rows, scale_columns), dtype=torch.float8_e4m3fn),
        input_value.new_empty((), dtype=torch.float32),
    )
    prepared = graph.call_function(
        torch.ops.piper_kernels.nvfp4_prepare_input.default,
        args=(input_node, activation_per_tensor_scale, dynamic_activation_scale),
    )
    prepared.meta["val"] = values
    outputs = []
    for index, value in enumerate(values):
        output = graph.call_function(operator.getitem, args=(prepared, index))
        output.meta["val"] = value
        outputs.append(output)
    return *outputs, tuple(input_value.shape[:-1])


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


__all__ = ["PreparedInputNodes", "emit_prepared_input", "emit_tuple_result"]
