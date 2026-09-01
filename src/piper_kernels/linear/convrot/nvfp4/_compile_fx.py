"""FX helpers for ConvRot NVFP4 compiler passes."""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch

from piper_kernels.linear import _input_activations as input_activations
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot._rotation import validate_group_size
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation

type PreparedInputNodes = nvfp4_compile_fx.PreparedInputNodes


@dataclass(frozen=True, slots=True)
class SemanticLinearNodes:
    """A canonical NVFP4 semantic linear plus its ConvRot group."""

    linear: nvfp4_compile_fx.SemanticLinearNodes
    group_size: int

    @classmethod
    def from_call(cls, node: torch.fx.Node) -> SemanticLinearNodes | None:
        """Parse one positional ConvRot NVFP4 semantic-linear call."""
        if node.kwargs or len(node.args) != 8:
            return None
        group_size = node.args[-1]
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            return None
        linear = nvfp4_compile_fx.SemanticLinearNodes.from_values(node.args[:-1])
        return None if linear is None else cls(linear, group_size)


def validated_semantic_linear(
    operands: SemanticLinearNodes,
    name: str,
) -> tuple[torch.Tensor, nvfp4_validation.LinearShape] | None:
    """Validate semantic NVFP4 metadata and ConvRot-specific invariants."""
    try:
        validate_group_size(operands.group_size)
    except ValueError:
        return None
    validated = nvfp4_compile_fx.validated_semantic_linear(operands.linear, name)
    if validated is None:
        return None
    input_value, shape = validated
    if (
        input_value.dtype not in (torch.float16, torch.bfloat16)
        or not isinstance(shape.input_features, int)
        or shape.input_features % operands.group_size
    ):
        return None
    return input_value, shape


def emit_prepared_input(
    graph: torch.fx.Graph,
    operands: SemanticLinearNodes,
    *,
    input_node: torch.fx.Node | None = None,
    activation_fn: str | None = None,
) -> PreparedInputNodes:
    """Emit shared ConvRot NVFP4 preparation with complete fake metadata."""
    input_node = operands.linear.input if input_node is None else input_node
    input_value = preparation_sharing.tensor_metadata(input_node)
    assert input_value is not None
    rows = input_value.numel() // input_value.shape[-1]
    input_features = input_value.shape[-1] // input_activations.input_activation_width(
        activation_fn
    )
    values = (
        input_value.new_empty(
            nvfp4_layout.qdata_shape(rows, input_features),
            dtype=torch.uint8,
        ),
        input_value.new_empty(
            nvfp4_layout.scale_shape(rows, input_features),
            dtype=torch.float8_e4m3fn,
        ),
        input_value.new_empty((), dtype=torch.float32),
    )
    prepared = graph.call_function(
        torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default,
        args=(
            input_node,
            operands.linear.activation_per_tensor_scale,
            operands.linear.dynamic_activation_scale,
            operands.group_size,
            activation_fn,
        ),
    )
    prepared.meta["val"] = values
    outputs = []
    for index, value in enumerate(values):
        output = graph.call_function(operator.getitem, args=(prepared, index))
        output.meta["val"] = value
        outputs.append(output)
    return *outputs, tuple(input_value.shape[:-1])


__all__ = [
    "PreparedInputNodes",
    "SemanticLinearNodes",
    "emit_prepared_input",
    "validated_semantic_linear",
]
