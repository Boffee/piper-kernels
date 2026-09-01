"""FX emission helpers shared by NVFP4 compiler passes."""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch
from torch._inductor.pattern_matcher import Match

from piper_kernels.linear import _input_activations as input_activations
from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _layout, _validation

type PreparedInputNodes = tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    tuple[int | torch.SymInt, ...],
]


@dataclass(frozen=True, slots=True)
class SemanticLinearNodes:
    """FX nodes belonging to one semantic NVFP4 linear."""

    input: torch.fx.Node
    weight_qdata: torch.fx.Node
    weight_scale: torch.fx.Node
    weight_per_tensor_scale: torch.fx.Node | None
    activation_per_tensor_scale: torch.fx.Node | None
    bias: torch.fx.Node | None
    dynamic_activation_scale: bool

    @classmethod
    def _from_values(cls, values: tuple[object, ...]) -> SemanticLinearNodes | None:
        if len(values) != 7:
            return None
        input_node, weight_qdata, weight_scale, weight_global, activation_scale, bias, dynamic = (
            values
        )
        if (
            not isinstance(input_node, torch.fx.Node)
            or not isinstance(weight_qdata, torch.fx.Node)
            or not isinstance(weight_scale, torch.fx.Node)
            or any(
                value is not None and not isinstance(value, torch.fx.Node)
                for value in (weight_global, activation_scale, bias)
            )
            or not isinstance(dynamic, bool)
        ):
            return None
        return cls(
            input_node,
            weight_qdata,
            weight_scale,
            weight_global,  # type: ignore[arg-type]
            activation_scale,  # type: ignore[arg-type]
            bias,  # type: ignore[arg-type]
            dynamic,
        )

    @classmethod
    def from_call(cls, node: torch.fx.Node) -> SemanticLinearNodes | None:
        """Parse positional operands from one semantic-linear call node."""
        if node.kwargs:
            return None
        return cls._from_values(node.args)

    @classmethod
    def from_match(cls, match: Match) -> SemanticLinearNodes | None:
        """Parse canonical semantic-linear keywords from a pattern match."""
        return cls._from_values(
            tuple(
                match.kwargs[name]
                for name in (
                    "input",
                    "weight_qdata",
                    "weight_scale",
                    "weight_per_tensor_scale",
                    "activation_per_tensor_scale",
                    "bias",
                    "dynamic_activation_scale",
                )
            )
        )


@dataclass(frozen=True, slots=True)
class PreparedLinearNodes:
    """FX nodes belonging to one canonical prepared NVFP4 linear."""

    input_qdata: torch.fx.Node
    input_scale: torch.fx.Node
    input_per_tensor_scale: torch.fx.Node
    weight_qdata: torch.fx.Node
    weight_scale: torch.fx.Node
    weight_per_tensor_scale: torch.fx.Node | None
    bias: torch.fx.Node | None
    logical_dtype: torch.dtype

    @classmethod
    def _from_values(cls, values: tuple[object, ...]) -> PreparedLinearNodes | None:
        if len(values) != 8:
            return None
        if any(not isinstance(value, torch.fx.Node) for value in values[:5]) or any(
            value is not None and not isinstance(value, torch.fx.Node) for value in values[5:7]
        ):
            return None
        logical_dtype = values[7]
        if not isinstance(logical_dtype, torch.dtype):
            return None
        return cls(*values[:7], logical_dtype)  # type: ignore[arg-type]

    @classmethod
    def from_call(cls, node: torch.fx.Node) -> PreparedLinearNodes | None:
        """Parse positional operands from one prepared-linear call node."""
        if node.kwargs:
            return None
        return cls._from_values(node.args)

    @classmethod
    def from_match(cls, match: Match, prefix: str) -> PreparedLinearNodes | None:
        """Parse a prefixed prepared-linear pattern match."""
        return cls._from_values(
            tuple(
                match.kwargs[f"{prefix}_{suffix}"]
                for suffix in (
                    "input_qdata",
                    "input_scale",
                    "input_per_tensor_scale",
                    "weight_qdata",
                    "weight_scale",
                    "weight_per_tensor_scale",
                    "bias",
                    "logical_dtype",
                )
            )
        )

    def storage_arguments(self) -> tuple[torch.fx.Node | None, ...]:
        """Return prepared-linear storage operands without the logical dtype."""
        return (
            self.input_qdata,
            self.input_scale,
            self.input_per_tensor_scale,
            self.weight_qdata,
            self.weight_scale,
            self.weight_per_tensor_scale,
            self.bias,
        )


def _metadata(node: torch.fx.Node | None) -> torch.Tensor | None:
    return None if node is None else preparation_sharing.tensor_metadata(node)


def validated_semantic_linear(
    operands: SemanticLinearNodes,
    name: str,
) -> tuple[torch.Tensor, _validation.LinearShape] | None:
    """Validate semantic-linear FX metadata and return its input and logical shape."""
    input_value = _metadata(operands.input)
    weight_qdata = _metadata(operands.weight_qdata)
    weight_scale = _metadata(operands.weight_scale)
    weight_global = _metadata(operands.weight_per_tensor_scale)
    activation_scale = _metadata(operands.activation_per_tensor_scale)
    bias = _metadata(operands.bias)
    if input_value is None or weight_qdata is None or weight_scale is None:
        return None
    if any(
        (node is None) != (value is None)
        for node, value in (
            (operands.weight_per_tensor_scale, weight_global),
            (operands.activation_per_tensor_scale, activation_scale),
            (operands.bias, bias),
        )
    ):
        return None
    try:
        shape = _validation.validate_semantic_linear(
            input_value,
            weight_qdata,
            weight_scale,
            weight_global,
            activation_scale,
            bias,
            operands.dynamic_activation_scale,
            name,
        )
    except ValueError:
        return None
    return input_value, shape


def validated_prepared_linear(
    operands: PreparedLinearNodes,
    name: str,
) -> _validation.LinearShape | None:
    """Validate prepared-linear FX metadata and return its logical shape."""
    input_qdata = _metadata(operands.input_qdata)
    input_scale = _metadata(operands.input_scale)
    input_global = _metadata(operands.input_per_tensor_scale)
    weight_qdata = _metadata(operands.weight_qdata)
    weight_scale = _metadata(operands.weight_scale)
    weight_global = _metadata(operands.weight_per_tensor_scale)
    bias = _metadata(operands.bias)
    if any(
        value is None
        for value in (input_qdata, input_scale, input_global, weight_qdata, weight_scale)
    ):
        return None
    if any(
        (node is None) != (value is None)
        for node, value in (
            (operands.weight_per_tensor_scale, weight_global),
            (operands.bias, bias),
        )
    ):
        return None
    assert input_qdata is not None
    assert input_scale is not None
    assert input_global is not None
    assert weight_qdata is not None
    assert weight_scale is not None
    try:
        return _validation.validate_prepared_linear(
            input_qdata,
            input_scale,
            input_global,
            weight_qdata,
            weight_scale,
            weight_global,
            bias,
            operands.logical_dtype,
            name,
        )
    except ValueError:
        return None


def emit_prepared_linear(
    graph: torch.fx.Graph,
    prepared: PreparedInputNodes,
    operands: SemanticLinearNodes,
) -> torch.fx.Node:
    """Emit the prepared equivalent of one validated semantic linear."""
    input_qdata, input_scale, input_per_tensor_scale, leading_shape = prepared
    input_value = preparation_sharing.tensor_metadata(operands.input)
    weight_value = preparation_sharing.tensor_metadata(operands.weight_qdata)
    assert input_value is not None
    assert weight_value is not None
    projected = graph.call_function(
        torch.ops.piper_kernels.nvfp4_linear_prepared.default,
        args=(
            input_qdata,
            input_scale,
            input_per_tensor_scale,
            operands.weight_qdata,
            operands.weight_scale,
            operands.weight_per_tensor_scale,
            operands.bias,
            input_value.dtype,
        ),
    )
    projected.meta["val"] = input_value.new_empty(
        (input_qdata.meta["val"].shape[0], weight_value.shape[0])
    )
    if len(leading_shape) == 1:
        return projected
    return graph.call_function(
        torch.ops.aten.reshape.default,
        args=(projected, (*leading_shape, weight_value.shape[0])),
    )


def emit_prepared_input(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    activation_per_tensor_scale: torch.fx.Node | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
) -> PreparedInputNodes:
    """Emit shared NVFP4 activation preparation with complete fake metadata."""
    input_value = preparation_sharing.tensor_metadata(input_node)
    assert input_value is not None
    rows = input_value.numel() // input_value.shape[-1]
    features = input_value.shape[-1] // input_activations.input_activation_width(activation_fn)
    values = (
        input_value.new_empty(_layout.qdata_shape(rows, features), dtype=torch.uint8),
        input_value.new_empty(_layout.scale_shape(rows, features), dtype=torch.float8_e4m3fn),
        input_value.new_empty((), dtype=torch.float32),
    )
    prepared = graph.call_function(
        torch.ops.piper_kernels.nvfp4_prepare_input.default,
        args=(
            input_node,
            activation_per_tensor_scale,
            dynamic_activation_scale,
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
    "PreparedLinearNodes",
    "SemanticLinearNodes",
    "emit_prepared_input",
    "emit_prepared_linear",
    "validated_prepared_linear",
    "validated_semantic_linear",
]
