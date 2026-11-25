"""FX graph-emission helpers shared by ConvRot compiler passes."""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch

from piper_kernels.linear import _bias
from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing

type PreparedInputNodes = tuple[torch.fx.Node, torch.fx.Node, torch.dtype]


@dataclass(frozen=True, slots=True)
class BoundLinearNodes:
    """One schema-bound unactivated ``convrot_int8_linear`` call."""

    node: torch.fx.Node
    input: torch.fx.Node
    weight_qdata: torch.fx.Node
    weight_scale: torch.fx.Node
    bias: torch.fx.Node | None
    group_size: int
    input_scale: torch.fx.Node | None

    @classmethod
    def from_call(cls, node: torch.fx.Node) -> BoundLinearNodes | None:
        """Bind an unactivated ConvRot INT8 linear call, verifying node and scalar types."""
        bound = linear_fx.bind_call(node, torch.ops.piper_kernels.convrot_int8_linear.default)
        if bound is None or bound["activation_fn"] is not None or bound["block_scales"] is not None:
            return None
        input_node = bound["input"]
        weight_qdata = bound["weight_qdata"]
        weight_scale = bound["weight_scale"]
        bias = bound["bias"]
        group_size = bound["group_size"]
        input_scale = bound["input_scale"]
        if (
            not isinstance(input_node, torch.fx.Node)
            or not isinstance(weight_qdata, torch.fx.Node)
            or not isinstance(weight_scale, torch.fx.Node)
            or (bias is not None and not isinstance(bias, torch.fx.Node))
            or isinstance(group_size, bool)
            or not isinstance(group_size, int)
            or (input_scale is not None and not isinstance(input_scale, torch.fx.Node))
        ):
            return None
        return cls(
            node=node,
            input=input_node,
            weight_qdata=weight_qdata,
            weight_scale=weight_scale,
            bias=bias,
            group_size=group_size,
            input_scale=input_scale,
        )


def _has_static_shape(tensor: torch.Tensor, expected: tuple[int, ...]) -> bool:
    return len(tensor.shape) == len(expected) and all(
        type(dim) is int and dim == expected_dim
        for dim, expected_dim in zip(tensor.shape, expected, strict=True)
    )


def _valid_linear_bias(
    bias_node: torch.fx.Node | None,
    output_features: int,
    device: torch.device,
) -> bool:
    """Validate optional contiguous bias metadata for a fused ConvRot INT8 projection."""
    if bias_node is None:
        return True
    bias = preparation_sharing.tensor_metadata(bias_node)
    return bool(
        bias is not None
        and _has_static_shape(bias, (output_features,))
        and _bias.is_supported_dtype(bias.dtype)
        and bias.device == device
        and bias.layout is torch.strided
        and bias.is_contiguous()
    )


def valid_linear_weight_and_bias(
    weight_qdata_node: torch.fx.Node,
    weight_scale_node: torch.fx.Node,
    bias_node: torch.fx.Node | None,
    *,
    input_features: int,
    output_features: int,
    device: torch.device,
) -> bool:
    """Validate contiguous INT8 weight, FP32 row scale, and optional bias metadata."""
    weight = preparation_sharing.tensor_metadata(weight_qdata_node)
    scale = preparation_sharing.tensor_metadata(weight_scale_node)
    if weight is None or scale is None:
        return False
    return (
        weight.dtype is torch.int8
        and _has_static_shape(weight, (output_features, input_features))
        and scale.dtype is torch.float32
        and _has_static_shape(scale, (output_features, 1))
        and weight.device == device
        and scale.device == device
        and weight.layout is torch.strided
        and scale.layout is torch.strided
        and weight.is_contiguous()
        and scale.is_contiguous()
        and _valid_linear_bias(bias_node, output_features, device)
    )


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


def match_prepare_input(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, int, torch.fx.Node | None] | None:
    """Return the source, group size, and optional scale for unactivated preparation."""
    bound = linear_fx.bind_call(node, torch.ops.piper_kernels.convrot_int8_prepare_input.default)
    if bound is None or bound["activation_fn"] is not None:
        return None
    source = bound["input"]
    group_size = bound["group_size"]
    input_scale = bound["input_scale"]
    if (
        not isinstance(source, torch.fx.Node)
        or type(group_size) is not int
        or group_size < 1
        or (input_scale is not None and not isinstance(input_scale, torch.fx.Node))
    ):
        return None
    return source, group_size, input_scale


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
    "BoundLinearNodes",
    "PreparedInputNodes",
    "emit_prepared_input",
    "match_prepare_input",
    "valid_input_scale",
    "valid_linear_weight_and_bias",
]
