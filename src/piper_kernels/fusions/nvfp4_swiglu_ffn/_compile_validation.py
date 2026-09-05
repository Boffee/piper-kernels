"""Shared compiler validation for semantic NVFP4 SwiGLU FFNs."""

from __future__ import annotations

import torch
from torch._inductor.pattern_matcher import Match

from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx


def _dimension_matches(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


def projection_call_matches(node: torch.fx.Node, match: Match, prefix: str) -> bool:
    """Identify a captured projection by all operands, not potentially shared weights."""
    if node.kwargs:
        return False
    convrot = node.target == torch.ops.piper_kernels.convrot_nvfp4_linear.default
    if convrot != (f"{prefix}_group_size" in match.kwargs):
        return False
    names = (
        "weight_qdata",
        "weight_scale",
        "weight_per_tensor_scale",
        "activation_per_tensor_scale",
        "bias",
        "dynamic_activation_scale",
        *(("group_size",) if convrot else ()),
        "high_first",
    )
    expected = tuple(
        match.kwargs.get(f"{prefix}_{name}", False)
        if name == "high_first"
        else match.kwargs[f"{prefix}_{name}"]
        for name in names
    )
    arguments = node.args[1:]
    if len(arguments) == len(expected) - 1:
        arguments = (*arguments, False)
    return arguments == expected


def valid_semantic_ffn(
    match: Match,
    gate: nvfp4_compile_fx.SemanticLinearNodes,
    value: nvfp4_compile_fx.SemanticLinearNodes,
    down: nvfp4_compile_fx.SemanticLinearNodes,
    *,
    promote_gate: bool | None,
) -> bool:
    """Validate common NVFP4 shapes, precision, and preparation compatibility."""
    validated_gate = nvfp4_compile_fx.validated_semantic_linear(
        gate,
        "NVFP4 FFN compiler gate projection",
    )
    validated_value = nvfp4_compile_fx.validated_semantic_linear(
        value,
        "NVFP4 FFN compiler value projection",
    )
    validated_down = nvfp4_compile_fx.validated_semantic_linear(
        down,
        "NVFP4 FFN compiler down projection",
    )
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    if (
        validated_gate is None
        or validated_value is None
        or validated_down is None
        or output_value is None
    ):
        return False
    input_value, gate_shape = validated_gate
    value_input, value_shape = validated_value
    _, down_shape = validated_down
    if promote_gate is True and match.kwargs["logical_dtype"] is not input_value.dtype:
        return False
    return bool(
        gate.input is value.input
        and input_value.dtype is torch.bfloat16
        and value_input.dtype is input_value.dtype
        and _dimension_matches(gate_shape.rows, value_shape.rows)
        and _dimension_matches(gate_shape.rows, down_shape.rows)
        and _dimension_matches(gate_shape.input_features, value_shape.input_features)
        and _dimension_matches(gate_shape.output_features, value_shape.output_features)
        and _dimension_matches(down_shape.input_features, gate_shape.output_features)
        and gate.high_first == value.high_first
        and output_value.dtype is input_value.dtype
        and output_value.device == input_value.device
        and output_value.ndim == input_value.ndim
        and all(
            _dimension_matches(output_dimension, input_dimension)
            for output_dimension, input_dimension in zip(
                output_value.shape[:-1],
                input_value.shape[:-1],
                strict=True,
            )
        )
        and _dimension_matches(output_value.shape[-1], down_shape.output_features)
    )


__all__ = ["valid_semantic_ffn"]
