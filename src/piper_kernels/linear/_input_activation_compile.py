"""Projection-independent graph grammar for absorbable linear input activations."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, Match

from piper_kernels.linear import _input_activations as input_activations
from piper_kernels.linear import _preparation_sharing as preparation_sharing

type LinearPattern = Callable[[CallFunction], CallFunction]


def swiglu_product_pattern(
    value: CallFunction,
    gate: CallFunction,
    *,
    promote_gate: bool | None,
    reverse_multiply: bool,
) -> CallFunction:
    """Build ``value * silu(gate)`` with the supported precision semantics."""
    if promote_gate is None:
        silu = CallFunction(torch.ops.aten.silu.default, gate, _users=1)
    else:
        gate_value = (
            CallFunction(
                torch.ops.prims.convert_element_type.default,
                gate,
                torch.float32,
                _users=2,
            )
            if promote_gate
            else gate
        )
        silu = CallFunction(
            torch.ops.aten.div.Tensor,
            gate_value,
            CallFunction(
                torch.ops.aten.add.Tensor,
                CallFunction(
                    torch.ops.aten.exp.default,
                    CallFunction(torch.ops.aten.neg.default, gate_value, _users=1),
                    _users=1,
                ),
                1,
                _users=1,
            ),
            _users=1,
        )
        if promote_gate:
            silu = CallFunction(
                torch.ops.prims.convert_element_type.default,
                silu,
                KeywordArg("logical_dtype"),
                _users=1,
            )
    multiply_args = (silu, value) if reverse_multiply else (value, silu)
    return CallFunction(torch.ops.aten.mul.Tensor, *multiply_args, _users=1)


def gelu_tanh_pattern(
    linear: LinearPattern,
    *,
    promote_input: bool,
) -> CallFunction:
    """Build PyTorch's normalized GELU-tanh decomposition feeding a linear."""
    input_node = KeywordArg("input")
    value = (
        CallFunction(
            torch.ops.prims.convert_element_type.default,
            input_node,
            torch.float32,
            _users=4,
        )
        if promote_input
        else input_node
    )
    half = CallFunction(torch.ops.aten.mul.Tensor, value, 0.5, _users=1)
    square = CallFunction(torch.ops.aten.mul.Tensor, value, value, _users=1)
    cube = CallFunction(torch.ops.aten.mul.Tensor, square, value, _users=1)
    cubic_term = CallFunction(
        torch.ops.aten.mul.Tensor,
        cube,
        input_activations.GELU_TANH_CUBIC_COEFFICIENT,
        _users=1,
    )
    inner = CallFunction(torch.ops.aten.add.Tensor, value, cubic_term, _users=1)
    scaled = CallFunction(
        torch.ops.aten.mul.Tensor,
        inner,
        input_activations.GELU_TANH_SCALE_COEFFICIENT,
        _users=1,
    )
    tanh = CallFunction(torch.ops.aten.tanh.default, scaled, _users=1)
    shifted = CallFunction(torch.ops.aten.add.Tensor, tanh, 1, _users=1)
    activated = CallFunction(torch.ops.aten.mul.Tensor, half, shifted, _users=1)
    if promote_input:
        activated = CallFunction(
            torch.ops.prims.convert_element_type.default,
            activated,
            KeywordArg("logical_dtype"),
            _users=1,
        )
    return linear(activated)


def valid_gelu_tanh(
    match: Match,
    *,
    promote_input: bool,
    input_features: int | torch.SymInt,
) -> bool:
    """Validate the projection-independent dimensions and promotion of GELU-tanh."""
    input_node = match.kwargs["input"]
    if not isinstance(input_node, torch.fx.Node):
        return False
    input_value = preparation_sharing.tensor_metadata(input_node)
    if (
        input_value is None
        or input_value.ndim == 0
        or preparation_sharing.dimension_key(input_value.shape[-1])
        != preparation_sharing.dimension_key(input_features)
    ):
        return False
    if promote_input:
        return match.kwargs["logical_dtype"] is input_value.dtype
    return input_value.dtype is torch.float32


__all__ = [
    "LinearPattern",
    "gelu_tanh_pattern",
    "swiglu_product_pattern",
    "valid_gelu_tanh",
]
