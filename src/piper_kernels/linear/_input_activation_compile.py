"""Imperative schema-bound FX matchers for absorbable linear input activations."""

from __future__ import annotations

import torch

from piper_kernels import _input_activations as input_activations
from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing

_aten = torch.ops.aten


def _match_gelu_cubic_body(
    value: torch.fx.Node, shifted: torch.fx.Node
) -> set[torch.fx.Node] | None:
    """Match ``1 + tanh(scale * (v + cubic * v^3))`` and return its users of ``value``."""
    tanh_node = linear_fx.match_node_and_scalar(shifted, _aten.add.Tensor, 1.0)
    scaled = (
        linear_fx.unary_arg(tanh_node, _aten.tanh.default)
        if tanh_node is not None and len(tanh_node.users) == 1
        else None
    )
    inner = (
        linear_fx.match_node_and_scalar(
            scaled, _aten.mul.Tensor, input_activations.GELU_TANH_SCALE_COEFFICIENT
        )
        if scaled is not None and len(scaled.users) == 1
        else None
    )
    inner_args = (
        linear_fx.binary_args(inner, _aten.add.Tensor)
        if inner is not None and len(inner.users) == 1
        else None
    )
    cubic_term = (
        None
        if inner_args is None
        else inner_args[1]
        if inner_args[0] is value
        else inner_args[0]
        if inner_args[1] is value
        else None
    )
    cube = (
        linear_fx.match_node_and_scalar(
            cubic_term, _aten.mul.Tensor, input_activations.GELU_TANH_CUBIC_COEFFICIENT
        )
        if isinstance(cubic_term, torch.fx.Node) and len(cubic_term.users) == 1
        else None
    )
    cube_args = (
        linear_fx.binary_args(cube, _aten.mul.Tensor)
        if cube is not None and len(cube.users) == 1
        else None
    )
    square = (
        None
        if cube_args is None
        else cube_args[0]
        if cube_args[1] is value
        else cube_args[1]
        if cube_args[0] is value
        else None
    )
    if (
        inner is None
        or cube is None
        or not isinstance(square, torch.fx.Node)
        or len(square.users) != 1
        or linear_fx.binary_args(square, _aten.mul.Tensor) != (value, value)
    ):
        return None
    return {square, cube, inner}


def _match_gelu_body(prod: torch.fx.Node) -> torch.fx.Node | None:
    """Match ``0.5 * v * (1 + tanh(scale * (v + cubic * v^3)))`` and return ``v``."""
    prod_args = linear_fx.binary_args(prod, _aten.mul.Tensor)
    if prod_args is None:
        return None
    for half, shifted in (prod_args, prod_args[::-1]):
        if (
            not isinstance(half, torch.fx.Node)
            or not isinstance(shifted, torch.fx.Node)
            or len(half.users) != 1
            or len(shifted.users) != 1
        ):
            continue
        value = linear_fx.match_node_and_scalar(half, _aten.mul.Tensor, 0.5)
        if value is None:
            continue
        body_users = _match_gelu_cubic_body(value, shifted)
        if body_users is not None and set(value.users) == {half, *body_users}:
            return value
    return None


def match_gelu_tanh_input(  # noqa: PLR0911
    activated: torch.fx.Node,
    *,
    allow_unpromoted_fp32: bool = True,
) -> tuple[torch.fx.Node, torch.dtype | None] | None:
    """Match PyTorch's normalized GELU-tanh decomposition feeding a single consumer.

    Returns ``(source_node, logical_dtype)`` where ``logical_dtype`` is the
    downcast target for promoted low-precision graphs or ``None`` for direct
    FP32 graphs.
    """
    if len(activated.users) != 1:
        return None
    downcast = linear_fx.convert_dtype(activated)
    if downcast is not None:
        prod, logical_dtype = downcast
        if len(prod.users) != 1:
            return None
        value = _match_gelu_body(prod)
        if value is None:
            return None
        upcast = linear_fx.convert_dtype(value)
        if upcast is not None and upcast[1] is torch.float32:
            return upcast[0], logical_dtype
        return None

    if not allow_unpromoted_fp32:
        return None
    value = _match_gelu_body(activated)
    return None if value is None else (value, None)


def valid_gelu_tanh_input(
    source: torch.fx.Node,
    logical_dtype: torch.dtype | None,
    *,
    input_features: int | torch.SymInt,
) -> bool:
    """Validate projection-independent dimensions and promotion of GELU-tanh."""
    input_value = preparation_sharing.tensor_metadata(source)
    if (
        input_value is None
        or input_value.ndim == 0
        or preparation_sharing.dimension_key(input_value.shape[-1])
        != preparation_sharing.dimension_key(input_features)
    ):
        return False
    if logical_dtype is not None:
        return logical_dtype is input_value.dtype
    return input_value.dtype is torch.float32


__all__ = [
    "match_gelu_tanh_input",
    "valid_gelu_tanh_input",
]
