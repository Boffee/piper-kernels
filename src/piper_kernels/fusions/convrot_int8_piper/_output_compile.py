"""Fold dense Q projection, attention, and a token-major ConvRot output linear."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.node import Argument, map_arg

from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as sharing
from piper_kernels.linear.convrot.int8 import _backend as linear_backend

from . import output

_aten = torch.ops.aten
_RESHAPES = (
    _aten.reshape.default,
    _aten.view.default,
    _aten._unsafe_view.default,
)
_QUERY_PROJECTION_ARG_NAMES = (
    "input_qdata",
    "input_scale",
    "weight_qdata",
    "weight_scale",
    "norm_weight",
    "cos",
    "sin",
    "norm_epsilon",
    "softmax_scale",
    "bias",
)


def _attention_source(node: object) -> tuple[torch.fx.Node, dict[str, Argument]] | None:  # noqa: PLR0911
    """Require an exclusive BHSD -> BSHD -> merged-head path."""
    if not isinstance(node, torch.fx.Node) or node.target not in _RESHAPES:
        return None
    while node.op == "call_function" and node.target in (*_RESHAPES, _aten.clone.default):
        if len(node.users) != 1:
            return None
        bound = linear_fx.bind_arguments(node)
        if bound is None or not isinstance(bound.get("self"), torch.fx.Node):
            return None
        node = bound["self"]
        assert isinstance(node, torch.fx.Node)
    if len(node.users) != 1:
        return None
    transpose = linear_fx.bind_call(node, _aten.transpose.int)
    if transpose is not None:
        if (transpose["dim0"], transpose["dim1"]) not in ((1, 2), (2, 1)):
            return None
        attention = transpose["self"]
    else:
        permute = linear_fx.bind_call(node, _aten.permute.default)
        if (
            permute is None
            or not isinstance(permute["dims"], (tuple, list))
            or tuple(permute["dims"]) != (0, 2, 1, 3)
        ):
            return None
        attention = permute["self"]
    if not isinstance(attention, torch.fx.Node) or len(attention.users) != 1:
        return None
    attn_bound = linear_fx.bind_call(
        attention, torch.ops.piper_kernels.piper_attention_from_quantized.default
    )
    return None if attn_bound is None else (attention, attn_bound)


def _metadata(node: torch.fx.Node) -> torch.Tensor | int | torch.SymInt:
    value = node.meta.get("val")
    if not isinstance(value, (torch.Tensor, int, torch.SymInt)):
        raise ValueError("missing tensor or size metadata")
    return value


def _fused_arguments(linear: torch.fx.Node) -> tuple[tuple[Argument, ...], dict[str, Any]] | None:  # noqa: PLR0911, PLR0912
    linear_bound = linear_fx.bind_call(linear, torch.ops.piper_kernels.convrot_int8_linear.default)
    if linear_bound is None or linear_bound["activation_fn"] is not None:
        return None
    linear_input = linear_bound["input"]
    if not isinstance(linear_input, torch.fx.Node):
        return None
    matched_attention = _attention_source(linear_input)
    if matched_attention is None:
        return None
    attention, attn_bound = matched_attention
    q = attn_bound["query"]
    qs = attn_bound["query_scale"]
    if not isinstance(q, torch.fx.Node) or not isinstance(qs, torch.fx.Node):
        return None
    producer = linear_fx.ordered_tuple_output_producer(
        (q, qs), torch.ops.piper_kernels.convrot_int8_piper_project_query.default
    )
    if (
        producer is None
        or set(producer.users) != {q, qs}
        or any(set(node.users) != {attention} for node in (q, qs))
    ):
        return None
    producer_bound = linear_fx.bind_arguments(producer)
    if producer_bound is None:
        return None
    input_value = sharing.tensor_metadata(linear_input)
    attention_value = sharing.tensor_metadata(attention)
    projected = sharing.tensor_metadata(linear)
    if input_value is None or attention_value is None or projected is None:
        return None
    if attention_value.ndim != 4 or input_value.ndim != 3 or projected.ndim != 3:
        return None
    batch, heads, sequence, head_dim = attention_value.shape
    expected = (batch, sequence, heads * head_dim)
    if any(
        not statically_known_true(a == b) for a, b in zip(input_value.shape, expected, strict=True)
    ):
        return None
    if (
        input_value.dtype is not attention_value.dtype
        or projected.dtype is not input_value.dtype
        or input_value.device != attention_value.device
        or projected.device != input_value.device
    ):
        return None
    # Native operands were validated by the preceding projection/attention rewrite.
    arguments: tuple[Argument, ...] = (
        *(producer_bound[name] for name in _QUERY_PROJECTION_ARG_NAMES),
        attn_bound["key"],
        attn_bound["key_scale"],
        attn_bound["key_center"],
        attn_bound["value"],
        attn_bound["multiplier"],
        attn_bound["log_scale"],
        attn_bound["value_mean"],
        attn_bound["key_length"],
        attn_bound["is_causal"],
        attn_bound["output_dtype"],
        linear_bound["weight_qdata"],
        linear_bound["weight_scale"],
        linear_bound["bias"],
        linear_bound["group_size"],
        linear_bound["input_scale"],
    )
    kwargs = {"head_dim": head_dim}
    try:
        values = cast(tuple[Any, ...], map_arg(arguments, _metadata))
        query_shape = output._validate_inputs(*values, **kwargs)
    except KeyError, TypeError, ValueError, RuntimeError:
        return None
    if any(
        not statically_known_true(a == b)
        for a, b in zip(query_shape, attention_value.shape, strict=True)
    ):
        return None
    if not all(
        statically_known_true(a == b)
        for a, b in zip(projected.shape, (batch, sequence, values[20].shape[0]), strict=True)
    ):
        return None
    if linear_backend.select_linear_backend(input_value) is None:
        return None
    return arguments, kwargs


def fold_attention_output(graph: torch.fx.Graph) -> bool:
    """Fuse complete Q/K/V attention regions with an exclusive output projection."""
    changed = False
    for node in list(graph.nodes):
        matched = _fused_arguments(node)
        if matched is None:
            continue
        arguments, kwargs = matched
        with graph.inserting_before(node):
            replacement = graph.call_function(
                torch.ops.piper_kernels.convrot_int8_piper_projected_query_attention_output.default,
                args=arguments,
                kwargs=kwargs,
            )
        replacement.meta = node.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True
    return changed
