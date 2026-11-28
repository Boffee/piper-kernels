"""Fold dense Q projection, attention, and a token-major ConvRot output linear."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.fx.node import Argument, map_arg

from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as sharing
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8 import _compile_fx

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


def _match_bhsd_transpose(node: torch.fx.Node) -> torch.fx.Node | None:
    """Match ``permute(x, [0, 2, 1, 3])`` or ``transpose(x, 1, 2)``."""
    transpose = linear_fx.bind_call(node, _aten.transpose.int)
    if transpose is not None:
        source = transpose["self"]
        if isinstance(source, torch.fx.Node) and (transpose["dim0"], transpose["dim1"]) in (
            (1, 2),
            (2, 1),
        ):
            return source
        return None
    permute = linear_fx.bind_call(node, _aten.permute.default)
    if (
        permute is not None
        and isinstance(permute["self"], torch.fx.Node)
        and isinstance(permute["dims"], (tuple, list))
        and tuple(permute["dims"]) == (0, 2, 1, 3)
    ):
        return permute["self"]
    return None


def _attention_source(node: object) -> tuple[torch.fx.Node, dict[str, Argument]] | None:
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
    attention = _match_bhsd_transpose(node)
    if not isinstance(attention, torch.fx.Node) or len(attention.users) != 1:
        return None
    attn_bound = linear_fx.bind_call(
        attention, torch.ops.piper_kernels.piper_attention_from_quantized.default
    )
    return None if attn_bound is None else (attention, attn_bound)


def _fused_arguments(linear: torch.fx.Node) -> tuple[tuple[Argument, ...], dict[str, Any]] | None:  # noqa: PLR0911, PLR0912
    bound = _compile_fx.BoundLinearNodes.from_call(linear)
    if bound is None:
        return None
    matched_attention = _attention_source(bound.input)
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
    input_value = sharing.tensor_metadata(bound.input)
    attention_value = sharing.tensor_metadata(attention)
    projected = sharing.tensor_metadata(linear)
    if input_value is None or attention_value is None or projected is None:
        return None
    if attention_value.ndim != 4 or input_value.ndim != 3 or projected.ndim != 3:
        return None
    batch, heads, sequence, head_dim = attention_value.shape
    expected = (batch, sequence, heads * head_dim)
    if not linear_fx.same_shape(input_value.shape, expected):
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
        bound.weight_qdata,
        bound.weight_scale,
        bound.bias,
        bound.group_size,
        bound.input_scale,
    )
    kwargs = {"head_dim": head_dim}
    try:
        values = cast(tuple[Any, ...], map_arg(arguments, linear_fx.require_node_metadata))
        query_shape = output.DenseOutputOperands(*values, head_dim=head_dim).validate()
    except KeyError, TypeError, ValueError, RuntimeError:
        return None
    if not linear_fx.same_shape(query_shape, attention_value.shape):
        return None
    if not linear_fx.same_shape(projected.shape, (batch, sequence, values[20].shape[0])):
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
