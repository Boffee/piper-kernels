"""Fold dense Q projection, attention, and a token-major ConvRot output linear."""

from typing import Any, cast

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.node import map_arg

from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as sharing
from piper_kernels.linear.convrot.int8 import _backend as linear_backend

from . import output

_RESHAPES = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
)


def _attention_source(node: object) -> torch.fx.Node | None:  # noqa: PLR0911
    """Require an exclusive BHSD -> BSHD -> merged-head path."""
    if not isinstance(node, torch.fx.Node) or node.target not in _RESHAPES:
        return None
    while node.target in (*_RESHAPES, torch.ops.aten.clone.default):
        if len(node.users) != 1 or not node.args or not isinstance(node.args[0], torch.fx.Node):
            return None
        node = node.args[0]
    if len(node.users) != 1 or not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None
    if node.target is torch.ops.aten.transpose.int:
        if node.args[1:] not in ((1, 2), (2, 1)):
            return None
    elif node.target is torch.ops.aten.permute.default:
        if not isinstance(node.args[1], (tuple, list)) or tuple(node.args[1]) != (0, 2, 1, 3):
            return None
    else:
        return None
    attention = node.args[0]
    return (
        attention
        if attention.target is torch.ops.piper_kernels.piper_attention_from_quantized.default
        and len(attention.users) == 1
        and not attention.kwargs
        and len(attention.args) == 12
        else None
    )


def _metadata(node: torch.fx.Node) -> torch.Tensor | int | torch.SymInt:
    value = node.meta.get("val")
    if not isinstance(value, (torch.Tensor, int, torch.SymInt)):
        raise ValueError("missing tensor or size metadata")
    return value


def _fused_arguments(linear: torch.fx.Node) -> tuple[tuple, dict] | None:  # noqa: PLR0911
    if (
        linear.op != "call_function"
        or linear.target is not torch.ops.piper_kernels.convrot_int8_linear.default
        or linear.kwargs
        or len(linear.args) != 7
        or linear.args[5] is not None
        or not isinstance(linear.args[0], torch.fx.Node)
    ):
        return None
    attention = _attention_source(linear.args[0])
    if attention is None:
        return None
    q, qs, *context = attention.args
    if not isinstance(q, torch.fx.Node) or not isinstance(qs, torch.fx.Node):
        return None
    producer = linear_fx.ordered_tuple_output_producer(
        (q, qs), torch.ops.piper_kernels.convrot_int8_piper_project_query.default
    )
    if (
        producer is None
        or len(producer.args) != 10
        or set(producer.users) != {q, qs}
        or any(set(node.users) != {attention} for node in (q, qs))
    ):
        return None
    input_value = sharing.tensor_metadata(linear.args[0])
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
    key, ks, value, multiplier, logs, mean, _query_length, key_length, causal, dtype = context
    arguments = (
        *producer.args,
        key,
        ks,
        value,
        multiplier,
        logs,
        mean,
        key_length,
        causal,
        dtype,
        linear.args[1],
        linear.args[2],
        linear.args[3],
        linear.args[4],
        linear.args[6],
    )
    kwargs = {"head_dim": head_dim}
    try:
        values = cast(tuple[Any, ...], map_arg(arguments, _metadata))
        query_shape = output._validate_inputs(*values, **kwargs)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    if any(
        not statically_known_true(a == b)
        for a, b in zip(query_shape, attention_value.shape, strict=True)
    ):
        return None
    if not all(
        statically_known_true(a == b)
        for a, b in zip(projected.shape, (batch, sequence, values[19].shape[0]), strict=True)
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
