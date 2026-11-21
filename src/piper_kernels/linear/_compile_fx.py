"""Schema-bound FX matching and emission helpers shared by compiler passes."""

from __future__ import annotations

import operator
from typing import cast

import torch
from torch.fx.node import Argument

_aten = torch.ops.aten
_prims = torch.ops.prims
SEMANTIC_LINEAR_NAMES = frozenset(
    {
        "piper_kernels::convrot_int8_linear",
        "piper_kernels::nvfp4_linear",
        "piper_kernels::convrot_nvfp4_linear",
    }
)


def schema_name(node: object) -> str | None:
    """Return ``namespace::op`` when ``node`` calls a PyTorch ``OpOverload``."""
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or not isinstance(node.target, torch._ops.OpOverload)
    ):
        return None
    return node.target._schema.name


def bind_arguments(node: torch.fx.Node) -> dict[str, Argument] | None:
    """Bind positional, keyword, and defaulted arguments by PyTorch operator schema."""
    if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
        return None
    schema = node.target._schema
    positional = [arg for arg in schema.arguments if not arg.kwarg_only]
    if len(node.args) > len(positional):
        return None
    allowed = {arg.name for arg in schema.arguments}
    if not set(node.kwargs).issubset(allowed):
        return None
    values: dict[str, Argument] = {}
    for index, arg in enumerate(positional):
        if index < len(node.args):
            if arg.name in node.kwargs:
                return None
            values[arg.name] = node.args[index]
    values.update(node.kwargs)
    for arg in schema.arguments:
        if arg.name not in values:
            if not arg.has_default_value():
                return None
            default = arg.default_value
            # TorchScript schemas encode non-None ScalarType defaults as C++ enum ints.
            if str(arg.real_type).startswith("ScalarType") and isinstance(default, int):
                continue
            values[arg.name] = default
    return values


def bind_call(node: object, target: torch._ops.OpOverload) -> dict[str, Argument] | None:
    """Bind ``node`` by operator schema when it calls ``target``."""
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not target
    ):
        return None
    return bind_arguments(node)


def same_argument(left: Argument, right: Argument) -> bool:
    """Compare two FX arguments structurally without triggering tensor boolean conversion."""
    if isinstance(left, torch.fx.Node) or isinstance(right, torch.fx.Node):
        return left is right
    if isinstance(left, (int, float, str, bool, type(None))) and isinstance(
        right, (int, float, str, bool, type(None))
    ):
        return left == right
    return False


def is_scalar(value: object, expected: float) -> bool:
    """Return whether ``value`` is a non-boolean numeric scalar equal to ``expected``."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == expected
    )


def binary_args(node: object, target: torch._ops.OpOverload) -> tuple[Argument, Argument] | None:
    """Return ``(self, other)`` for a schema-bound binary op, requiring ``alpha == 1`` on adds."""
    bound = bind_call(node, target)
    if bound is None:
        return None
    if target in (_aten.add.Tensor, _aten.add.Scalar) and not is_scalar(bound.get("alpha"), 1.0):
        return None
    left = bound.get("self")
    right = bound.get("other")
    if left is None or right is None:
        return None
    return left, right


def unary_arg(node: object, target: torch._ops.OpOverload) -> torch.fx.Node | None:
    """Return ``self`` when ``node`` calls unary ``target`` on an FX node."""
    bound = bind_call(node, target)
    if bound is None:
        return None
    source = bound.get("self")
    return source if isinstance(source, torch.fx.Node) else None


def convert_dtype(node: object) -> tuple[torch.fx.Node, torch.dtype] | None:
    """Match ``prims.convert_element_type(a, dtype)`` by schema name."""
    bound = bind_call(node, _prims.convert_element_type.default)
    if bound is None:
        return None
    source = bound.get("a")
    dtype = bound.get("dtype")
    if not isinstance(source, torch.fx.Node) or not isinstance(dtype, torch.dtype):
        return None
    return source, dtype


def match_node_and_scalar(
    node: object, target: torch._ops.OpOverload, scalar: float
) -> torch.fx.Node | None:
    """Match a commutative binary call of one FX node and one numeric scalar."""
    args = binary_args(node, target)
    if args is None:
        return None
    left, right = args
    if isinstance(left, torch.fx.Node) and is_scalar(right, scalar):
        return left
    if isinstance(right, torch.fx.Node) and is_scalar(left, scalar):
        return right
    return None


def emit_tuple_result(
    graph: torch.fx.Graph,
    target: object,
    args: tuple[object, ...],
    values: tuple[torch.Tensor, ...],
    *,
    kwargs: dict[str, object] | None = None,
) -> tuple[torch.fx.Node, ...]:
    """Emit a tuple-returning custom op and metadata-bearing getitems."""
    result = graph.call_function(target, args=args, kwargs=kwargs)  # pyright: ignore[reportArgumentType]
    result.meta["val"] = values
    outputs = []
    for index, value in enumerate(values):
        output = graph.call_function(operator.getitem, args=(result, index))
        output.meta["val"] = value
        outputs.append(output)
    return tuple(outputs)


def ordered_tuple_output_producer(
    outputs: tuple[object, ...],
    target: object,
) -> torch.fx.Node | None:
    """Return the common producer of ordered ``getitem`` tuple outputs."""
    if not outputs or not all(isinstance(output, torch.fx.Node) for output in outputs):
        return None
    nodes = cast(tuple[torch.fx.Node, ...], outputs)
    if any(
        node.op != "call_function"
        or node.target is not operator.getitem
        or node.kwargs
        or len(node.args) != 2
        for node in nodes
    ):
        return None
    producer = nodes[0].args[0]
    if (
        not isinstance(producer, torch.fx.Node)
        or producer.op != "call_function"
        or producer.target is not target
        or any(node.args[0] is not producer for node in nodes)
        or tuple(node.args[1] for node in nodes) != tuple(range(len(nodes)))
    ):
        return None
    return producer


def coalesce_input_casts(graph: torch.fx.Graph) -> bool:
    """Share identical input dtype casts consumed exclusively as semantic linear inputs.

    Post-AOT graphs represent input mutations as distinct values. Match exact
    source/dtype identity and keep casts with external consumers independent,
    so this does not change externally visible storage aliasing.
    """
    casts: dict[tuple[torch.fx.Node, torch.dtype], torch.fx.Node] = {}
    changed = False
    for node in list(graph.nodes):
        converted = convert_dtype(node)
        if converted is None or not node.users:
            continue
        source, dtype = converted
        if any(
            schema_name(user) not in SEMANTIC_LINEAR_NAMES
            or (user_bound := bind_arguments(user)) is None
            or user_bound.get("input") is not node
            or sum(arg is node for arg in user_bound.values()) != 1
            for user in node.users
        ):
            continue
        key = (source, dtype)
        previous = casts.get(key)
        if previous is None:
            casts[key] = node
        else:
            node.replace_all_uses_with(previous)
            graph.erase_node(node)
            changed = True
    if changed:
        graph.lint()
    return changed


__all__ = [
    "SEMANTIC_LINEAR_NAMES",
    "binary_args",
    "bind_arguments",
    "bind_call",
    "coalesce_input_casts",
    "convert_dtype",
    "emit_tuple_result",
    "is_scalar",
    "match_node_and_scalar",
    "ordered_tuple_output_producer",
    "same_argument",
    "schema_name",
    "unary_arg",
]
