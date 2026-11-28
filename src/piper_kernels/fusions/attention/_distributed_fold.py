"""Shared FX rewrites folding ``projected_query_attention_output -> reduce_scatter``."""

from __future__ import annotations

from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels._distributed import _fx as distributed_fx
from piper_kernels.fusions.residual_updates import _graph as residual_graph
from piper_kernels.linear import _compile_fx as linear_fx


def _match_exclusive_projected_output(
    reduce_scatter_match: distributed_fx.SequenceReduceScatterMatch,
    output_op: torch._ops.OpOverload,
) -> tuple[torch.fx.Node, dict[str, Argument]] | None:
    producer = reduce_scatter_match.input_node
    if producer.op != "call_function" or producer.target is not output_op:
        return None
    consumer = (
        reduce_scatter_match.split_node
        if reduce_scatter_match.split_node is not None
        else reduce_scatter_match.repack_node
    )
    if set(producer.users) != {consumer}:
        return None
    operands = linear_fx.bind_arguments(producer)
    if (
        operands is None
        or operands.get("comm_group_size") is not None
        or operands.get("comm_group_name") is not None
    ):
        return None
    return producer, operands


def _fold_attention_reduce_scatter_residual(
    graph: torch.fx.Graph,
    call: torch.fx.Node,
    *,
    output_op: torch._ops.OpOverload,
    residual_op: torch._ops.OpOverload,
) -> bool:
    residual_match = distributed_fx.match_distributed_residual_add(
        call, residual_graph.RESIDUAL_ADD
    )
    if residual_match is None:
        return False
    base, reduce_scatter_match = residual_match
    matched = _match_exclusive_projected_output(reduce_scatter_match, output_op)
    base_val = linear_fx.node_tensor_metadata(base)
    if matched is None or base_val is None:
        return False
    producer, operands = matched
    if operands.get("output_dtype", torch.bfloat16) is not base_val.dtype:
        return False

    values: dict[str, Argument] = {
        "base": base,
        "gate": cast(Argument, call.kwargs["gate"]),
        "gate_indices": cast(Argument, call.kwargs["gate_indices"]),
        "python_indexing": cast(Argument, call.kwargs["python_indexing"]),
        **{name: value for name, value in operands.items() if name != "output_dtype"},
        "comm_group_size": reduce_scatter_match.group_size,
        "comm_group_name": reduce_scatter_match.group_name,
    }
    with graph.inserting_before(call):
        result = residual_graph.call_functional(graph, residual_op, values)
    for user in list(call.users):
        user.replace_all_uses_with(result)
        graph.erase_node(user)
    graph.erase_node(call)
    distributed_fx.erase_sequence_reduce_scatter(graph, reduce_scatter_match)
    if not producer.users:
        graph.erase_node(producer)
    return True


def _fold_attention_reduce_scatter_standalone(
    graph: torch.fx.Graph,
    wait_node: torch.fx.Node,
    *,
    output_op: torch._ops.OpOverload,
) -> bool:
    reduce_scatter_match = distributed_fx.match_sequence_reduce_scatter(wait_node)
    if reduce_scatter_match is None:
        return False
    matched = _match_exclusive_projected_output(reduce_scatter_match, output_op)
    if matched is None:
        return False
    producer, operands = matched
    values = {
        **operands,
        "comm_group_size": reduce_scatter_match.group_size,
        "comm_group_name": reduce_scatter_match.group_name,
    }
    with graph.inserting_before(wait_node):
        replacement = linear_fx.call_in_schema_order(graph, output_op, values)
    replacement.meta = wait_node.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    wait_node.replace_all_uses_with(replacement)
    distributed_fx.erase_sequence_reduce_scatter(graph, reduce_scatter_match)
    if not producer.users:
        graph.erase_node(producer)
    return True


def fold_distributed_projected_output(
    graph: torch.fx.Graph,
    *,
    output_op: torch._ops.OpOverload,
    residual_op: torch._ops.OpOverload,
) -> bool:
    """Fold ``output_op -> reduce_scatter -> (optional) residual_add_``."""
    if not distributed_fx.has_c10d_collectives(graph):
        return False

    changed = False
    for node in list(graph.nodes):
        changed = (
            _fold_attention_reduce_scatter_residual(
                graph,
                node,
                output_op=output_op,
                residual_op=residual_op,
            )
            or changed
        )
    for node in list(graph.nodes):
        changed = (
            _fold_attention_reduce_scatter_standalone(
                graph,
                node,
                output_op=output_op,
            )
            or changed
        )

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__ = ["fold_distributed_projected_output"]
