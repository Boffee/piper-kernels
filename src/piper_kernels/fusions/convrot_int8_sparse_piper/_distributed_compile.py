"""Distributed FX rewrites for fused ConvRot INT8 sparse Piper attention."""

from __future__ import annotations

import operator
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels._distributed import _fx as distributed_fx
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.fusions.attention import _distributed_fold as distributed_fold
from piper_kernels.fusions.residual_updates import _graph as residual_graph
from piper_kernels.linear.convrot.int8 import _compile_fx

from . import (
    distributed_output as distributed_output_operators,  # noqa: F401 - registers distributed output ops
)
from . import kv_gather as kv_gather_operators  # noqa: F401 - registers distributed KV gather

_ops = torch.ops.piper_kernels
_PROJECT_KEY = _ops.convrot_int8_sparse_piper_project_key.default
_PROJECT_VALUE = _ops.convrot_int8_sparse_piper_project_value.default
_PREPARE_GATHER_PROJECT_KV = _ops.convrot_int8_sparse_piper_prepare_gather_project_kv.default
_PROJECTED_QUERY_OUTPUT = _ops.convrot_int8_sparse_piper_projected_query_attention_output.default
_PROJECTED_QUERY_RS_RESIDUAL_ADD = (
    _ops.convrot_int8_sparse_piper_projected_query_attention_reduce_scatter_residual_add_.default
)
_PROJECTED_QUERY_RS = (
    _ops.convrot_int8_sparse_piper_projected_query_attention_reduce_scatter.default
)


def fold_distributed_attention_output(graph: torch.fx.Graph) -> bool:
    """Fold ``projected_query_attention_output -> reduce_scatter -> (optional) residual_add_``."""
    return distributed_fold.fold_distributed_projected_output(
        graph,
        output_op=_PROJECTED_QUERY_OUTPUT,
        reduce_scatter_op=_PROJECTED_QUERY_RS,
        reduce_scatter_residual_op=_PROJECTED_QUERY_RS_RESIDUAL_ADD,
    )


def _find_exclusive_kv_pair(
    qdata_items: list[torch.fx.Node],
    scale_items: list[torch.fx.Node],
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    qdata_set = set(qdata_items)
    scale_set = set(scale_items)
    key_nodes: list[torch.fx.Node] = []
    value_nodes: list[torch.fx.Node] = []
    for item in qdata_items:
        for user in item.users:
            if user.op != "call_function":
                continue
            if user.target is _PROJECT_KEY and user not in key_nodes:
                key_nodes.append(user)
            elif user.target is _PROJECT_VALUE and user not in value_nodes:
                value_nodes.append(user)
    if len(key_nodes) != 1 or len(value_nodes) != 1:
        return None
    key_node, value_node = key_nodes[0], value_nodes[0]
    for item in scale_items:
        for user in item.users:
            if (
                user.op == "call_function"
                and user.target in (_PROJECT_KEY, _PROJECT_VALUE)
                and user not in (key_node, value_node)
            ):
                return None
    key_operands = residual_graph.bind_arguments(key_node)
    value_operands = residual_graph.bind_arguments(value_node)
    if (
        key_operands.get("input_qdata") not in qdata_set
        or key_operands.get("input_scale") not in scale_set
    ):
        return None
    if (
        value_operands.get("input_qdata") not in qdata_set
        or value_operands.get("input_scale") not in scale_set
    ):
        return None
    return key_node, value_node


def _extract_kv_meta_vals(
    prepare_node: torch.fx.Node,
    key_node: torch.fx.Node,
    value_node: torch.fx.Node,
) -> tuple[object, ...] | None:
    prep_val = prepare_node.meta.get("val")
    key_val = key_node.meta.get("val")
    value_val = value_node.meta.get("val")
    if (
        isinstance(prep_val, (tuple, list))
        and len(prep_val) == 2
        and isinstance(key_val, (tuple, list))
        and len(key_val) == 6
        and isinstance(value_val, (tuple, list))
        and len(value_val) == 3
    ):
        return (*prep_val, *key_val, *value_val)
    return None


def _match_prepare_kv_bundle(
    prepare_node: torch.fx.Node,
) -> (
    tuple[
        distributed_fx.SequenceAllGatherMatch,
        int,
        torch.fx.Node | None,
        list[torch.fx.Node],
        tuple[list[torch.fx.Node], list[torch.fx.Node]],
        tuple[torch.fx.Node, list[list[torch.fx.Node]], torch.fx.Node, list[list[torch.fx.Node]]],
    ]
    | None
):
    parsed = _compile_fx.match_prepare_input(prepare_node)
    if parsed is None:
        return None
    source, prepare_group_size, input_scale = parsed

    all_gather_match = distributed_fx.match_sequence_all_gather(source)
    if all_gather_match is None:
        return None
    sym_size_users = distributed_fx.collect_sym_size_users(
        all_gather_match.output_node, prepare_node
    )
    prep_outputs = distributed_fx.collect_indexed_outputs(prepare_node, 2)
    kv_pair = (
        None
        if sym_size_users is None or prep_outputs is None
        else _find_exclusive_kv_pair(*prep_outputs)
    )
    if kv_pair is None:
        return None
    assert sym_size_users is not None
    assert prep_outputs is not None
    qdata_items, scale_items = prep_outputs
    key_node, value_node = kv_pair
    key_outputs = distributed_fx.collect_indexed_outputs(key_node, 6)
    value_outputs = distributed_fx.collect_indexed_outputs(value_node, 3)
    if key_outputs is None or value_outputs is None:
        return None
    return (
        all_gather_match,
        prepare_group_size,
        input_scale,
        sym_size_users,
        (qdata_items, scale_items),
        (key_node, key_outputs, value_node, value_outputs),
    )


def _fold_one_prepare_gather_project_kv(
    graph: torch.fx.Graph,
    prepare_node: torch.fx.Node,
) -> bool:
    bundle = _match_prepare_kv_bundle(prepare_node)
    if bundle is None:
        return False
    (
        all_gather_match,
        prepare_group_size,
        input_scale,
        sym_size_users,
        prep_tuple,
        kv_tuple,
    ) = bundle
    qdata_items, scale_items = prep_tuple
    key_node, key_outputs, value_node, value_outputs = kv_tuple

    key_operands = residual_graph.bind_arguments(key_node)
    value_operands = residual_graph.bind_arguments(value_node)
    replaced = {key_node, value_node}
    external_consumers = [
        *(
            user
            for bucket in (qdata_items, scale_items)
            for item in bucket
            for user in item.users
            if user not in replaced
        ),
        *(
            user
            for bucket in (*key_outputs, *value_outputs)
            for item in bucket
            for user in item.users
        ),
    ]
    required_defs: list[torch.fx.Node] = [
        all_gather_match.input_node,
        *([input_scale] if isinstance(input_scale, torch.fx.Node) else []),
        *(
            value
            for op_map in (key_operands, value_operands)
            for name, value in op_map.items()
            if name not in ("input_qdata", "input_scale") and isinstance(value, torch.fx.Node)
        ),
    ]
    insert_point = distributed_fx.find_insertion_point(graph, required_defs, external_consumers)
    if insert_point is None:
        return False

    meta_vals = _extract_kv_meta_vals(prepare_node, key_node, value_node)
    values: dict[str, Argument] = {
        "input": all_gather_match.input_node,
        "comm_group_size": all_gather_match.group_size,
        "comm_group_name": all_gather_match.group_name,
        "prepare_group_size": prepare_group_size,
        "input_scale": cast(Argument, input_scale),
        "k_weight_qdata": key_operands["weight_qdata"],
        "k_weight_scale": key_operands["weight_scale"],
        "k_norm_weight": key_operands["norm_weight"],
        "cos": key_operands["cos"],
        "sin": key_operands["sin"],
        "k_norm_epsilon": key_operands["norm_epsilon"],
        "routing_mode": key_operands["routing_mode"],
        "k_block_lengths": key_operands.get("block_lengths", None),
        "k_bias": key_operands.get("bias", None),
        "k_head_dim": key_operands.get("head_dim", None),
        "v_weight_qdata": value_operands["weight_qdata"],
        "v_weight_scale": value_operands["weight_scale"],
        "v_block_lengths": value_operands.get("block_lengths", None),
        "v_head_dim": value_operands.get("head_dim", 128),
        "v_bias": value_operands.get("bias", None),
        "v_value_format": value_operands.get("value_format", int(ValueFormat.TRANSPOSED)),
    }

    distributed_fx.rewrite_sym_size_users(graph, sym_size_users, all_gather_match)
    with graph.inserting_before(insert_point):
        fused = distributed_fx.call_in_schema_order(graph, _PREPARE_GATHER_PROJECT_KV, values)
        if meta_vals is not None:
            fused.meta["val"] = meta_vals
        for out_idx, old_items in enumerate(
            [qdata_items, scale_items, *key_outputs, *value_outputs]
        ):
            if not old_items:
                continue
            new_item = graph.call_function(operator.getitem, args=(fused, out_idx))
            if meta_vals is not None:
                new_item.meta["val"] = meta_vals[out_idx]
            for old_item in old_items:
                old_item.replace_all_uses_with(new_item)

    for bucket in (*key_outputs, *value_outputs, qdata_items, scale_items):
        for old_item in bucket:
            if not old_item.users:
                graph.erase_node(old_item)
    for dead_node in (key_node, value_node, prepare_node):
        if not dead_node.users:
            graph.erase_node(dead_node)
    distributed_fx.erase_sequence_all_gather(graph, all_gather_match)
    return True


def fold_prepare_gather_project_kv(graph: torch.fx.Graph) -> bool:
    """Fold ``all_gather -> prepare_input -> (project_key, project_value)``."""
    if not distributed_fx.has_c10d_collectives(graph):
        return False

    changed = False
    for node in list(graph.nodes):
        changed = _fold_one_prepare_gather_project_kv(graph, node) or changed

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__ = ["fold_distributed_attention_output", "fold_prepare_gather_project_kv"]
