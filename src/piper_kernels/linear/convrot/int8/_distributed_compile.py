"""Compiler pass commuting ConvRot INT8 input preparation ahead of sequence all-gather.

Per-row Hadamard rotation and INT8 absmax quantization depend only on one token row at a
time, so preparing local shards before ``all_gather_into_tensor_coalesced`` is bit-exact,
halves activation collective byte traffic, reduces preparation compute by ``world_size``,
and drops the full-sequence floating-point gathered buffer.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import cast

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.node import Argument

from piper_kernels._distributed import _fx as distributed_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _compile_fx

_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT_TENSOR = torch.ops._c10d_functional.wait_tensor.default
_aten = torch.ops.aten


def _emit_coalesced_wait(
    graph: torch.fx.Graph,
    local_qdata: torch.fx.Node,
    local_scale: torch.fx.Node,
    group_size: int,
    group_name: str,
) -> tuple[torch.fx.Node, torch.fx.Node]:
    local_qdata_val = cast(torch.Tensor, local_qdata.meta["val"])
    local_scale_val = cast(torch.Tensor, local_scale.meta["val"])
    gathered_qdata_val = local_qdata_val.new_empty(
        (group_size * local_qdata_val.shape[0], *local_qdata_val.shape[1:])
    )
    gathered_scale_val = local_scale_val.new_empty(
        (group_size * local_scale_val.shape[0], *local_scale_val.shape[1:])
    )

    all_gather_node = graph.call_function(
        _ALL_GATHER_COALESCED,
        args=([local_qdata, local_scale], group_size, group_name),
    )
    all_gather_node.meta["val"] = [gathered_qdata_val, gathered_scale_val]

    qdata_item = graph.call_function(operator.getitem, args=(all_gather_node, 0))
    qdata_item.meta["val"] = gathered_qdata_val
    scale_item = graph.call_function(operator.getitem, args=(all_gather_node, 1))
    scale_item.meta["val"] = gathered_scale_val

    gathered_qdata = graph.call_function(_WAIT_TENSOR, args=(qdata_item,))
    gathered_qdata.meta["val"] = gathered_qdata_val
    gathered_scale = graph.call_function(_WAIT_TENSOR, args=(scale_item,))
    gathered_scale.meta["val"] = gathered_scale_val
    return gathered_qdata, gathered_scale


def _reconstruct_full_sequence(
    graph: torch.fx.Graph,
    gathered_qdata: torch.fx.Node,
    gathered_scale: torch.fx.Node,
    all_gather_match: distributed_fx.SequenceAllGatherMatch,
) -> tuple[torch.fx.Node, torch.fx.Node]:
    output_val = preparation_sharing.tensor_metadata(all_gather_match.output_node)
    assert output_val is not None
    batch, full_sequence, hidden = output_val.shape
    local_val = preparation_sharing.tensor_metadata(all_gather_match.input_node)
    assert local_val is not None
    full_qdata_val = local_val.new_empty((batch, full_sequence, hidden), dtype=torch.int8)
    full_scale_val = local_val.new_empty((batch, full_sequence), dtype=torch.float32)

    if all_gather_match.split_node is None and statically_known_true(batch == 1):
        assert all_gather_match.full_sequence_arg is not None
        shape_arg = all_gather_match.output_node.args[1]
        assert isinstance(shape_arg, (list, tuple))
        assert len(shape_arg) == 3
        hidden_arg = cast(Argument, cast(Sequence[object], shape_arg)[2])
        full_qdata = graph.call_function(
            _aten.reshape.default,
            args=(gathered_qdata, [1, all_gather_match.full_sequence_arg, hidden_arg]),
        )
        full_scale = graph.call_function(
            _aten.reshape.default,
            args=(gathered_scale, [1, all_gather_match.full_sequence_arg]),
        )
    else:
        assert all_gather_match.batch_split_arg is not None
        batch_arg = all_gather_match.batch_split_arg
        gathered_qdata_val = cast(torch.Tensor, gathered_qdata.meta["val"])
        gathered_scale_val = cast(torch.Tensor, gathered_scale.meta["val"])
        qdata_chunk_val = gathered_qdata_val.new_empty((batch, gathered_qdata_val.shape[1], hidden))
        scale_chunk_val = gathered_scale_val.new_empty((batch, gathered_scale_val.shape[1]))
        qdata_split = graph.call_function(_aten.split.Tensor, args=(gathered_qdata, batch_arg))
        qdata_split.meta["val"] = [qdata_chunk_val] * all_gather_match.group_size
        scale_split = graph.call_function(_aten.split.Tensor, args=(gathered_scale, batch_arg))
        scale_split.meta["val"] = [scale_chunk_val] * all_gather_match.group_size
        qdata_chunks: list[torch.fx.Node] = []
        scale_chunks: list[torch.fx.Node] = []
        for rank_index in range(all_gather_match.group_size):
            qdata_chunk = graph.call_function(operator.getitem, args=(qdata_split, rank_index))
            qdata_chunk.meta["val"] = qdata_chunk_val
            scale_chunk = graph.call_function(operator.getitem, args=(scale_split, rank_index))
            scale_chunk.meta["val"] = scale_chunk_val
            qdata_chunks.append(qdata_chunk)
            scale_chunks.append(scale_chunk)
        full_qdata = graph.call_function(_aten.cat.default, args=(qdata_chunks, 1))
        full_scale = graph.call_function(_aten.cat.default, args=(scale_chunks, 1))

    full_qdata.meta["val"] = full_qdata_val
    full_scale.meta["val"] = full_scale_val
    return full_qdata, full_scale


def _fold_one_quantized_all_gather(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    parsed = distributed_fx.parse_convrot_int8_prepare_input(node)
    if parsed is None:
        return False
    source, group_size, input_scale = parsed

    all_gather_match = distributed_fx.match_sequence_all_gather(source)
    if all_gather_match is None:
        return False
    sym_size_users = distributed_fx.collect_sym_size_users(all_gather_match.output_node, node)
    if sym_size_users is None:
        return False
    required_defs = [
        all_gather_match.input_node,
        *([input_scale] if input_scale is not None else []),
    ]
    insert_point = distributed_fx.find_insertion_point(
        graph,
        required_defs,
        [node, *sym_size_users],
    )
    outputs = distributed_fx.collect_indexed_outputs(node, 2)
    if insert_point is None or outputs is None:
        return False
    qdata_users, scale_users = outputs

    local_input_val = preparation_sharing.tensor_metadata(all_gather_match.input_node)
    assert local_input_val is not None
    with graph.inserting_before(insert_point):
        local_qdata, local_scale, _ = _compile_fx.emit_prepared_input(
            graph,
            all_gather_match.input_node,
            group_size,
            None,
            tuple(local_input_val.shape),
            input_scale,
        )
        gathered_qdata, gathered_scale = _emit_coalesced_wait(
            graph,
            local_qdata,
            local_scale,
            all_gather_match.group_size,
            all_gather_match.group_name,
        )
        full_qdata, full_scale = _reconstruct_full_sequence(
            graph, gathered_qdata, gathered_scale, all_gather_match
        )

    for sym_user in sym_size_users:
        sym_user.args = (full_qdata, *sym_user.args[1:])
    for qdata_user in qdata_users:
        qdata_user.replace_all_uses_with(full_qdata)
        graph.erase_node(qdata_user)
    for scale_user in scale_users:
        scale_user.replace_all_uses_with(full_scale)
        graph.erase_node(scale_user)
    if not node.users:
        graph.erase_node(node)
    distributed_fx.erase_sequence_all_gather(graph, all_gather_match)
    return True


def fold_quantized_all_gather(graph: torch.fx.Graph) -> bool:
    """Hoist ``convrot_int8_prepare_input`` ahead of sequence all-gathers when possible."""
    if not distributed_fx.has_c10d_collectives(graph):
        return False

    changed = False
    for node in list(graph.nodes):
        if _fold_one_quantized_all_gather(graph, node):
            changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__ = ["fold_quantized_all_gather"]
