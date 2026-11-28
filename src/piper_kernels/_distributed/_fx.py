"""FX pattern matchers for standard DTensor sequence-parallel collectives.

Pure FX metadata inspection without Triton or collective runtime imports. Only matches
contiguous ``[batch, sequence, features]`` transfers over PyTorch's functional c10d ops.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels._triton import runtime

_c10d = torch.ops._c10d_functional
_aten = torch.ops.aten

_ALL_GATHER = _c10d.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = _c10d.all_gather_into_tensor_coalesced.default
_REDUCE_SCATTER = _c10d.reduce_scatter_tensor.default
_WAIT_TENSOR = _c10d.wait_tensor.default

_RESHAPE_TARGETS = (
    _aten.reshape.default,
    _aten.view.default,
    _aten._unsafe_view.default,
)


def has_c10d_collectives(graph: torch.fx.Graph) -> bool:
    """Return whether ``graph`` contains any functional c10d collective."""
    return any(
        node.op == "call_function"
        and node.target in (_ALL_GATHER, _ALL_GATHER_COALESCED, _REDUCE_SCATTER, _WAIT_TENSOR)
        for node in graph.nodes
    )


def _static_group_size(value: object) -> int | None:
    return value if type(value) is int and value >= 2 else None


def _is_valid_group_name(value: object) -> bool:
    return isinstance(value, str)


def _is_contiguous_3d_device_tensor(value: torch.Tensor | None) -> bool:
    return bool(
        value is not None
        and value.ndim == 3
        and value.layout is torch.strided
        and value.is_contiguous()
        and runtime.supports_device(value.device)
    )


def _bind_reshape(node: torch.fx.Node) -> tuple[torch.fx.Node, Argument] | None:
    if node.op != "call_function" or node.target not in _RESHAPE_TARGETS:
        return None
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    bound = linear_fx.bind_arguments(node)
    if bound is None:
        return None
    source = bound.get("self")
    shape_arg = bound["shape"] if "shape" in bound else bound.get("size")
    if not isinstance(source, torch.fx.Node):
        return None
    return source, shape_arg


@dataclass(frozen=True, slots=True)
class SequenceAllGatherMatch:
    """Matched DTensor ``Shard(1) -> Replicate()`` all-gather over ``[B, T_local, H]``."""

    input_node: torch.fx.Node
    group_size: int
    group_name: str
    all_gather_node: torch.fx.Node
    wait_node: torch.fx.Node
    output_node: torch.fx.Node
    # Non-None when matched via split(dim=0) + cat(dim=1) for B >= 1.
    split_node: torch.fx.Node | None
    batch_split_arg: torch.fx.Node | int | None
    # Sequence extent expression for [1, T_full, H] reshape reconstruction.
    full_sequence_arg: torch.fx.Node | int | None


@dataclass(frozen=True, slots=True)
class SequenceReduceScatterMatch:
    """Matched DTensor ``Partial('sum') -> Shard(1)`` reduce-scatter over ``[B, T_full, H]``."""

    input_node: torch.fx.Node
    group_size: int
    group_name: str
    split_node: torch.fx.Node | None
    cat_node: torch.fx.Node | None
    repack_node: torch.fx.Node
    reduce_scatter_node: torch.fx.Node
    wait_node: torch.fx.Node


def _match_split_getitem(item: object, expected_index: int) -> torch.fx.Node | None:
    if (
        not isinstance(item, torch.fx.Node)
        or item.op != "call_function"
        or item.target is not operator.getitem
        or item.kwargs
        or len(item.args) != 2
        or item.args[1] != expected_index
        or len(item.users) != 1
    ):
        return None
    producer = item.args[0]
    return producer if isinstance(producer, torch.fx.Node) else None


def _match_split_getitems(
    cat_items: Sequence[object],
    group_size: int,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node | int, int] | None:
    """Recover a single ``split(tensor, split_size, dim)`` feeding ``0..W-1`` getitems."""
    if len(cat_items) != group_size:
        return None
    split_node: torch.fx.Node | None = None
    seen_getitems: set[torch.fx.Node] = set()
    for expected_index, item in enumerate(cat_items):
        producer = _match_split_getitem(item, expected_index)
        if producer is None or (split_node is not None and producer is not split_node):
            return None
        split_node = producer
        seen_getitems.add(cast(torch.fx.Node, item))
    if split_node is None or set(split_node.users) != seen_getitems:
        return None
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    split_bound = linear_fx.bind_call(split_node, _aten.split.Tensor)
    if split_bound is None:
        return None
    source = split_bound["self"]
    split_size = split_bound["split_size"]
    split_dim = split_bound["dim"]
    if (
        not isinstance(source, torch.fx.Node)
        or not isinstance(split_size, (torch.fx.Node, int))
        or isinstance(split_size, bool)
        or type(split_dim) is not int
    ):
        return None
    return split_node, source, split_size, split_dim


def _match_all_gather_view(
    node: torch.fx.Node,
    output_value: torch.Tensor,
) -> (
    tuple[
        torch.fx.Node, torch.fx.Node | None, torch.fx.Node | int | None, torch.fx.Node | int | None
    ]
    | None
):
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    reshaped = _bind_reshape(node)
    if reshaped is not None:
        wait_node, shape_arg = reshaped
        sequence_expr = (
            shape_arg[1] if isinstance(shape_arg, (list, tuple)) and len(shape_arg) == 3 else None
        )
        if (
            linear_fx.same_dimension(output_value.shape[0], 1)
            and isinstance(sequence_expr, (torch.fx.Node, int))
            and not isinstance(sequence_expr, bool)
        ):
            return (wait_node, None, None, sequence_expr)
        return None
    cat_bound = linear_fx.bind_call(node, _aten.cat.default)
    cat_items = cat_bound["tensors"] if cat_bound is not None and cat_bound["dim"] == 1 else None
    if not isinstance(cat_items, (list, tuple)) or len(cat_items) < 2:
        return None
    matched_split = _match_split_getitems(cat_items, len(cat_items))
    if matched_split is None or matched_split[3] != 0:
        return None
    split_node, wait_node, batch_split_arg, _ = matched_split
    return (wait_node, split_node, batch_split_arg, None)


def _match_all_gather_wait(
    wait_node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, int, str] | None:
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    wait_bound = linear_fx.bind_call(wait_node, _WAIT_TENSOR)
    if wait_bound is None or len(wait_node.users) != 1:
        return None
    all_gather_node = wait_bound["tensor"]
    ag_bound = linear_fx.bind_call(all_gather_node, _ALL_GATHER)
    if (
        ag_bound is None
        or not isinstance(all_gather_node, torch.fx.Node)
        or len(all_gather_node.users) != 1
    ):
        return None
    input_node = ag_bound["input"]
    group_size = _static_group_size(ag_bound["group_size"])
    group_name = ag_bound["group_name"]
    if (
        not isinstance(input_node, torch.fx.Node)
        or group_size is None
        or not _is_valid_group_name(group_name)
    ):
        return None
    return all_gather_node, input_node, group_size, cast(str, group_name)


def match_sequence_all_gather(node: torch.fx.Node) -> SequenceAllGatherMatch | None:
    """Match DTensor ``Shard(1) -> Replicate()`` reconstructing ``[B, W * T_local, H]``."""
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    output_value = linear_fx.node_tensor_metadata(node)
    if node.op != "call_function" or not _is_contiguous_3d_device_tensor(output_value):
        return None
    assert output_value is not None

    view_match = _match_all_gather_view(node, output_value)
    if view_match is None:
        return None
    wait_node, split_node, batch_split_arg, full_sequence_arg = view_match

    wait_match = _match_all_gather_wait(wait_node)
    if wait_match is None:
        return None
    all_gather_node, input_node, group_size, group_name = wait_match

    input_value = linear_fx.node_tensor_metadata(input_node)
    wait_value = linear_fx.node_tensor_metadata(wait_node)
    if (
        not _is_contiguous_3d_device_tensor(input_value)
        or wait_value is None
        or input_value is None
        or input_value.device != output_value.device
        or input_value.dtype is not output_value.dtype
    ):
        return None

    batch, local_sequence, hidden = input_value.shape
    shapes_ok = linear_fx.same_shape(
        wait_value.shape, (batch * group_size, local_sequence, hidden)
    ) and linear_fx.same_shape(output_value.shape, (batch, local_sequence * group_size, hidden))
    if split_node is not None:
        batch_val = linear_fx.argument_metadata(batch_split_arg)
        shapes_ok = (
            shapes_ok
            and isinstance(batch_val, (int, torch.SymInt))
            and not isinstance(batch_val, bool)
            and linear_fx.same_dimension(batch_val, batch)
        )
    if not shapes_ok:
        return None

    return SequenceAllGatherMatch(
        input_node=input_node,
        group_size=group_size,
        group_name=group_name,
        all_gather_node=all_gather_node,
        wait_node=wait_node,
        output_node=node,
        split_node=split_node,
        batch_split_arg=batch_split_arg,
        full_sequence_arg=full_sequence_arg,
    )


def _match_reduce_scatter_repack(
    repack_node: torch.fx.Node,
    group_size: int,
    output_value: torch.Tensor,
) -> tuple[torch.fx.Node, torch.fx.Node | None, torch.fx.Node | None] | None:
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    cat_bound = linear_fx.bind_call(repack_node, _aten.cat.default)
    if (
        cat_bound is not None
        and cat_bound["dim"] == 0
        and isinstance(cat_bound["tensors"], (list, tuple))
    ):
        matched_split = _match_split_getitems(cat_bound["tensors"], group_size)
        if matched_split is None:
            return None
        split_node, source, _, split_dim = matched_split
        if split_dim != 1:
            return None
        return source, split_node, repack_node

    reshaped = _bind_reshape(repack_node)
    if reshaped is not None and linear_fx.same_dimension(output_value.shape[0], 1):
        return reshaped[0], None, None
    return None


def _match_reduce_scatter_wait(
    wait_node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, int, str, torch.Tensor] | None:
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    wait_bound = linear_fx.bind_call(wait_node, _WAIT_TENSOR)
    if wait_bound is None:
        return None
    reduce_scatter_node = wait_bound["tensor"]
    rs_bound = linear_fx.bind_call(reduce_scatter_node, _REDUCE_SCATTER)
    if (
        rs_bound is None
        or not isinstance(reduce_scatter_node, torch.fx.Node)
        or len(reduce_scatter_node.users) != 1
    ):
        return None
    repack_node = rs_bound["input"]
    reduce_op = rs_bound["reduce_op"]
    group_size = _static_group_size(rs_bound["group_size"])
    group_name = rs_bound["group_name"]
    output_value = linear_fx.node_tensor_metadata(wait_node)
    if (
        not isinstance(repack_node, torch.fx.Node)
        or reduce_op != "sum"
        or group_size is None
        or not _is_valid_group_name(group_name)
        or len(repack_node.users) != 1
        or not _is_contiguous_3d_device_tensor(output_value)
    ):
        return None
    assert output_value is not None
    return reduce_scatter_node, repack_node, group_size, cast(str, group_name), output_value


def match_sequence_reduce_scatter(wait_node: torch.fx.Node) -> SequenceReduceScatterMatch | None:
    """Match DTensor ``Partial('sum') -> Shard(1)`` reducing ``[B, W * T_local, H]``."""
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    wait_match = _match_reduce_scatter_wait(wait_node)
    if wait_match is None:
        return None
    reduce_scatter_node, repack_node, group_size, group_name, output_value = wait_match

    repack_match = _match_reduce_scatter_repack(repack_node, group_size, output_value)
    if repack_match is None:
        return None
    input_node, split_node, cat_node = repack_match

    input_value = linear_fx.node_tensor_metadata(input_node)
    repack_value = linear_fx.node_tensor_metadata(repack_node)
    if (
        not _is_contiguous_3d_device_tensor(input_value)
        or repack_value is None
        or input_value is None
        or input_value.device != output_value.device
        or input_value.dtype is not output_value.dtype
    ):
        return None

    batch, local_sequence, hidden = output_value.shape
    if not (
        linear_fx.same_shape(input_value.shape, (batch, local_sequence * group_size, hidden))
        and linear_fx.same_shape(repack_value.shape, (batch * group_size, local_sequence, hidden))
    ):
        return None

    return SequenceReduceScatterMatch(
        input_node=input_node,
        group_size=group_size,
        group_name=group_name,
        split_node=split_node,
        cat_node=cat_node,
        repack_node=repack_node,
        reduce_scatter_node=reduce_scatter_node,
        wait_node=wait_node,
    )


def match_distributed_residual_add(
    call: torch.fx.Node,
    residual_add_op: object,
) -> tuple[torch.fx.Node, SequenceReduceScatterMatch] | None:
    """Match ``auto_functionalized(residual_add_op, base=..., update=wait_node)``."""
    if (
        call.op != "call_function"
        or call.target is not torch.ops.higher_order.auto_functionalized
        or not call.args
        or call.args[0] is not residual_add_op
    ):
        return None
    update = call.kwargs.get("update")
    base = call.kwargs.get("base")
    if (
        not isinstance(update, torch.fx.Node)
        or not isinstance(base, torch.fx.Node)
        or set(update.users) != {call}
        or any(user.target is not operator.getitem or user.args[1] != 1 for user in call.users)
    ):
        return None
    reduce_scatter_match = match_sequence_reduce_scatter(update)
    if reduce_scatter_match is None:
        return None
    return base, reduce_scatter_match


def _erase_if_unused(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
    if not node.users:
        graph.erase_node(node)


def _erase_split_chain(graph: torch.fx.Graph, split_node: torch.fx.Node) -> None:
    for item in list(split_node.users):
        if item.target is operator.getitem:
            _erase_if_unused(graph, item)
    _erase_if_unused(graph, split_node)


def erase_sequence_all_gather(graph: torch.fx.Graph, match: SequenceAllGatherMatch) -> None:
    """Explicitly erase an unused sequence all-gather chain (``wait_tensor`` is impure)."""
    _erase_if_unused(graph, match.output_node)
    if match.split_node is not None:
        _erase_split_chain(graph, match.split_node)
    _erase_if_unused(graph, match.wait_node)
    _erase_if_unused(graph, match.all_gather_node)


def erase_sequence_reduce_scatter(graph: torch.fx.Graph, match: SequenceReduceScatterMatch) -> None:
    """Explicitly erase an unused sequence reduce-scatter chain (``wait_tensor`` is impure)."""
    _erase_if_unused(graph, match.wait_node)
    _erase_if_unused(graph, match.reduce_scatter_node)
    _erase_if_unused(graph, match.repack_node)
    if match.split_node is not None:
        _erase_split_chain(graph, match.split_node)


def collect_sym_size_users(
    gathered_node: torch.fx.Node,
    consumer_node: torch.fx.Node,
) -> list[torch.fx.Node] | None:
    """Verify ``gathered_node`` only feeds ``consumer_node`` and valid 3-D ``aten.sym_size.int``."""
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    sym_size_users: list[torch.fx.Node] = []
    saw_consumer = False
    for user in gathered_node.users:
        if user is consumer_node:
            saw_consumer = True
            continue
        sym_bound = linear_fx.bind_call(user, _aten.sym_size.int)
        if (
            sym_bound is not None
            and sym_bound["self"] is gathered_node
            and type(sym_bound["dim"]) is int
            and -3 <= sym_bound["dim"] < 3
        ):
            sym_size_users.append(user)
        else:
            return None
    return sym_size_users if saw_consumer else None


def collect_indexed_outputs(
    node: torch.fx.Node,
    count: int,
) -> list[list[torch.fx.Node]] | None:
    """Group ``operator.getitem(node, idx)`` users by index in ``[0, count)``."""
    buckets: list[list[torch.fx.Node]] = [[] for _ in range(count)]
    saw_any = False
    for user in node.users:
        if (
            user.op != "call_function"
            or user.target is not operator.getitem
            or user.kwargs
            or len(user.args) != 2
            or user.args[0] is not node
        ):
            return None
        index = user.args[1]
        if type(index) is not int or not 0 <= index < count:
            return None
        buckets[index].append(user)
        saw_any = True
    return buckets if saw_any else None


def rewrite_sym_size_users(
    graph: torch.fx.Graph,
    sym_size_users: Sequence[torch.fx.Node],
    all_gather_match: SequenceAllGatherMatch,
) -> None:
    """Rewrite ``aten.sym_size.int(all_gather_output, dim)`` users to read ``input_node``."""
    if not sym_size_users:
        return
    from piper_kernels.linear import _compile_fx as linear_fx  # noqa: PLC0415

    positions = {node: idx for idx, node in enumerate(graph.nodes)}
    full_sequence_arg = all_gather_match.full_sequence_arg
    for sym_user in sym_size_users:
        sym_bound = linear_fx.bind_call(sym_user, _aten.sym_size.int)
        assert sym_bound is not None
        raw_dim = cast(int, sym_bound["dim"])
        dim = raw_dim % 3
        if dim == 1:
            if (
                isinstance(full_sequence_arg, torch.fx.Node)
                and positions.get(full_sequence_arg, len(positions)) < positions[sym_user]
            ):
                replacement: torch.fx.Node = full_sequence_arg
            else:
                with graph.inserting_before(sym_user):
                    local_sequence = graph.call_function(
                        _aten.sym_size.int,
                        args=(all_gather_match.input_node, 1),
                    )
                    sym_val = sym_user.meta.get("val")
                    if isinstance(sym_val, (int, torch.SymInt)) and not isinstance(sym_val, bool):
                        local_sequence.meta["val"] = sym_val // all_gather_match.group_size
                    replacement = graph.call_function(
                        operator.mul,
                        args=(local_sequence, all_gather_match.group_size),
                    )
                    if sym_val is not None:
                        replacement.meta["val"] = sym_val
            sym_user.replace_all_uses_with(replacement)
            graph.erase_node(sym_user)
        else:
            sym_user.args = (all_gather_match.input_node, raw_dim)
            sym_user.kwargs = {}


def find_insertion_point(
    graph: torch.fx.Graph,
    required_defs: Sequence[torch.fx.Node],
    consumers: Sequence[torch.fx.Node],
) -> torch.fx.Node | None:
    """Return the earliest consumer if every required definition strictly precedes it."""
    if not consumers:
        return None
    positions = {node: idx for idx, node in enumerate(graph.nodes)}
    if any(node not in positions for node in consumers):
        return None
    earliest_node = min(consumers, key=lambda node: positions[node])
    earliest_pos = positions[earliest_node]
    if any(positions.get(node, -1) >= earliest_pos for node in required_defs):
        return None
    return earliest_node


__all__ = [
    "SequenceAllGatherMatch",
    "SequenceReduceScatterMatch",
    "collect_indexed_outputs",
    "collect_sym_size_users",
    "erase_sequence_all_gather",
    "erase_sequence_reduce_scatter",
    "find_insertion_point",
    "has_c10d_collectives",
    "match_distributed_residual_add",
    "match_sequence_all_gather",
    "match_sequence_reduce_scatter",
    "rewrite_sym_size_users",
]
