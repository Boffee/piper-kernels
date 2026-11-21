"""Compiler folding for sparse Piper attention followed by a ConvRot INT8 projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels.attention.sparse_piper_attention._contracts import SUPPORTED_DTYPES
from piper_kernels.fusions.sparse_piper import _compile as sparse_piper_compile
from piper_kernels.linear import _bias
from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _compile_fx

from . import _backend, _layout, output

type _PreparedGateProjectionNodes = tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node | None,
]

type _PreparedQueryProjectionNodes = tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node | None,
    torch.fx.Node,
    torch.fx.Node,
    float,
    float,
]

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
)


@dataclass(frozen=True, slots=True)
class _MatchedAttentionOutput:
    """Validated anchor match for quantized sparse attention -> reshape -> ConvRot INT8 linear."""

    linear_node: torch.fx.Node
    reshape_node: torch.fx.Node
    attention_node: torch.fx.Node
    attention_arguments: tuple[Argument, ...]
    weight_qdata: torch.fx.Node
    weight_scale: torch.fx.Node
    bias: torch.fx.Node | None
    group_size: int
    input_scale: torch.fx.Node | None
    bounded_arguments: tuple[Argument, ...]
    force_kwargs: dict[str, Argument]


def _prepared_gate_projection(
    gate: object,
) -> _PreparedGateProjectionNodes | None:
    """Extract a reshaped prepared ConvRot INT8 gate for bounded projection."""
    if not isinstance(gate, torch.fx.Node):
        return None
    linear = sparse_piper_compile.unwrap_shape_only_views(gate)
    if linear is None or linear is gate:
        return None
    bound = linear_fx.bind_call(
        linear, torch.ops.piper_kernels.convrot_int8_linear_prepared.default
    )
    if bound is None or bound["logical_dtype"] not in SUPPORTED_DTYPES:
        return None
    input_qdata = bound["input_qdata"]
    input_scale = bound["input_scale"]
    weight_qdata = bound["weight_qdata"]
    weight_scale = bound["weight_scale"]
    bias = bound["bias"]
    if (
        not isinstance(input_qdata, torch.fx.Node)
        or not isinstance(input_scale, torch.fx.Node)
        or not isinstance(weight_qdata, torch.fx.Node)
        or not isinstance(weight_scale, torch.fx.Node)
        or (bias is not None and not isinstance(bias, torch.fx.Node))
    ):
        return None
    return input_qdata, input_scale, weight_qdata, weight_scale, bias


def _prepared_query_projection(
    query: object,
    query_scale: object,
    query_summary: object,
    routing_mode: object,
    block_lengths: Argument | None,
) -> tuple[_PreparedQueryProjectionNodes, torch.fx.Node | None] | None:
    """Recover one shared ConvRot INT8 Q producer from its three tuple outputs."""
    producer = sparse_piper_compile.ordered_tuple_output_producer(
        (query, query_scale, query_summary),
        torch.ops.piper_kernels.convrot_int8_sparse_piper_project_query.default,
    )
    query_value = (
        preparation_sharing.tensor_metadata(query) if isinstance(query, torch.fx.Node) else None
    )
    if producer is None or query_value is None or query_value.ndim != 4:
        return None
    bound = linear_fx.bind_arguments(producer)
    if bound is None:
        return None
    head_dim = bound.get("head_dim")
    if (
        (head_dim is not None and head_dim != query_value.shape[-1])
        or bound["routing_mode"] != routing_mode
        or bound["block_lengths"] is not block_lengths
    ):
        return None
    projection = tuple(bound[name] for name in _QUERY_PROJECTION_ARG_NAMES)
    bias = bound["bias"]
    if bias is not None and not isinstance(bias, torch.fx.Node):
        return None
    if (
        any(not isinstance(value, torch.fx.Node) for value in (*projection[:4], *projection[5:7]))
        or (projection[4] is not None and not isinstance(projection[4], torch.fx.Node))
        or not isinstance(projection[7], float)
        or not isinstance(projection[8], float)
    ):
        return None
    return cast(_PreparedQueryProjectionNodes, projection), bias


def _can_reuse_source_as_output(source: torch.fx.Node, original: torch.fx.Node) -> bool:
    """Prove that preparing this fresh input exhausts every use of its storage."""
    if not preparation_sharing.operator_returns_fresh_tensor(source):
        return False
    source_value = preparation_sharing.tensor_metadata(source)
    output_value = preparation_sharing.tensor_metadata(original)
    if source_value is None or output_value is None:
        return False
    if (
        source_value.dtype is not output_value.dtype
        or source_value.device != output_value.device
        or not source_value.is_contiguous()
        or tuple(map(preparation_sharing.dimension_key, source_value.shape))
        != tuple(map(preparation_sharing.dimension_key, output_value.shape))
    ):
        return False
    before_output = set()
    for node in source.graph.nodes:
        if node is original:
            break
        before_output.add(node)
    return all(
        user.op == "call_function"
        and (
            user.target is torch.ops.aten.sym_size.int
            or (
                user in before_output
                and user.target is torch.ops.piper_kernels.convrot_int8_prepare_input.default
            )
        )
        for user in source.users
    )


def _input_preparation(
    qdata: torch.fx.Node,
    scales: torch.fx.Node,
) -> tuple[torch.fx.Node, int, torch.fx.Node | None] | None:
    """Recover an ordinary input and its scale from a prepared tuple."""
    producer = sparse_piper_compile.ordered_tuple_output_producer(
        (qdata, scales), torch.ops.piper_kernels.convrot_int8_prepare_input.default
    )
    if producer is None:
        return None
    bound = linear_fx.bind_arguments(producer)
    if bound is None:
        return None
    source = bound["input"]
    group_size = bound["group_size"]
    activation = bound["activation_fn"]
    scale = bound["input_scale"]
    if (
        not isinstance(source, torch.fx.Node)
        or sparse_piper_compile.static_int(group_size) is None
        or activation is not None
        or (scale is not None and not isinstance(scale, torch.fx.Node))
    ):
        return None
    return source, cast(int, group_size), scale


def _valid_attention_output(  # noqa: PLR0911
    linear_node: torch.fx.Node,
    reshaped_match: sparse_piper_compile.ReshapedQuantizedAttentionMatch,
    weight_qdata_node: torch.fx.Node,
    weight_scale_node: torch.fx.Node,
    bias_node: torch.fx.Node | None,
    group_size: int,
    input_scale_node: torch.fx.Node | None,
) -> bool:
    query_arg = reshaped_match.attention_args["query"]
    if not isinstance(query_arg, torch.fx.Node):
        return False
    query = preparation_sharing.tensor_metadata(query_arg)
    weight = preparation_sharing.tensor_metadata(weight_qdata_node)
    scale = preparation_sharing.tensor_metadata(weight_scale_node)
    attention = preparation_sharing.tensor_metadata(reshaped_match.attention_node)
    reshaped = preparation_sharing.tensor_metadata(reshaped_match.reshape_node)
    projected = preparation_sharing.tensor_metadata(linear_node)
    if (
        query is None
        or weight is None
        or scale is None
        or attention is None
        or reshaped is None
        or projected is None
    ):
        return False
    if (
        query.ndim != 4
        or attention.ndim != 4
        or reshaped.ndim != 3
        or projected.ndim != 3
        or weight.ndim != 2
        or query.dtype is not torch.int8
        or attention.dtype not in SUPPORTED_DTYPES
        or reshaped_match.attention_args.get("output_dtype", torch.bfloat16) is not attention.dtype
        or reshaped.dtype is not attention.dtype
        or projected.dtype is not attention.dtype
        or weight.dtype is not torch.int8
        or scale.dtype is not torch.float32
    ):
        return False
    if _backend.select_output_backend(query) is None:
        return False

    batch, sequence_length = attention.shape[:2]
    heads = sparse_piper_compile.static_int(attention.shape[2])
    head_dim = sparse_piper_compile.static_int(attention.shape[3])
    input_features = sparse_piper_compile.static_int(weight.shape[1])
    output_features = sparse_piper_compile.static_int(weight.shape[0])
    if (
        input_features is None
        or output_features is None
        or heads is None
        or head_dim not in _layout.SUPPORTED_HEAD_DIMS
        or input_features != heads * head_dim
        or output_features < 1
        or not sparse_piper_compile.same_dimension(reshaped.shape[0], batch)
        or not sparse_piper_compile.same_dimension(reshaped.shape[1], sequence_length)
        or reshaped.shape[2] != input_features
        or not sparse_piper_compile.same_dimension(projected.shape[0], batch)
        or not sparse_piper_compile.same_dimension(projected.shape[1], sequence_length)
        or projected.shape[2] != output_features
        or tuple(scale.shape) != (output_features, 1)
        or not sparse_piper_compile.same_dimension(query.shape[0], batch)
        or query.shape[1] != heads
        or query.shape[3] != head_dim
    ):
        return False
    devices = query.device, attention.device, reshaped.device, projected.device
    if any(device != query.device for device in devices[1:]) or any(
        value.device != query.device for value in (weight, scale)
    ):
        return False
    if any(
        value.layout is not torch.strided or not value.is_contiguous()
        for value in (query, attention, reshaped, projected, weight, scale)
    ):
        return False

    shape = reshaped_match.attention_shape
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        return False
    if group_size < 1 or input_features % group_size:
        return False
    if not _compile_fx.valid_input_scale(input_scale_node, query.device):
        return False
    if bias_node is None:
        return True
    bias = preparation_sharing.tensor_metadata(bias_node)
    return bool(
        bias is not None
        and bias.shape == (output_features,)
        and _bias.is_supported_dtype(bias.dtype)
        and bias.device == query.device
        and bias.layout is torch.strided
        and bias.is_contiguous()
    )


def _match_attention_output(linear_node: torch.fx.Node) -> _MatchedAttentionOutput | None:
    """Anchor-match one quantized sparse attention -> reshape -> ConvRot INT8 linear."""
    linear_args = linear_fx.bind_call(
        linear_node, torch.ops.piper_kernels.convrot_int8_linear.default
    )
    if linear_args is None or linear_args["activation_fn"] is not None:
        return None
    reshaped_match = sparse_piper_compile.match_reshaped_quantized_attention(linear_args["input"])
    if reshaped_match is None:
        return None
    weight_qdata = linear_args["weight_qdata"]
    weight_scale = linear_args["weight_scale"]
    bias = linear_args["bias"]
    group_size = sparse_piper_compile.static_int(linear_args["group_size"])
    input_scale = linear_args["input_scale"]
    if (
        not isinstance(weight_qdata, torch.fx.Node)
        or not isinstance(weight_scale, torch.fx.Node)
        or (bias is not None and not isinstance(bias, torch.fx.Node))
        or group_size is None
        or (input_scale is not None and not isinstance(input_scale, torch.fx.Node))
    ):
        return None
    if not _valid_attention_output(
        linear_node,
        reshaped_match,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
    ):
        return None
    return _MatchedAttentionOutput(
        linear_node=linear_node,
        reshape_node=reshaped_match.reshape_node,
        attention_node=reshaped_match.attention_node,
        attention_arguments=reshaped_match.attention_arguments,
        weight_qdata=weight_qdata,
        weight_scale=weight_scale,
        bias=bias,
        group_size=group_size,
        input_scale=input_scale,
        bounded_arguments=reshaped_match.bounded_arguments,
        force_kwargs=reshaped_match.force_kwargs,
    )


def _replace_attention_output(graph: torch.fx.Graph, matched: _MatchedAttentionOutput) -> None:
    original = matched.linear_node
    bounded_arguments = matched.bounded_arguments
    block_lengths, coarse_gate, coarse_scale, coarse_key_blocks, query_blocks = bounded_arguments
    gate_projection = _prepared_gate_projection(coarse_gate)
    gate_arguments: tuple[Argument, ...] = ()
    if gate_projection is not None:
        bounded_arguments = (
            block_lengths,
            None,
            coarse_scale,
            coarse_key_blocks,
            query_blocks,
        )
        gate_arguments = gate_projection

    attention_arguments = matched.attention_arguments
    query_projection = _prepared_query_projection(
        attention_arguments[0],
        attention_arguments[1],
        attention_arguments[2],
        attention_arguments[-1],
        block_lengths,
    )
    output_kwargs: dict[str, Argument] = {
        "output_dtype": original.meta["val"].dtype,
        "output_input_scale": matched.input_scale,
        **matched.force_kwargs,
    }
    reusable_input = None
    with graph.inserting_before(original):
        attention_tail = (
            *attention_arguments[3:],
            matched.weight_qdata,
            matched.weight_scale,
            matched.bias,
            matched.group_size,
        )
        if query_projection is None:
            target = torch.ops.piper_kernels.convrot_int8_sparse_piper_attention_output.default
            common_arguments = (
                *attention_arguments[:3],
                *attention_tail,
            )
        else:
            projected_query_output = (
                torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output
            )
            target = projected_query_output.default
            prepared_arguments, query_bias = query_projection
            projection_arguments: tuple[Argument, ...] = prepared_arguments
            if gate_projection is not None:
                query_source = _input_preparation(*prepared_arguments[:2])
                gate_source = _input_preparation(*gate_projection[:2])
                if (
                    query_source is not None
                    and gate_source is not None
                    and query_source[:2] == gate_source[:2]
                    and query_source[2] is not gate_source[2]
                ):
                    # Prepare independent Q/gate scales inside the bounded chunk loop.
                    # Keeping both full-sequence INT8 inputs defeats that memory bound.
                    source, input_group_size, query_scale = query_source
                    projection_arguments = (
                        source,
                        query_scale,
                        *projection_arguments[2:],
                    )
                    # Chunk-prepared Q and gate always share ``source``. Pass it
                    # once so the custom-op mutation contract has one owner.
                    gate_arguments = (None, gate_source[2], *gate_projection[2:])
                    output_kwargs["input_group_size"] = input_group_size
                    if _can_reuse_source_as_output(source, original):
                        reusable_input = source
                        ops = torch.ops.piper_kernels
                        reuse = ops.convrot_int8_sparse_piper_projected_query_attention_output_
                        target = reuse.default
            common_arguments = (*projection_arguments, *attention_tail)
            output_kwargs["query_bias"] = query_bias
        replacement = graph.call_function(
            target,
            args=(
                *common_arguments,
                output._DEFAULT_QUERY_CHUNK_ROWS,
                *bounded_arguments,
                *gate_arguments,
            ),
            kwargs=output_kwargs,
        )
    if reusable_input is None:
        replacement.meta = original.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        original.replace_all_uses_with(replacement)
    else:
        replacement.meta["val"] = None
        original.replace_all_uses_with(reusable_input)
    graph.erase_node(original)
    graph.erase_node(matched.reshape_node)
    graph.erase_node(matched.attention_node)


def _fold_attention_output(graph: torch.fx.Graph) -> bool:
    """Replace compatible materialized attention-to-output regions."""
    changed = False
    for node in list(graph.nodes):
        matched = _match_attention_output(node)
        if matched is None:
            continue
        _replace_attention_output(graph, matched)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__: list[str] = []
