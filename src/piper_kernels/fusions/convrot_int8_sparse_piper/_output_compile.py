"""Compiler folding for sparse Piper attention followed by a ConvRot INT8 projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels.attention.sparse_piper_attention._contracts import SUPPORTED_DTYPES
from piper_kernels.fusions.attention import _sparse_compile as sparse_piper_compile
from piper_kernels.linear import _bias
from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _compile_fx

from . import _backend, output

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
    reshaped: sparse_piper_compile.ReshapedQuantizedAttentionMatch
    weight_qdata: torch.fx.Node
    weight_scale: torch.fx.Node
    bias: torch.fx.Node | None
    group_size: int
    input_scale: torch.fx.Node | None


def _prepared_gate_projection(
    gate: object,
) -> _PreparedGateProjectionNodes | None:
    """Extract a reshaped prepared ConvRot INT8 gate for bounded projection."""
    linear = sparse_piper_compile.unwrap_reshaped_producer(gate)
    if linear is None:
        return None
    bound = linear_fx.bind_call(
        linear, torch.ops.piper_kernels.convrot_int8_linear_prepared.default
    )
    if bound is None or bound["logical_dtype"] not in SUPPORTED_DTYPES:
        return None
    nodes = tuple(
        bound[name] for name in ("input_qdata", "input_scale", "weight_qdata", "weight_scale")
    )
    bias = bound["bias"]
    if any(not isinstance(node, torch.fx.Node) for node in nodes) or (
        bias is not None and not isinstance(bias, torch.fx.Node)
    ):
        return None
    return cast(_PreparedGateProjectionNodes, (*nodes, bias))


def _prepared_query_projection(
    query: object,
    query_scale: object,
    query_summary: object,
    routing_mode: object,
    block_lengths: Argument | None,
) -> tuple[_PreparedQueryProjectionNodes, torch.fx.Node | None] | None:
    """Recover one shared ConvRot INT8 Q producer from its three tuple outputs."""
    bound = sparse_piper_compile.match_projected_query_producer(
        query,
        query_scale,
        query_summary,
        torch.ops.piper_kernels.convrot_int8_sparse_piper_project_query.default,
        routing_mode=routing_mode,
        block_lengths=block_lengths,
    )
    if bound is None:
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


def _valid_attention_output(
    linear_node: torch.fx.Node,
    reshaped_match: sparse_piper_compile.ReshapedQuantizedAttentionMatch,
    weight_qdata_node: torch.fx.Node,
    weight_scale_node: torch.fx.Node,
    bias_node: torch.fx.Node | None,
    group_size: int,
    input_scale_node: torch.fx.Node | None,
) -> bool:
    common = sparse_piper_compile.validate_reshaped_attention_output(linear_node, reshaped_match)
    if common is None:
        return False
    weight = preparation_sharing.tensor_metadata(weight_qdata_node)
    scale = preparation_sharing.tensor_metadata(weight_scale_node)
    if weight is None or scale is None or weight.ndim != 2:
        return False
    input_features = sparse_piper_compile.static_int(weight.shape[1])
    output_features = sparse_piper_compile.static_int(weight.shape[0])
    device = common.query.device
    if (
        weight.dtype is not torch.int8
        or scale.dtype is not torch.float32
        or input_features != common.input_features
        or output_features is None
        or output_features < 1
        or common.projected.shape[2] != output_features
        or tuple(scale.shape) != (output_features, 1)
        or group_size < 1
        or common.input_features % group_size
        or not _compile_fx.valid_input_scale(input_scale_node, device)
        or any(
            value.device != device or value.layout is not torch.strided or not value.is_contiguous()
            for value in (weight, scale)
        )
    ):
        return False
    if bias_node is not None:
        bias = preparation_sharing.tensor_metadata(bias_node)
        if (
            bias is None
            or bias.shape != (output_features,)
            or not _bias.is_supported_dtype(bias.dtype)
            or bias.device != device
            or bias.layout is not torch.strided
            or not bias.is_contiguous()
        ):
            return False
    return _backend.select_output_backend(common.query) is not None


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
        reshaped=reshaped_match,
        weight_qdata=weight_qdata,
        weight_scale=weight_scale,
        bias=bias,
        group_size=group_size,
        input_scale=input_scale,
    )


def _replace_attention_output(graph: torch.fx.Graph, matched: _MatchedAttentionOutput) -> None:
    original = matched.linear_node
    bounded_arguments, gate_projection = matched.reshaped.defer_gate(_prepared_gate_projection)
    block_lengths = bounded_arguments[0]
    gate_arguments: tuple[Argument, ...] = () if gate_projection is None else gate_projection

    attention_arguments = matched.reshaped.attention_arguments
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
        **matched.reshaped.force_kwargs,
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
    sparse_piper_compile.replace_matched_nodes(
        graph,
        original,
        replacement,
        matched.reshaped.reshape_node,
        matched.reshaped.attention_node,
        in_place_output=reusable_input,
    )


def _fold_attention_output(graph: torch.fx.Graph) -> bool:
    """Replace compatible materialized attention-to-output regions."""
    return sparse_piper_compile.fold_matched_nodes(
        graph,
        _match_attention_output,
        _replace_attention_output,
    )


__all__: list[str] = []
