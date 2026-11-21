"""Compiler folding for sparse Piper attention followed by NVFP4 or ConvRot NVFP4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MINMAX_ROUTING,
    SUPPORTED_DTYPES,
)
from piper_kernels.fusions.sparse_piper import _compile as sparse_piper_compile
from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _compile_fx
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops  # noqa: F401 - registers nvfp4_linear
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation

from . import (
    _output,
    output,  # noqa: F401 - register output operators
)

type _PreparedGateProjectionNodes = tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node | None,
    torch.fx.Node | None,
]
type _PreparedQueryProjectionNodes = tuple[Argument, ...]

_NVFP4_LINEAR = "piper_kernels::nvfp4_linear"
_CONVROT_NVFP4_LINEAR = "piper_kernels::convrot_nvfp4_linear"
_QUERY_PROJECTION_ARGUMENT_NAMES = (
    "input_qdata",
    "input_scale",
    "input_per_tensor_scale",
    "weight_qdata",
    "weight_scale",
    "weight_per_tensor_scale",
    "bias",
    "norm_weight",
    "cos",
    "sin",
    "norm_epsilon",
    "softmax_scale",
)


@dataclass(frozen=True, slots=True)
class _MatchedAttentionOutput:
    """Validated anchor match for attention -> reshape -> (ConvRot) NVFP4 linear."""

    linear_node: torch.fx.Node
    reshape_node: torch.fx.Node
    attention_node: torch.fx.Node
    attention_arguments: tuple[Argument, ...]
    projection_arguments: tuple[Argument, ...]
    bounded_arguments: tuple[Argument, ...]
    force_kwargs: dict[str, Argument]
    high_first: bool
    dynamic_activation_scale: bool
    group_size: int | None
    output_dtype: torch.dtype


def _shape_dimensions(value: object) -> tuple[int | torch.SymInt, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    dimensions = tuple(
        sparse_piper_compile.integer_scalar_metadata(dimension) for dimension in value
    )
    if any(dimension is None for dimension in dimensions):
        return None
    return dimensions  # type: ignore[return-value]


def _prepared_gate_projection(
    gate: object,
) -> _PreparedGateProjectionNodes | None:
    """Extract a reshaped prepared NVFP4 gate for bounded projection."""
    if not isinstance(gate, torch.fx.Node):
        return None
    linear = sparse_piper_compile.unwrap_shape_only_views(gate)
    if (
        linear is None
        or linear is gate
        or linear.target is not torch.ops.piper_kernels.nvfp4_linear_prepared.default
    ):
        return None
    operands = _compile_fx.PreparedLinearNodes.from_call(linear)
    if operands is None or operands.logical_dtype not in SUPPORTED_DTYPES:
        return None
    shape = _compile_fx.validated_prepared_linear(
        operands,
        "fused sparse Piper NVFP4 compiler gate",
    )
    gate_value = preparation_sharing.tensor_metadata(gate)
    if (
        shape is None
        or gate_value is None
        or gate_value.ndim != 4
        or gate_value.dtype not in SUPPORTED_DTYPES
        or gate_value.shape[0] != 1
        or not sparse_piper_compile.same_dimension(gate_value.shape[1], shape.rows)
        or not sparse_piper_compile.same_dimension(
            gate_value.shape[2] * gate_value.shape[3],
            shape.output_features,
        )
        or gate_value.layout is not torch.strided
        or not gate_value.is_contiguous()
    ):
        return None
    return (
        operands.input_qdata,
        operands.input_scale,
        operands.input_per_tensor_scale,
        operands.weight_qdata,
        operands.weight_scale,
        operands.weight_per_tensor_scale,
        operands.bias,
    )


def _prepared_query_projection(
    query: object,
    query_scale: object,
    query_summary: object,
    routing_mode: object,
    block_lengths: object,
) -> _PreparedQueryProjectionNodes | None:
    """Recover one shared NVFP4 Q producer from its tuple outputs."""
    producer = sparse_piper_compile.ordered_tuple_output_producer(
        (query, query_scale, query_summary),
        torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default,
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
        or bound.get("routing_mode", _MINMAX_ROUTING) != routing_mode
        or bound.get("block_lengths") is not block_lengths
    ):
        return None
    return tuple(bound[name] for name in _QUERY_PROJECTION_ARGUMENT_NAMES)


def _valid_attention_output_metadata(  # noqa: PLR0911
    linear_node: torch.fx.Node,
    attention_node: torch.fx.Node,
    attention_shape_arg: Argument,
    linear_args: dict[str, Argument],
    attention_args: dict[str, Argument],
    *,
    dynamic_activation_scale: bool,
    group_size: int | None,
) -> torch.dtype | None:
    """Validate tensor metadata for one attention -> reshape -> (ConvRot) NVFP4 match."""
    query_node = attention_args["query"]
    weight_qdata_node = linear_args["weight_qdata"]
    weight_scale_node = linear_args["weight_scale"]
    if any(
        not isinstance(node, torch.fx.Node)
        for node in (query_node, weight_qdata_node, weight_scale_node)
    ):
        return None
    query = preparation_sharing.tensor_metadata(cast(torch.fx.Node, query_node))
    weight_qdata = preparation_sharing.tensor_metadata(cast(torch.fx.Node, weight_qdata_node))
    weight_scale = preparation_sharing.tensor_metadata(cast(torch.fx.Node, weight_scale_node))
    projected = preparation_sharing.tensor_metadata(linear_node)
    attention = preparation_sharing.tensor_metadata(attention_node)
    if (
        query is None
        or weight_qdata is None
        or weight_scale is None
        or projected is None
        or attention is None
    ):
        return None
    if (
        query.ndim != 4
        or query.dtype is not torch.int8
        or projected.ndim != 3
        or projected.dtype not in SUPPORTED_DTYPES
        or query.device.type != "cuda"
        or not AcceleratorTarget.from_device(query.device).is_cuda_capability(12, 0)
    ):
        return None

    batch = query.shape[0]
    heads = sparse_piper_compile.static_int(query.shape[1])
    head_dim = sparse_piper_compile.static_int(query.shape[3])
    attention_shape = _shape_dimensions(attention_shape_arg)
    if (
        heads is None
        or head_dim not in SUPPORTED_HEAD_DIMS
        or attention.ndim != 4
        or attention.dtype is not projected.dtype
        or attention_args.get("output_dtype", torch.bfloat16) is not attention.dtype
        or attention.device != query.device
        or attention.layout is not torch.strided
        or not attention.is_contiguous()
        or attention_shape is None
        or len(attention_shape) != 3
    ):
        return None

    input_features = heads * head_dim
    if (
        not sparse_piper_compile.same_dimension(attention.shape[0], batch)
        or attention.shape[2] != heads
        or attention.shape[3] != head_dim
        or not sparse_piper_compile.same_dimension(attention_shape[0], batch)
        or not sparse_piper_compile.same_dimension(attention_shape[1], attention.shape[1])
        or attention_shape[2] != input_features
        or not sparse_piper_compile.same_dimension(projected.shape[0], batch)
        or not sparse_piper_compile.same_dimension(projected.shape[1], attention.shape[1])
    ):
        return None

    weight_global_valid, weight_global = sparse_piper_compile.optional_tensor_metadata(
        linear_args["weight_per_tensor_scale"]
    )
    bias_valid, bias = sparse_piper_compile.optional_tensor_metadata(linear_args["bias"])
    activation_valid, activation_scale = sparse_piper_compile.optional_tensor_metadata(
        linear_args["activation_per_tensor_scale"]
    )
    if not weight_global_valid or not bias_valid or not activation_valid:
        return None
    try:
        _output.validate_optional_group_size(group_size, input_features)
        nvfp4_validation.validate_activation_scale(
            activation_scale,
            dynamic_activation_scale,
            query.device,
            "fused sparse Piper NVFP4 compiler output",
        )
        output_features = nvfp4_validation.validate_weight(
            weight_qdata,
            weight_scale,
            weight_global,
            bias,
            input_features=input_features,
            device=query.device,
            name="fused sparse Piper NVFP4 compiler output",
        )
    except ValueError:
        return None
    if not (
        projected.shape[2] == output_features
        and projected.device == query.device
        and projected.layout is torch.strided
        and projected.is_contiguous()
    ):
        return None
    return projected.dtype


def _match_attention_output(linear_node: torch.fx.Node) -> _MatchedAttentionOutput | None:  # noqa: PLR0911
    """Anchor-match one attention -> reshape -> (ConvRot) NVFP4 linear region."""
    op_name = linear_fx.schema_name(linear_node)
    if op_name not in (_NVFP4_LINEAR, _CONVROT_NVFP4_LINEAR):
        return None
    is_convrot = op_name == _CONVROT_NVFP4_LINEAR
    linear_args = linear_fx.bind_arguments(linear_node)
    if linear_args is None:
        return None

    reshaped_match = sparse_piper_compile.match_reshaped_quantized_attention(linear_args["input"])
    if reshaped_match is None:
        return None
    attention_args = reshaped_match.attention_args

    dynamic_activation_scale = linear_args["dynamic_activation_scale"]
    high_first = linear_args.get("high_first", False)
    if not isinstance(dynamic_activation_scale, bool) or not isinstance(high_first, bool):
        return None

    group_size: int | None = None
    if is_convrot:
        raw_group_size = linear_args["group_size"]
        if isinstance(raw_group_size, bool) or not isinstance(raw_group_size, int):
            return None
        group_size = raw_group_size

    output_dtype = _valid_attention_output_metadata(
        linear_node,
        reshaped_match.attention_node,
        reshaped_match.attention_shape,
        linear_args,
        attention_args,
        dynamic_activation_scale=dynamic_activation_scale,
        group_size=group_size,
    )
    if output_dtype is None:
        return None

    return _MatchedAttentionOutput(
        linear_node=linear_node,
        reshape_node=reshaped_match.reshape_node,
        attention_node=reshaped_match.attention_node,
        attention_arguments=reshaped_match.attention_arguments,
        projection_arguments=(
            linear_args["weight_qdata"],
            linear_args["weight_scale"],
            linear_args["weight_per_tensor_scale"],
            linear_args["activation_per_tensor_scale"],
            linear_args["bias"],
        ),
        bounded_arguments=reshaped_match.bounded_arguments,
        force_kwargs=reshaped_match.force_kwargs,
        high_first=high_first,
        dynamic_activation_scale=dynamic_activation_scale,
        group_size=group_size,
        output_dtype=output_dtype,
    )


def _replace_attention_output(graph: torch.fx.Graph, matched: _MatchedAttentionOutput) -> None:
    """Emit one NVFP4 output operator with a materialized or deferred gate."""
    original = matched.linear_node
    query_chunk_rows = _output.default_query_chunk_rows(matched.dynamic_activation_scale)
    block_lengths, coarse_gate, coarse_scale, coarse_key_blocks, query_blocks = (
        matched.bounded_arguments
    )
    bounded_arguments = matched.bounded_arguments
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
        gate_arguments = cast(tuple[Argument, ...], gate_projection)

    attention_arguments = matched.attention_arguments
    target = torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default
    query_projection = _prepared_query_projection(
        attention_arguments[0],
        attention_arguments[1],
        attention_arguments[2],
        attention_arguments[-1],
        block_lengths,
    )
    if query_projection is not None:
        target = torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        attention_arguments = (*query_projection, *attention_arguments[3:])

    with graph.inserting_before(original):
        replacement = graph.call_function(
            target,
            args=(
                *attention_arguments,
                *matched.projection_arguments,
                query_chunk_rows,
                *bounded_arguments,
                *gate_arguments,
            ),
            kwargs={
                "high_first": matched.high_first,
                "output_dtype": matched.output_dtype,
                "dynamic_activation_scale": matched.dynamic_activation_scale,
                "group_size": matched.group_size,
                **matched.force_kwargs,
            },
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    graph.erase_node(original)
    graph.erase_node(matched.reshape_node)
    graph.erase_node(matched.attention_node)


def _fold_attention_output(graph: torch.fx.Graph) -> bool:
    """Replace compatible materialized attention-to-(ConvRot)-NVFP4 regions."""
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
