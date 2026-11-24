"""Compiler folding for sparse Piper attention followed by NVFP4 or ConvRot NVFP4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx.node import Argument

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._contracts import SUPPORTED_DTYPES
from piper_kernels.fusions.attention import _sparse_compile as sparse_piper_compile
from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _compile_fx
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops  # noqa: F401 - registers nvfp4_linear
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation

from . import output

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
    reshaped: sparse_piper_compile.ReshapedQuantizedAttentionMatch
    projection_arguments: tuple[Argument, ...]
    high_first: bool
    dynamic_activation_scale: bool
    group_size: int | None
    output_dtype: torch.dtype


def _prepared_gate_projection(
    gate: object,
) -> tuple[torch.fx.Node | None, ...] | None:
    """Extract a reshaped prepared NVFP4 gate for bounded projection."""
    linear = sparse_piper_compile.unwrap_reshaped_producer(gate)
    if linear is None or linear.target is not torch.ops.piper_kernels.nvfp4_linear_prepared.default:
        return None
    assert isinstance(gate, torch.fx.Node)
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
    return operands.storage_arguments()


def _valid_attention_output_metadata(
    linear_node: torch.fx.Node,
    reshaped_match: sparse_piper_compile.ReshapedQuantizedAttentionMatch,
    linear_args: dict[str, Argument],
    *,
    dynamic_activation_scale: bool,
    group_size: int | None,
) -> torch.dtype | None:
    """Validate tensor metadata for one attention -> reshape -> (ConvRot) NVFP4 match."""
    common = sparse_piper_compile.validate_reshaped_attention_output(linear_node, reshaped_match)
    if common is None:
        return None
    weight_qdata_node = linear_args["weight_qdata"]
    weight_scale_node = linear_args["weight_scale"]
    if not isinstance(weight_qdata_node, torch.fx.Node) or not isinstance(
        weight_scale_node, torch.fx.Node
    ):
        return None
    weight_qdata = preparation_sharing.tensor_metadata(weight_qdata_node)
    weight_scale = preparation_sharing.tensor_metadata(weight_scale_node)
    weight_global_valid, weight_global = sparse_piper_compile.optional_tensor_metadata(
        linear_args["weight_per_tensor_scale"]
    )
    bias_valid, bias = sparse_piper_compile.optional_tensor_metadata(linear_args["bias"])
    activation_valid, activation_scale = sparse_piper_compile.optional_tensor_metadata(
        linear_args["activation_per_tensor_scale"]
    )
    if (
        weight_qdata is None
        or weight_scale is None
        or not weight_global_valid
        or not bias_valid
        or not activation_valid
    ):
        return None
    device = common.query.device
    try:
        output._validate_optional_group_size(group_size, common.input_features)
        nvfp4_validation.validate_activation_scale(
            activation_scale,
            dynamic_activation_scale,
            device,
            "fused sparse Piper NVFP4 compiler output",
        )
        output_features = nvfp4_validation.validate_weight(
            weight_qdata,
            weight_scale,
            weight_global,
            bias,
            input_features=common.input_features,
            device=device,
            name="fused sparse Piper NVFP4 compiler output",
        )
    except ValueError:
        return None
    if (
        common.projected.shape[2] != output_features
        or device.type != "cuda"
        or not AcceleratorTarget.from_device(device).is_cuda_capability(12, 0)
    ):
        return None
    return common.projected.dtype


def _match_attention_output(linear_node: torch.fx.Node) -> _MatchedAttentionOutput | None:  # noqa: PLR0911
    """Anchor-match one attention -> reshape -> (ConvRot) NVFP4 linear region."""
    op_name = linear_fx.schema_name(linear_node)
    if op_name not in (_NVFP4_LINEAR, _CONVROT_NVFP4_LINEAR):
        return None
    is_convrot = op_name == _CONVROT_NVFP4_LINEAR
    linear_args = linear_fx.bind_arguments(linear_node)
    if linear_args is None or linear_args.get("weight_only") is True:
        return None

    reshaped_match = sparse_piper_compile.match_reshaped_quantized_attention(linear_args["input"])
    if reshaped_match is None:
        return None

    dynamic_activation_scale = linear_args["dynamic_activation_scale"]
    high_first = linear_args.get("high_first", False)
    if not isinstance(dynamic_activation_scale, bool) or not isinstance(high_first, bool):
        return None

    group_size: int | None = None
    if is_convrot:
        group_size = sparse_piper_compile.static_int(linear_args["group_size"])
        if group_size is None:
            return None

    output_dtype = _valid_attention_output_metadata(
        linear_node,
        reshaped_match,
        linear_args,
        dynamic_activation_scale=dynamic_activation_scale,
        group_size=group_size,
    )
    if output_dtype is None:
        return None

    return _MatchedAttentionOutput(
        linear_node=linear_node,
        reshaped=reshaped_match,
        projection_arguments=(
            linear_args["weight_qdata"],
            linear_args["weight_scale"],
            linear_args["weight_per_tensor_scale"],
            linear_args["activation_per_tensor_scale"],
            linear_args["bias"],
        ),
        high_first=high_first,
        dynamic_activation_scale=dynamic_activation_scale,
        group_size=group_size,
        output_dtype=output_dtype,
    )


def _replace_attention_output(graph: torch.fx.Graph, matched: _MatchedAttentionOutput) -> None:
    """Emit one NVFP4 output operator with a materialized or deferred gate."""
    original = matched.linear_node
    query_chunk_rows = output._default_query_chunk_rows(matched.dynamic_activation_scale)
    bounded_arguments, gate_projection = matched.reshaped.defer_gate(_prepared_gate_projection)
    gate_arguments: tuple[Argument, ...] = () if gate_projection is None else gate_projection

    attention_arguments = matched.reshaped.attention_arguments
    target = torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default
    query_bound = matched.reshaped.match_projected_query(
        torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default
    )
    if query_bound is not None:
        target = torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        attention_arguments = (
            *(query_bound[name] for name in _QUERY_PROJECTION_ARGUMENT_NAMES),
            *attention_arguments[3:],
        )

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
                **matched.reshaped.force_kwargs,
            },
        )
    sparse_piper_compile.replace_matched_nodes(
        graph,
        original,
        replacement,
        matched.reshaped.reshape_node,
        matched.reshaped.attention_node,
    )


def _fold_attention_output(graph: torch.fx.Graph) -> bool:
    """Replace compatible materialized attention-to-(ConvRot)-NVFP4 regions."""
    return sparse_piper_compile.fold_matched_nodes(
        graph,
        _match_attention_output,
        _replace_attention_output,
    )


__all__: list[str] = []
