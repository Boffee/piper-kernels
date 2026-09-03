"""Compiler folding for sparse Piper attention followed by static NVFP4."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.fx.node import Argument

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.sparse_piper import _compile as sparse_piper_compile
from piper_kernels.fusions.sparse_piper import _pattern as sparse_piper_pattern
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _compile_fx
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation

from . import output

type _PreparedGateProjectionNodes = tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node | None,
    torch.fx.Node | None,
]


def _attention_output_pattern(
    *,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
) -> CallFunction:
    return CallFunction(
        torch.ops.piper_kernels.nvfp4_linear.default,
        sparse_piper_pattern.reshaped_quantized_attention_pattern(
            with_block_lengths=with_block_lengths,
            with_coarse=with_coarse,
            with_sparse_query_blocks=with_sparse_query_blocks,
        ),
        KeywordArg("output_weight_qdata"),
        KeywordArg("output_weight_scale"),
        KeywordArg("output_weight_per_tensor_scale"),
        KeywordArg("output_activation_scale"),
        KeywordArg("output_bias"),
        False,
    )


def _optional_tensor_metadata(value: object) -> tuple[bool, torch.Tensor | None]:
    if value is None:
        return True, None
    if not isinstance(value, torch.fx.Node):
        return False, None
    metadata = preparation_sharing.tensor_metadata(value)
    return metadata is not None, metadata


def _shape_dimensions(value: object) -> tuple[int | torch.SymInt, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    dimensions = tuple(
        sparse_piper_compile.integer_scalar_metadata(dimension) for dimension in value
    )
    if any(dimension is None for dimension in dimensions):
        return None
    return dimensions  # type: ignore[return-value]


def _same_dimension(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


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
    if operands is None or operands.logical_dtype is not torch.bfloat16:
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
        or gate_value.dtype is not torch.bfloat16
        or gate_value.shape[0] != 1
        or not _same_dimension(gate_value.shape[1], shape.rows)
        or not _same_dimension(
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


def _valid_attention_output(match: Match) -> bool:  # noqa: PLR0911
    required_names = (
        "output_query",
        "output_activation_scale",
        "output_weight_qdata",
        "output_weight_scale",
    )
    if any(not isinstance(match.kwargs[name], torch.fx.Node) for name in required_names):
        return False
    metadata = {
        name: preparation_sharing.tensor_metadata(match.kwargs[name])  # type: ignore[arg-type]
        for name in required_names
    }
    if any(value is None for value in metadata.values()):
        return False
    query = metadata["output_query"]
    activation_scale = metadata["output_activation_scale"]
    weight_qdata = metadata["output_weight_qdata"]
    weight_scale = metadata["output_weight_scale"]
    projected = preparation_sharing.tensor_metadata(match.output_node())
    if any(
        value is None for value in (query, activation_scale, weight_qdata, weight_scale, projected)
    ):
        return False
    assert query is not None
    assert activation_scale is not None
    assert weight_qdata is not None
    assert weight_scale is not None
    assert projected is not None
    if (
        query.ndim != 4
        or query.dtype is not torch.int8
        or projected.ndim != 3
        or projected.dtype is not torch.bfloat16
        or query.device.type != "cuda"
    ):
        return False
    if not AcceleratorTarget.from_device(query.device).is_cuda_capability(12, 0):
        return False

    batch = query.shape[0]
    heads = sparse_piper_compile.static_int(query.shape[1])
    head_dim = sparse_piper_compile.static_int(query.shape[3])
    reshaped_node = match.output_node().args[0]
    if not isinstance(reshaped_node, torch.fx.Node):
        return False
    attention_node = reshaped_node.args[0]
    if not isinstance(attention_node, torch.fx.Node):
        return False
    attention = preparation_sharing.tensor_metadata(attention_node)
    attention_shape = _shape_dimensions(match.kwargs["output_attention_shape"])
    if (
        heads is None
        or head_dim != 128
        or attention is None
        or attention.ndim != 4
        or attention.dtype is not torch.bfloat16
        or attention.device != query.device
        or attention.layout is not torch.strided
        or not attention.is_contiguous()
        or attention_shape is None
        or len(attention_shape) != 3
    ):
        return False
    input_features = heads * head_dim
    if (
        not _same_dimension(attention.shape[0], batch)
        or attention.shape[2] != heads
        or attention.shape[3] != head_dim
        or not _same_dimension(attention_shape[0], batch)
        or not _same_dimension(attention_shape[1], attention.shape[1])
        or attention_shape[2] != input_features
        or not _same_dimension(projected.shape[0], batch)
        or not _same_dimension(projected.shape[1], attention.shape[1])
    ):
        return False

    weight_global_valid, weight_global = _optional_tensor_metadata(
        match.kwargs["output_weight_per_tensor_scale"]
    )
    bias_valid, bias = _optional_tensor_metadata(match.kwargs["output_bias"])
    if not weight_global_valid or not bias_valid:
        return False
    try:
        nvfp4_validation.validate_activation_scale(
            activation_scale,
            False,
            query.device,
            "fused sparse Piper NVFP4 compiler output",
        )
        output_features = nvfp4_validation.validate_weight(
            weight_qdata,
            weight_scale,
            weight_global,
            bias,
            input_features=input_features,
            logical_dtype=torch.bfloat16,
            device=query.device,
            name="fused sparse Piper NVFP4 compiler output",
        )
    except ValueError:
        return False
    return bool(
        projected.shape[2] == output_features
        and projected.device == query.device
        and projected.layout is torch.strided
        and projected.is_contiguous()
    )


def _replace_attention_output_with_target(
    match: Match,
    target: Callable[..., torch.Tensor],
    projection_arguments: tuple[Argument, ...],
    query_chunk_rows: int,
) -> None:
    """Emit one NVFP4 output operator with a materialized or deferred gate."""
    original = match.output_node()
    graph = match.graph
    bounded_arguments = sparse_piper_pattern.bounded_attention_arguments(match)
    block_lengths, block_mean, coarse_gate, coarse_scale, coarse_key_blocks, query_blocks = (
        bounded_arguments
    )
    gate_projection = _prepared_gate_projection(coarse_gate)
    gate_arguments: tuple[Argument, ...] = ()
    if gate_projection is not None:
        bounded_arguments = (
            block_lengths,
            block_mean,
            None,
            coarse_scale,
            coarse_key_blocks,
            query_blocks,
        )
        gate_arguments = cast(tuple[Argument, ...], gate_projection)
    with graph.inserting_before(original):
        replacement = graph.call_function(
            target,
            args=(
                *sparse_piper_pattern.quantized_attention_arguments(match),
                *projection_arguments,
                query_chunk_rows,
                *bounded_arguments,
                *gate_arguments,
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


def _replace_attention_output(match: Match, **_unused: object) -> None:
    _replace_attention_output_with_target(
        match,
        torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default,
        (
            match.kwargs["output_weight_qdata"],
            match.kwargs["output_weight_scale"],
            match.kwargs["output_weight_per_tensor_scale"],
            match.kwargs["output_activation_scale"],
            match.kwargs["output_bias"],
        ),
        output._DEFAULT_QUERY_CHUNK_ROWS,
    )


_patterns = PatternMatcherPass("nvfp4_sparse_piper_attention_output")
for _with_block_lengths in (False, True):
    for _with_coarse in (False, True):
        for _with_sparse_query_blocks in (False, True):
            register_graph_pattern(
                _attention_output_pattern(
                    with_block_lengths=_with_block_lengths,
                    with_coarse=_with_coarse,
                    with_sparse_query_blocks=_with_sparse_query_blocks,
                ),
                extra_check=_valid_attention_output,
                pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
            )(_replace_attention_output)


def _fold_attention_output(graph: torch.fx.Graph) -> bool:
    """Replace one compatible materialized attention-to-output region."""
    changed = _patterns.apply(graph) > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__: list[str] = []
