"""Compiler folding for sparse Piper attention followed by a ConvRot projection."""

from __future__ import annotations

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
from piper_kernels.fusions.sparse_piper import _pattern as sparse_piper_pattern
from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _layout, output


def _attention_output_pattern(
    *,
    explicit_activation: bool,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
) -> CallFunction:
    reshaped = sparse_piper_pattern.reshaped_quantized_attention_pattern(
        with_block_lengths=with_block_lengths,
        with_coarse=with_coarse,
        with_sparse_query_blocks=with_sparse_query_blocks,
    )
    arguments: list[object] = [
        reshaped,
        KeywordArg("output_weight_qdata"),
        KeywordArg("output_weight_scale"),
        KeywordArg("output_bias"),
        KeywordArg("output_group_size"),
    ]
    if explicit_activation:
        arguments.append(None)
    return CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        *arguments,
    )


def _static_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _same_dimension(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


def _valid_attention_output(match: Match) -> bool:  # noqa: PLR0911, PLR0912
    tensor_names = (
        "output_query",
        "output_weight_qdata",
        "output_weight_scale",
    )
    if any(not isinstance(match.kwargs[name], torch.fx.Node) for name in tensor_names):
        return False
    metadata = {
        name: preparation_sharing.tensor_metadata(match.kwargs[name])  # type: ignore[arg-type]
        for name in tensor_names
    }
    if any(value is None for value in metadata.values()):
        return False
    query = metadata["output_query"]
    weight = metadata["output_weight_qdata"]
    scale = metadata["output_weight_scale"]
    projected_node = match.output_node()
    reshaped_node = projected_node.args[0]
    if not isinstance(reshaped_node, torch.fx.Node):
        return False
    attention_node = reshaped_node.args[0]
    if not isinstance(attention_node, torch.fx.Node):
        return False
    attention = preparation_sharing.tensor_metadata(attention_node)
    reshaped = preparation_sharing.tensor_metadata(reshaped_node)
    projected = preparation_sharing.tensor_metadata(projected_node)
    if any(value is None for value in (query, weight, scale, attention, reshaped, projected)):
        return False
    assert query is not None
    assert weight is not None
    assert scale is not None
    assert attention is not None
    assert reshaped is not None
    assert projected is not None
    if (
        query.ndim != 4
        or attention.ndim != 4
        or reshaped.ndim != 3
        or projected.ndim != 3
        or weight.ndim != 2
        or query.dtype is not torch.int8
        or attention.dtype is not torch.bfloat16
        or reshaped.dtype is not torch.bfloat16
        or projected.dtype is not torch.bfloat16
        or weight.dtype is not torch.int8
        or scale.dtype is not torch.float32
        or query.device.type != "cuda"
    ):
        return False
    target = AcceleratorTarget.from_device(query.device)
    if not target.is_cuda_capability(12, 0):
        return False

    batch, sequence_length = attention.shape[:2]
    heads = _static_int(attention.shape[2])
    head_dim = _static_int(attention.shape[3])
    input_features = _static_int(weight.shape[1])
    output_features = _static_int(weight.shape[0])
    if (
        input_features is None
        or output_features is None
        or heads is None
        or head_dim != _layout.HEAD_DIM
        or input_features != heads * head_dim
        or output_features < 1
        or not _same_dimension(reshaped.shape[0], batch)
        or not _same_dimension(reshaped.shape[1], sequence_length)
        or reshaped.shape[2] != input_features
        or not _same_dimension(projected.shape[0], batch)
        or not _same_dimension(projected.shape[1], sequence_length)
        or projected.shape[2] != output_features
        or tuple(scale.shape) != (output_features, 1)
        or not _same_dimension(query.shape[0], batch)
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

    shape = match.kwargs["output_attention_shape"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        return False
    group_size = _static_int(match.kwargs["output_group_size"])
    if group_size is None or group_size < 1 or input_features % group_size:
        return False
    bias_argument = match.kwargs["output_bias"]
    if bias_argument is None:
        return True
    if not isinstance(bias_argument, torch.fx.Node):
        return False
    bias = preparation_sharing.tensor_metadata(bias_argument)
    return bool(
        bias is not None
        and bias.shape == (output_features,)
        and bias.dtype is torch.bfloat16
        and bias.device == query.device
        and bias.layout is torch.strided
        and bias.is_contiguous()
    )


def _replace_attention_output(  # noqa: PLR0913, PLR0917
    match: Match,
    output_query: torch.fx.Node,
    output_query_scale: torch.fx.Node,
    output_query_summary: torch.fx.Node,
    output_key: torch.fx.Node,
    output_key_scale: torch.fx.Node,
    output_key_summary: torch.fx.Node,
    output_key_aux: torch.fx.Node,
    output_value: torch.fx.Node,
    output_value_scale_multiplier: torch.fx.Node,
    output_value_mean: torch.fx.Node,
    output_head_keep_ratio_units: list[int],
    output_sparse_key_blocks: Argument,
    output_logical_sequence_length: Argument,
    output_routing_mode: int,
    output_weight_qdata: torch.fx.Node,
    output_weight_scale: torch.fx.Node,
    output_bias: torch.fx.Node | None,
    output_group_size: int,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.convrot_sparse_piper_attention_output.default,
            args=(
                output_query,
                output_query_scale,
                output_query_summary,
                output_key,
                output_key_scale,
                output_key_summary,
                output_key_aux,
                output_value,
                output_value_scale_multiplier,
                output_value_mean,
                output_head_keep_ratio_units,
                output_sparse_key_blocks,
                output_logical_sequence_length,
                output_routing_mode,
                output_weight_qdata,
                output_weight_scale,
                output_bias,
                output_group_size,
                output._DEFAULT_QUERY_CHUNK_ROWS,
                *sparse_piper_pattern.bounded_attention_arguments(match),
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


_patterns = PatternMatcherPass("convrot_sparse_piper_attention_output")
for _explicit_activation in (False, True):
    for _with_block_lengths in (False, True):
        for _with_coarse in (False, True):
            for _with_sparse_query_blocks in (False, True):
                register_graph_pattern(
                    _attention_output_pattern(
                        explicit_activation=_explicit_activation,
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
