"""Compiler folding for sparse Piper attention followed by static NVFP4."""

from __future__ import annotations

import torch
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.sparse_piper import _compile as sparse_piper_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation

from . import output

_ATTENTION_ARGUMENT_NAMES = (
    "output_query",
    "output_query_scale",
    "output_query_summary",
    "output_key",
    "output_key_scale",
    "output_key_max",
    "output_key_min",
    "output_value",
    "output_value_scale_multiplier",
    "output_value_mean",
    "output_head_keep_ratio_units",
    "output_sparse_key_blocks",
    "output_logical_sequence_length",
)


def _reshaped_attention_pattern() -> CallFunction:
    """Match the common materialized sparse-attention output boundary."""
    attention = CallFunction(
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
        KeywordArg("output_query"),
        KeywordArg("output_query_scale"),
        KeywordArg("output_query_summary"),
        KeywordArg("output_key"),
        KeywordArg("output_key_scale"),
        KeywordArg("output_key_max"),
        KeywordArg("output_key_min"),
        KeywordArg("output_value"),
        KeywordArg("output_value_scale_multiplier"),
        KeywordArg("output_value_mean"),
        KeywordArg("output_head_keep_ratio_units"),
        KeywordArg("output_sparse_key_blocks"),
        KeywordArg("output_logical_sequence_length"),
        _users=1,
    )
    return CallFunction(
        torch.ops.aten.reshape.default,
        attention,
        KeywordArg("output_attention_shape"),
        _users=1,
    )


def _attention_output_pattern() -> CallFunction:
    return CallFunction(
        torch.ops.piper_kernels.nvfp4_linear.default,
        _reshaped_attention_pattern(),
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
    logical_sequence_length = sparse_piper_compile.integer_scalar_metadata(
        match.kwargs["output_logical_sequence_length"]
    )
    attention_shape = _shape_dimensions(match.kwargs["output_attention_shape"])
    if (
        heads is None
        or head_dim != 128
        or logical_sequence_length is None
        or attention_shape is None
        or len(attention_shape) != 3
    ):
        return False
    input_features = heads * head_dim
    if (
        not _same_dimension(attention_shape[0], batch)
        or not _same_dimension(attention_shape[1], logical_sequence_length)
        or attention_shape[2] != input_features
        or not _same_dimension(projected.shape[0], batch)
        or not _same_dimension(projected.shape[1], logical_sequence_length)
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


def _replace_attention_output(match: Match, **_unused: object) -> None:
    original = match.output_node()
    graph = match.graph
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default,
            args=(
                *(match.kwargs[name] for name in _ATTENTION_ARGUMENT_NAMES),
                match.kwargs["output_weight_qdata"],
                match.kwargs["output_weight_scale"],
                match.kwargs["output_weight_per_tensor_scale"],
                match.kwargs["output_activation_scale"],
                match.kwargs["output_bias"],
                output._DEFAULT_QUERY_CHUNK_ROWS,
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


_patterns = PatternMatcherPass("nvfp4_sparse_piper_attention_output")
register_graph_pattern(
    _attention_output_pattern(),
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
