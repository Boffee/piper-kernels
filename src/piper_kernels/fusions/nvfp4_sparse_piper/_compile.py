"""NVFP4 projection folding for sparse Piper attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.fx.node import Argument

from piper_kernels.attention.kernels.sparse_piper import layout
from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_triton,
)
from piper_kernels.attention.sparse_piper_attention import (
    _quantized_dispatch,
    dispatch,
)
from piper_kernels.attention.sparse_piper_attention._routes import (
    _MEAN_ROUTING,
    is_valid_routing_mode,
)
from piper_kernels.fusions.projected_qk import triton as projected_qk_triton
from piper_kernels.fusions.sparse_piper import _compile as sparse_piper_compile
from piper_kernels.fusions.sparse_piper import _output as sparse_piper_output
from piper_kernels.fusions.sparse_piper import _pattern as sparse_piper_pattern
from piper_kernels.linear import _compile_fx as linear_compile_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _chunking as nvfp4_chunking
from piper_kernels.linear.nvfp4 import _compile as nvfp4_compile
from piper_kernels.linear.nvfp4 import _compile_fx
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import _projection as nvfp4_projection
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation
from piper_kernels.linear.nvfp4 import triton as nvfp4_triton

from . import _epilogue, _output, _output_compile, _validation, key, output, query, value

_COMPILE_PASS_VERSION = "nvfp4-sparse-piper-compile-v4"
_HEAD_DIM = layout.HEAD_DIM
_TILE_ROWS = layout.TILE_ROWS
_QUERY_SCALE_ROWS = layout.QUERY_SCALE_ROWS


def _source_files() -> tuple[str, ...]:
    return tuple(
        file_name
        for file_name in (
            __file__,
            nvfp4_chunking.__file__,
            _epilogue.__file__,
            _output.__file__,
            _validation.__file__,
            _output_compile.__file__,
            key.__file__,
            output.__file__,
            query.__file__,
            value.__file__,
            layout.__file__,
            sparse_piper_triton.__file__,
            *sparse_piper_compile.source_files(),
            sparse_piper_output.__file__,
            sparse_piper_pattern.__file__,
            linear_compile_fx.__file__,
            nvfp4_layout.__file__,
            projected_qk_triton.__file__,
            nvfp4_projection.__file__,
            nvfp4_validation.__file__,
            nvfp4_triton.__file__,
            _quantized_dispatch.__file__,
            dispatch.__file__,
            _compile_fx.__file__,
        )
        if file_name is not None
    )


def _linear_pattern(prefix: str) -> CallFunction:
    prepared = CallFunction(
        torch.ops.piper_kernels.nvfp4_linear_prepared.default,
        KeywordArg(f"{prefix}_input_qdata"),
        KeywordArg(f"{prefix}_input_scale"),
        KeywordArg(f"{prefix}_input_per_tensor_scale"),
        KeywordArg(f"{prefix}_weight_qdata"),
        KeywordArg(f"{prefix}_weight_scale"),
        KeywordArg(f"{prefix}_weight_per_tensor_scale"),
        KeywordArg(f"{prefix}_bias"),
        KeywordArg(f"{prefix}_logical_dtype"),
        _users=1,
    )
    return CallFunction(
        torch.ops.aten.reshape.default,
        prepared,
        KeywordArg(f"{prefix}_linear_shape"),
        _users=1,
    )


def _optional_tensor_metadata(value: object) -> tuple[bool, torch.Tensor | None]:
    if value is None:
        return True, None
    if not isinstance(value, torch.fx.Node):
        return False, None
    metadata = preparation_sharing.tensor_metadata(value)
    return metadata is not None, metadata


def _valid_projection(match: Match) -> bool:  # noqa: PLR0911
    if not is_valid_routing_mode(match.kwargs["sparse_routing_mode"]):
        return False
    prefixes = ("sparse_q", "sparse_k", "sparse_v")
    projections = tuple(
        _compile_fx.PreparedLinearNodes.from_match(match, prefix) for prefix in prefixes
    )
    if any(projection is None for projection in projections):
        return False
    q_projection = projections[0]
    assert q_projection is not None
    q_input = preparation_sharing.tensor_metadata(q_projection.input_qdata)
    q_weight = preparation_sharing.tensor_metadata(q_projection.weight_qdata)
    if q_input is None or q_weight is None or q_input.ndim != 2:
        return False
    sequence_length = q_input.shape[0]
    output_features = sparse_piper_compile.static_int(q_weight.shape[0])
    if output_features is None or output_features % _HEAD_DIM:
        return False
    heads = output_features // _HEAD_DIM

    for prefix, projection in zip(prefixes, projections, strict=True):
        assert projection is not None
        input_qdata = preparation_sharing.tensor_metadata(projection.input_qdata)
        input_scale = preparation_sharing.tensor_metadata(projection.input_scale)
        input_global_scale = preparation_sharing.tensor_metadata(projection.input_per_tensor_scale)
        weight = preparation_sharing.tensor_metadata(projection.weight_qdata)
        weight_scale = preparation_sharing.tensor_metadata(projection.weight_scale)
        global_scale_valid, global_scale = _optional_tensor_metadata(
            projection.weight_per_tensor_scale
        )
        bias_valid, bias = _optional_tensor_metadata(projection.bias)
        if (
            input_qdata is None
            or input_scale is None
            or input_global_scale is None
            or weight is None
            or weight_scale is None
            or not global_scale_valid
            or not bias_valid
        ):
            return False
        try:
            projection_sequence_length, projection_heads = _validation.validate_projection(
                input_qdata,
                input_scale,
                input_global_scale,
                weight,
                weight_scale,
                global_scale,
                bias,
                nvfp4_chunking.DEFAULT_CHUNK_ROWS,
                f"{prefix} compiler projection",
            )
        except ValueError:
            return False

        linear_shape = match.kwargs[f"{prefix}_linear_shape"]
        linear_sequence_length = (
            sparse_piper_compile.integer_scalar_metadata(linear_shape[1])
            if isinstance(linear_shape, (list, tuple)) and len(linear_shape) == 3
            else None
        )
        if (
            preparation_sharing.dimension_key(projection_sequence_length)
            != preparation_sharing.dimension_key(sequence_length)
            or projection_heads != heads
            or not isinstance(linear_shape, (list, tuple))
            or len(linear_shape) != 3
            or sparse_piper_compile.static_int(linear_shape[0]) != 1
            or linear_sequence_length is None
            or preparation_sharing.dimension_key(linear_sequence_length)
            != preparation_sharing.dimension_key(sequence_length)
            or sparse_piper_compile.static_int(linear_shape[2]) != output_features
            or projection.logical_dtype is not torch.bfloat16
        ):
            return False

    return sparse_piper_compile.valid_sparse_piper_attention(
        match,
        batch=1,
        sequence_length=sequence_length,
        heads=heads,
        device=q_input.device,
        head_dim=_HEAD_DIM,
        tile_rows=_TILE_ROWS,
    )


def _matched_node(match: Match, name: str) -> torch.fx.Node:
    node = match.kwargs[name]
    assert isinstance(node, torch.fx.Node)
    return node


def _replace_projection(
    match: Match,
    sparse_attention_shape: list[Argument],
    sparse_head_keep_ratio_units: list[int],
    sparse_q_norm_epsilon: float,
    sparse_k_norm_epsilon: float,
    sparse_key_blocks: Argument,
    sparse_softmax_scale: float,
    sparse_routing_mode: int,
    sparse_query_blocks: Argument | None = None,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    query_projection = _compile_fx.PreparedLinearNodes.from_match(match, "sparse_q")
    key_projection = _compile_fx.PreparedLinearNodes.from_match(match, "sparse_k")
    value_projection = _compile_fx.PreparedLinearNodes.from_match(match, "sparse_v")
    assert query_projection is not None
    assert key_projection is not None
    assert value_projection is not None
    q_norm_weight = _matched_node(match, "sparse_q_norm_weight")
    k_norm_weight = _matched_node(match, "sparse_k_norm_weight")
    cos = _matched_node(match, "sparse_cos")
    sin = _matched_node(match, "sparse_sin")
    input_value = preparation_sharing.tensor_metadata(query_projection.input_qdata)
    q_weight_value = preparation_sharing.tensor_metadata(query_projection.weight_qdata)
    assert input_value is not None
    assert q_weight_value is not None
    sequence_length = input_value.shape[0]
    heads = q_weight_value.shape[0] // _HEAD_DIM
    storage_sequence_length = layout.padded_sequence_length(sequence_length)
    logical_sequence_length = sparse_piper_compile.integer_scalar_argument(
        sparse_attention_shape[1]
    )
    assert logical_sequence_length is not None
    with graph.inserting_before(original):
        query_values = (
            input_value.new_empty((1, heads, storage_sequence_length, _HEAD_DIM), dtype=torch.int8),
            input_value.new_empty(
                (1, heads, storage_sequence_length // _QUERY_SCALE_ROWS),
                dtype=torch.float32,
            ),
            input_value.new_empty(
                (1, heads, storage_sequence_length // _TILE_ROWS, _HEAD_DIM),
                dtype=torch.float32,
            ),
        )
        query_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default,
            (
                *query_projection.storage_arguments(),
                q_norm_weight,
                cos,
                sin,
                sparse_q_norm_epsilon,
                sparse_softmax_scale,
                nvfp4_chunking.DEFAULT_CHUNK_ROWS,
                sparse_routing_mode,
            ),
            query_values,
        )
        key_summary = input_value.new_empty(
            (1, heads, storage_sequence_length // _TILE_ROWS, _HEAD_DIM),
            dtype=torch.float32,
        )
        key_values = (
            input_value.new_empty((1, heads, storage_sequence_length, _HEAD_DIM), dtype=torch.int8),
            input_value.new_empty(
                (1, heads, storage_sequence_length // _TILE_ROWS), dtype=torch.float32
            ),
            key_summary,
            (
                input_value.new_empty((1, heads, 0, _HEAD_DIM), dtype=torch.float32)
                if sparse_routing_mode == _MEAN_ROUTING
                else input_value.new_empty(key_summary.shape, dtype=torch.float32)
            ),
        )
        key_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default,
            (
                *key_projection.storage_arguments(),
                k_norm_weight,
                cos,
                sin,
                sparse_k_norm_epsilon,
                nvfp4_chunking.DEFAULT_CHUNK_ROWS,
                sparse_routing_mode,
            ),
            key_values,
        )
        value_mean_flat = graph.call_function(
            torch.ops.piper_kernels.nvfp4_linear_mean.default,
            args=(
                *value_projection.storage_arguments(),
                1,
                logical_sequence_length,
            ),
        )
        value_mean_flat.meta["val"] = input_value.new_empty(
            (1, heads * _HEAD_DIM), dtype=torch.float32
        )
        value_mean = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(value_mean_flat, (1, heads, _HEAD_DIM)),
        )
        value_mean.meta["val"] = input_value.new_empty((1, heads, _HEAD_DIM), dtype=torch.float32)
        value_values = (
            input_value.new_empty((1, heads, _HEAD_DIM, storage_sequence_length), dtype=torch.int8),
            input_value.new_empty(
                (1, heads, storage_sequence_length // _TILE_ROWS, 1),
                dtype=torch.float32,
            ),
        )
        value_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default,
            (
                *value_projection.storage_arguments(),
                value_mean,
                nvfp4_chunking.DEFAULT_CHUNK_ROWS,
            ),
            value_values,
        )
        replacement = graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
            args=(
                *query_nodes,
                *key_nodes,
                *value_nodes,
                value_mean,
                sparse_head_keep_ratio_units,
                sparse_key_blocks,
                logical_sequence_length,
                sparse_routing_mode,
                *sparse_piper_pattern.optional_attention_layout_arguments(
                    None,
                    sparse_query_blocks,
                ),
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


_patterns = PatternMatcherPass("nvfp4_sparse_piper_projection")
for _with_sparse_query_blocks in (True, False):
    register_graph_pattern(
        sparse_piper_pattern.sparse_piper_projection_pattern(
            _linear_pattern,
            with_sparse_query_blocks=_with_sparse_query_blocks,
        ),
        extra_check=_valid_projection,
        pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
    )(_replace_projection)


def _fold_projection(graph: torch.fx.Graph) -> bool:
    changed = _patterns.apply(graph) > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            _fold_projection(graph)
            _output_compile._fold_attention_output(graph)

    def uuid(self) -> bytes:
        return get_hash_for_files(_source_files(), extra=_COMPILE_PASS_VERSION)


compile_pass = _CompilePass()


def nvfp4_sparse_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Canonicalize repeated NVFP4 projections before sparse-attention fusion."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (nvfp4_compile.compile_pass, compile_pass),
    )


__all__ = ["nvfp4_sparse_piper_compile_options"]
