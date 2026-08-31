"""NVFP4 projection folding for sparse Piper attention."""

from __future__ import annotations

import math
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
    _budget,
    _quantized_dispatch,
    dispatch,
)
from piper_kernels.fusions.projected_qk import _pattern as projected_qk_pattern
from piper_kernels.fusions.projected_qk import triton as projected_qk_triton
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _compile as nvfp4_compile
from piper_kernels.linear.nvfp4 import _compile_fx
from piper_kernels.linear.nvfp4 import _projection as nvfp4_projection
from piper_kernels.linear.nvfp4 import triton as nvfp4_triton

from . import _chunking, _epilogue, _validation, key, query, value

_COMPILE_PASS_VERSION = "nvfp4-sparse-piper-compile-v2"
_HEAD_DIM = layout.HEAD_DIM
_TILE_ROWS = layout.TILE_ROWS
_QUERY_SCALE_ROWS = layout.QUERY_SCALE_ROWS


def _source_files() -> tuple[str, ...]:
    return tuple(
        file_name
        for file_name in (
            __file__,
            _chunking.__file__,
            _epilogue.__file__,
            _validation.__file__,
            key.__file__,
            query.__file__,
            value.__file__,
            layout.__file__,
            sparse_piper_triton.__file__,
            projected_qk_pattern.__file__,
            projected_qk_triton.__file__,
            nvfp4_projection.__file__,
            nvfp4_triton.__file__,
            _budget.__file__,
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


def _static_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _integer_scalar_metadata(value: object) -> int | torch.SymInt | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, torch.SymInt)):
        return value
    if isinstance(value, torch.fx.Node):
        metadata = getattr(value, "meta", {}).get("val")
        if isinstance(metadata, (int, torch.SymInt)) and not isinstance(metadata, bool):
            return metadata
    return None


def _integer_scalar_argument(value: object) -> Argument | None:
    if _integer_scalar_metadata(value) is None:
        return None
    return value if isinstance(value, (int, torch.SymInt, torch.fx.Node)) else None


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted > 0 else None


def _optional_tensor_metadata(value: object) -> tuple[bool, torch.Tensor | None]:
    if value is None:
        return True, None
    if not isinstance(value, torch.fx.Node):
        return False, None
    metadata = preparation_sharing.tensor_metadata(value)
    return metadata is not None, metadata


def _valid_projection(match: Match) -> bool:  # noqa: PLR0911, PLR0912, PLR0915
    prefixes = ("sparse_q", "sparse_k", "sparse_v")
    required_nodes = (
        *(
            f"{prefix}_{suffix}"
            for prefix in prefixes
            for suffix in (
                "input_qdata",
                "input_scale",
                "input_per_tensor_scale",
                "weight_qdata",
                "weight_scale",
            )
        ),
        "sparse_q_norm_weight",
        "sparse_k_norm_weight",
        "sparse_cos",
        "sparse_sin",
    )
    if any(not isinstance(match.kwargs[name], torch.fx.Node) for name in required_nodes):
        return False
    metadata = {
        name: preparation_sharing.tensor_metadata(match.kwargs[name])  # type: ignore[arg-type]
        for name in required_nodes
    }
    if any(value is None for value in metadata.values()):
        return False
    if any(
        value.layout is not torch.strided or not value.is_contiguous()
        for value in metadata.values()
        if value is not None
    ):
        return False

    q_input = metadata["sparse_q_input_qdata"]
    q_weight = metadata["sparse_q_weight_qdata"]
    assert q_input is not None
    assert q_weight is not None
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    shape = match.kwargs["sparse_attention_shape"]
    if (
        q_input.ndim != 2
        or output_value is None
        or output_value.ndim != 4
        or not isinstance(shape, (list, tuple))
        or len(shape) != 4
    ):
        return False
    batch = _static_int(shape[0])
    shape_sequence_length = _integer_scalar_metadata(shape[1])
    sequence_length = q_input.shape[0]
    output_features = _static_int(q_weight.shape[0])
    if (
        batch != 1
        or shape_sequence_length is None
        or preparation_sharing.dimension_key(shape_sequence_length)
        != preparation_sharing.dimension_key(sequence_length)
        or output_features is None
        or output_features % _HEAD_DIM
    ):
        return False
    heads = output_features // _HEAD_DIM
    output_shape = output_value.shape
    if (
        output_value.dtype is not torch.bfloat16
        or output_value.device != q_input.device
        or tuple(output_shape[2:]) != (heads, _HEAD_DIM)
        or preparation_sharing.dimension_key(output_shape[0])
        != preparation_sharing.dimension_key(batch)
        or preparation_sharing.dimension_key(output_shape[1])
        != preparation_sharing.dimension_key(sequence_length)
        or _static_int(shape[2]) != heads
        or _static_int(shape[3]) != _HEAD_DIM
        or (isinstance(sequence_length, int) and sequence_length < _TILE_ROWS)
    ):
        return False

    for prefix in prefixes:
        input_qdata = metadata[f"{prefix}_input_qdata"]
        input_scale = metadata[f"{prefix}_input_scale"]
        input_global_scale = metadata[f"{prefix}_input_per_tensor_scale"]
        weight = metadata[f"{prefix}_weight_qdata"]
        weight_scale = metadata[f"{prefix}_weight_scale"]
        global_scale_valid, global_scale = _optional_tensor_metadata(
            match.kwargs[f"{prefix}_weight_per_tensor_scale"]
        )
        bias_valid, bias = _optional_tensor_metadata(match.kwargs[f"{prefix}_bias"])
        linear_shape = match.kwargs[f"{prefix}_linear_shape"]
        logical_dtype = match.kwargs[f"{prefix}_logical_dtype"]
        assert input_qdata is not None
        assert input_scale is not None
        assert input_global_scale is not None
        assert weight is not None
        assert weight_scale is not None
        if not global_scale_valid or not bias_valid:
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
                query.DEFAULT_CHUNK_ROWS,
                f"{prefix} compiler projection",
            )
        except ValueError:
            return False
        linear_sequence_length = (
            _integer_scalar_metadata(linear_shape[1])
            if isinstance(linear_shape, (list, tuple)) and len(linear_shape) == 3
            else None
        )
        if (
            preparation_sharing.dimension_key(projection_sequence_length)
            != preparation_sharing.dimension_key(sequence_length)
            or projection_heads != heads
            or not isinstance(linear_shape, (list, tuple))
            or len(linear_shape) != 3
            or _static_int(linear_shape[0]) != 1
            or linear_sequence_length is None
            or preparation_sharing.dimension_key(linear_sequence_length)
            != preparation_sharing.dimension_key(sequence_length)
            or _static_int(linear_shape[2]) != output_features
            or logical_dtype is not torch.bfloat16
        ):
            return False

    for name in ("sparse_q_norm_weight", "sparse_k_norm_weight"):
        norm = metadata[name]
        assert norm is not None
        if (
            norm.dtype is not torch.bfloat16
            or tuple(norm.shape) != (_HEAD_DIM,)
            or norm.device != q_input.device
        ):
            return False
    if any(
        _positive_float(match.kwargs[name]) is None
        for name in (
            "sparse_q_norm_epsilon",
            "sparse_k_norm_epsilon",
            "sparse_softmax_scale",
        )
    ):
        return False

    rotary_dim = _integer_scalar_metadata(match.kwargs["sparse_rotary_dim"])
    half_rotary_dim = _integer_scalar_metadata(match.kwargs["sparse_half_rotary_dim"])
    cos = metadata["sparse_cos"]
    sin = metadata["sparse_sin"]
    assert cos is not None
    assert sin is not None
    if (
        not isinstance(rotary_dim, int)
        or not isinstance(half_rotary_dim, int)
        or rotary_dim < 2
        or rotary_dim > _HEAD_DIM
        or rotary_dim % 2
        or half_rotary_dim != rotary_dim // 2
        or cos.dtype is not torch.float32
        or sin.dtype is not torch.float32
        or tuple(cos.shape) != (sequence_length, rotary_dim)
        or tuple(sin.shape) != (sequence_length, rotary_dim)
        or cos.device != q_input.device
        or sin.device != q_input.device
    ):
        return False

    sparse_key_blocks = _integer_scalar_argument(match.kwargs["sparse_key_blocks"])
    static_sparse_key_blocks = _static_int(sparse_key_blocks)
    if (
        sparse_key_blocks is None
        or (static_sparse_key_blocks is not None and static_sparse_key_blocks < 1)
        or (
            static_sparse_key_blocks is not None
            and isinstance(sequence_length, int)
            and static_sparse_key_blocks > sequence_length // _TILE_ROWS
        )
    ):
        return False
    ratios = match.kwargs["sparse_head_keep_ratio_units"]
    return bool(
        isinstance(ratios, (list, tuple))
        and len(ratios) == heads
        and all(
            isinstance(units, int)
            and not isinstance(units, bool)
            and 1 <= units <= _budget._RATIO_SCALE
            for units in ratios
        )
    )


def _replace_projection(  # noqa: PLR0913, PLR0917
    match: Match,
    sparse_q_input_qdata: torch.fx.Node,
    sparse_q_input_scale: torch.fx.Node,
    sparse_q_input_per_tensor_scale: torch.fx.Node,
    sparse_q_weight_qdata: torch.fx.Node,
    sparse_q_weight_scale: torch.fx.Node,
    sparse_q_weight_per_tensor_scale: torch.fx.Node | None,
    sparse_q_bias: torch.fx.Node | None,
    sparse_k_input_qdata: torch.fx.Node,
    sparse_k_input_scale: torch.fx.Node,
    sparse_k_input_per_tensor_scale: torch.fx.Node,
    sparse_k_weight_qdata: torch.fx.Node,
    sparse_k_weight_scale: torch.fx.Node,
    sparse_k_weight_per_tensor_scale: torch.fx.Node | None,
    sparse_k_bias: torch.fx.Node | None,
    sparse_v_input_qdata: torch.fx.Node,
    sparse_v_input_scale: torch.fx.Node,
    sparse_v_input_per_tensor_scale: torch.fx.Node,
    sparse_v_weight_qdata: torch.fx.Node,
    sparse_v_weight_scale: torch.fx.Node,
    sparse_v_weight_per_tensor_scale: torch.fx.Node | None,
    sparse_v_bias: torch.fx.Node | None,
    sparse_q_norm_weight: torch.fx.Node,
    sparse_k_norm_weight: torch.fx.Node,
    sparse_cos: torch.fx.Node,
    sparse_sin: torch.fx.Node,
    sparse_attention_shape: list[Argument],
    sparse_head_keep_ratio_units: list[int],
    sparse_q_norm_epsilon: float,
    sparse_k_norm_epsilon: float,
    sparse_key_blocks: Argument,
    sparse_softmax_scale: float,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    input_value = preparation_sharing.tensor_metadata(sparse_q_input_qdata)
    q_weight_value = preparation_sharing.tensor_metadata(sparse_q_weight_qdata)
    assert input_value is not None
    assert q_weight_value is not None
    sequence_length = input_value.shape[0]
    heads = q_weight_value.shape[0] // _HEAD_DIM
    storage_sequence_length = layout.padded_sequence_length(sequence_length)
    logical_sequence_length = _integer_scalar_argument(sparse_attention_shape[1])
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
        query_nodes = _compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default,
            (
                sparse_q_input_qdata,
                sparse_q_input_scale,
                sparse_q_input_per_tensor_scale,
                sparse_q_weight_qdata,
                sparse_q_weight_scale,
                sparse_q_weight_per_tensor_scale,
                sparse_q_bias,
                sparse_q_norm_weight,
                sparse_cos,
                sparse_sin,
                sparse_q_norm_epsilon,
                sparse_softmax_scale,
                query.DEFAULT_CHUNK_ROWS,
            ),
            query_values,
        )
        key_values = (
            input_value.new_empty((1, heads, storage_sequence_length, _HEAD_DIM), dtype=torch.int8),
            input_value.new_empty(
                (1, heads, storage_sequence_length // _TILE_ROWS), dtype=torch.float32
            ),
            input_value.new_empty(
                (1, heads, storage_sequence_length // _TILE_ROWS, _HEAD_DIM),
                dtype=torch.float32,
            ),
            input_value.new_empty(
                (1, heads, storage_sequence_length // _TILE_ROWS, _HEAD_DIM),
                dtype=torch.float32,
            ),
        )
        key_nodes = _compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default,
            (
                sparse_k_input_qdata,
                sparse_k_input_scale,
                sparse_k_input_per_tensor_scale,
                sparse_k_weight_qdata,
                sparse_k_weight_scale,
                sparse_k_weight_per_tensor_scale,
                sparse_k_bias,
                sparse_k_norm_weight,
                sparse_cos,
                sparse_sin,
                sparse_k_norm_epsilon,
                query.DEFAULT_CHUNK_ROWS,
            ),
            key_values,
        )
        value_mean_flat = graph.call_function(
            torch.ops.piper_kernels.nvfp4_linear_mean.default,
            args=(
                sparse_v_input_qdata,
                sparse_v_input_scale,
                sparse_v_input_per_tensor_scale,
                sparse_v_weight_qdata,
                sparse_v_weight_scale,
                sparse_v_weight_per_tensor_scale,
                sparse_v_bias,
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
        value_nodes = _compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default,
            (
                sparse_v_input_qdata,
                sparse_v_input_scale,
                sparse_v_input_per_tensor_scale,
                sparse_v_weight_qdata,
                sparse_v_weight_scale,
                sparse_v_weight_per_tensor_scale,
                sparse_v_bias,
                value_mean,
                query.DEFAULT_CHUNK_ROWS,
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
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


_patterns = PatternMatcherPass("nvfp4_sparse_piper_projection")
register_graph_pattern(
    projected_qk_pattern.sparse_piper_projection_pattern(_linear_pattern),
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
