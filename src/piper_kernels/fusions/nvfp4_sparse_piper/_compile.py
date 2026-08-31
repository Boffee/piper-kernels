"""NVFP4 projection folding for sparse Piper attention."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

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

from piper_kernels._triton.targets import AcceleratorTarget
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

_COMPILE_PASS_VERSION = "nvfp4-sparse-piper-compile-v1"
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
    return CallFunction(
        torch.ops.piper_kernels.nvfp4_linear.default,
        KeywordArg("sparse_input"),
        KeywordArg(f"{prefix}_weight_qdata"),
        KeywordArg(f"{prefix}_weight_scale"),
        KeywordArg(f"{prefix}_weight_per_tensor_scale"),
        KeywordArg(f"{prefix}_activation_per_tensor_scale"),
        KeywordArg(f"{prefix}_bias"),
        KeywordArg(f"{prefix}_dynamic_activation_scale"),
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
    required_nodes = (
        "sparse_input",
        "sparse_q_weight_qdata",
        "sparse_q_weight_scale",
        "sparse_k_weight_qdata",
        "sparse_k_weight_scale",
        "sparse_v_weight_qdata",
        "sparse_v_weight_scale",
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

    input_value = metadata["sparse_input"]
    assert input_value is not None
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    shape = match.kwargs["sparse_attention_shape"]
    if (
        input_value.ndim != 3
        or output_value is None
        or output_value.ndim != 4
        or not isinstance(shape, (list, tuple))
        or len(shape) != 4
    ):
        return False
    batch = _static_int(input_value.shape[0])
    sequence_length = input_value.shape[1]
    input_features = _static_int(input_value.shape[2])
    q_weight = metadata["sparse_q_weight_qdata"]
    assert q_weight is not None
    output_features = _static_int(q_weight.shape[0])
    if (
        batch != 1
        or input_features is None
        or input_features % 16
        or output_features is None
        or output_features % _HEAD_DIM
    ):
        return False
    heads = output_features // _HEAD_DIM
    output_shape = output_value.shape
    if (
        input_value.dtype is not torch.bfloat16
        or input_value.device.type != "cuda"
        or output_value.dtype is not torch.bfloat16
        or output_value.device != input_value.device
        or tuple(output_shape[2:]) != (heads, _HEAD_DIM)
        or preparation_sharing.dimension_key(output_shape[0])
        != preparation_sharing.dimension_key(input_value.shape[0])
        or preparation_sharing.dimension_key(output_shape[1])
        != preparation_sharing.dimension_key(sequence_length)
        or _static_int(shape[2]) != heads
        or _static_int(shape[3]) != _HEAD_DIM
        or (isinstance(sequence_length, int) and sequence_length < _TILE_ROWS)
        or not AcceleratorTarget.from_device(input_value.device).is_cuda_capability(12, 0)
    ):
        return False

    expected_scale_shape = (
        (output_features + 127) // 128 * 32,
        (input_features + 63) // 64 * 16,
    )
    for prefix in ("sparse_q", "sparse_k", "sparse_v"):
        weight = metadata[f"{prefix}_weight_qdata"]
        scale = metadata[f"{prefix}_weight_scale"]
        global_scale_valid, global_scale = _optional_tensor_metadata(
            match.kwargs[f"{prefix}_weight_per_tensor_scale"]
        )
        bias_valid, bias = _optional_tensor_metadata(match.kwargs[f"{prefix}_bias"])
        activation_scale_valid, activation_scale = _optional_tensor_metadata(
            match.kwargs[f"{prefix}_activation_per_tensor_scale"]
        )
        dynamic_activation_scale = match.kwargs[f"{prefix}_dynamic_activation_scale"]
        assert weight is not None
        assert scale is not None
        if (
            weight.dtype is not torch.uint8
            or tuple(weight.shape) != (output_features, input_features // 2)
            or scale.dtype is not torch.float8_e4m3fn
            or tuple(scale.shape) != expected_scale_shape
            or weight.device != input_value.device
            or scale.device != input_value.device
            or not global_scale_valid
            or not bias_valid
            or not activation_scale_valid
            or not isinstance(dynamic_activation_scale, bool)
            or (activation_scale is None and not dynamic_activation_scale)
        ):
            return False
        if activation_scale is not None:
            activation_scale_value = cast(torch.Tensor, activation_scale)
            if (
                activation_scale_value.shape != ()
                or activation_scale_value.dtype is not torch.float32
                or activation_scale_value.device != input_value.device
            ):
                return False
        if global_scale is not None:
            global_scale_value = cast(torch.Tensor, global_scale)
            if (
                global_scale_value.shape != ()
                or global_scale_value.dtype is not torch.float32
                or global_scale_value.device != input_value.device
            ):
                return False
        if bias is not None:
            bias_value = cast(torch.Tensor, bias)
            if (
                bias_value.shape != (output_features,)
                or bias_value.dtype is not torch.bfloat16
                or bias_value.device != input_value.device
            ):
                return False

    for name in ("sparse_q_norm_weight", "sparse_k_norm_weight"):
        norm = metadata[name]
        assert norm is not None
        if (
            norm.dtype is not torch.bfloat16
            or tuple(norm.shape) != (_HEAD_DIM,)
            or norm.device != input_value.device
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
        or cos.device != input_value.device
        or sin.device != input_value.device
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
    sparse_input: torch.fx.Node,
    sparse_q_weight_qdata: torch.fx.Node,
    sparse_q_weight_scale: torch.fx.Node,
    sparse_q_weight_per_tensor_scale: torch.fx.Node | None,
    sparse_q_activation_per_tensor_scale: torch.fx.Node | None,
    sparse_q_bias: torch.fx.Node | None,
    sparse_q_dynamic_activation_scale: bool,
    sparse_k_weight_qdata: torch.fx.Node,
    sparse_k_weight_scale: torch.fx.Node,
    sparse_k_weight_per_tensor_scale: torch.fx.Node | None,
    sparse_k_activation_per_tensor_scale: torch.fx.Node | None,
    sparse_k_bias: torch.fx.Node | None,
    sparse_k_dynamic_activation_scale: bool,
    sparse_v_weight_qdata: torch.fx.Node,
    sparse_v_weight_scale: torch.fx.Node,
    sparse_v_weight_per_tensor_scale: torch.fx.Node | None,
    sparse_v_activation_per_tensor_scale: torch.fx.Node | None,
    sparse_v_bias: torch.fx.Node | None,
    sparse_v_dynamic_activation_scale: bool,
    sparse_q_norm_weight: torch.fx.Node,
    sparse_k_norm_weight: torch.fx.Node,
    sparse_cos: torch.fx.Node,
    sparse_sin: torch.fx.Node,
    sparse_head_keep_ratio_units: list[int],
    sparse_q_norm_epsilon: float,
    sparse_k_norm_epsilon: float,
    sparse_key_blocks: Argument,
    sparse_softmax_scale: float,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    input_value = preparation_sharing.tensor_metadata(sparse_input)
    q_weight_value = preparation_sharing.tensor_metadata(sparse_q_weight_qdata)
    assert input_value is not None
    assert q_weight_value is not None
    sequence_length = input_value.shape[1]
    heads = q_weight_value.shape[0] // _HEAD_DIM
    storage_sequence_length = layout.padded_sequence_length(sequence_length)
    with graph.inserting_before(original):
        logical_sequence_length = graph.call_function(
            torch.ops.aten.sym_size.int,
            args=(sparse_input, 1),
        )
        logical_sequence_length.meta["val"] = sequence_length
        prepared_inputs: dict[
            tuple[torch.fx.Node | None, bool], _compile_fx.PreparedInputNodes
        ] = {}

        def prepare(
            activation_scale: torch.fx.Node | None,
            dynamic_activation_scale: bool,
        ) -> _compile_fx.PreparedInputNodes:
            preparation_key = (activation_scale, dynamic_activation_scale)
            prepared = prepared_inputs.get(preparation_key)
            if prepared is None:
                prepared = _compile_fx.emit_prepared_input(
                    graph,
                    sparse_input,
                    activation_scale,
                    dynamic_activation_scale,
                )
                prepared_inputs[preparation_key] = prepared
            return prepared

        q_input_qdata, q_input_scale, q_input_global_scale, _q_shape = prepare(
            sparse_q_activation_per_tensor_scale,
            sparse_q_dynamic_activation_scale,
        )
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
                q_input_qdata,
                q_input_scale,
                q_input_global_scale,
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
        k_input_qdata, k_input_scale, k_input_global_scale, _k_shape = prepare(
            sparse_k_activation_per_tensor_scale,
            sparse_k_dynamic_activation_scale,
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
                k_input_qdata,
                k_input_scale,
                k_input_global_scale,
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
        v_input_qdata, v_input_scale, v_input_global_scale, _v_shape = prepare(
            sparse_v_activation_per_tensor_scale,
            sparse_v_dynamic_activation_scale,
        )
        value_mean_flat = graph.call_function(
            torch.ops.piper_kernels.nvfp4_linear_mean.default,
            args=(
                v_input_qdata,
                v_input_scale,
                v_input_global_scale,
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
                v_input_qdata,
                v_input_scale,
                v_input_global_scale,
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
    """Install sparse projection fusion before ordinary NVFP4 preparation sharing."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (compile_pass, nvfp4_compile.compile_pass),
    )


__all__ = ["nvfp4_sparse_piper_compile_options"]
