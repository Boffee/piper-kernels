"""ConvRot projection folding for sparse Piper attention."""

from __future__ import annotations

import math
import operator
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

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import (
    _quantized_dispatch,
    dispatch,
)
from piper_kernels.fusions.convrot_sage_qk import triton as convrot_sage_qk
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _compile as convrot_compile
from piper_kernels.linear.convrot.int8 import _compile_fx

from . import key, query, value

_COMPILE_PASS_VERSION = "convrot-sparse-piper-compile-v1"
_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"
_HEAD_DIM = 128
_QUERY_BLOCK_ROWS = 64
_QUERY_SCALE_ROWS = 32
_KEY_BLOCK_ROWS = 64
_SLICE_END = torch.iinfo(torch.int64).max


def source_files() -> tuple[str, ...]:
    """Return every source file whose changes invalidate this graph rewrite."""
    return tuple(
        file_name
        for file_name in (
            __file__,
            convrot_sage_qk.__file__,
            query.__file__,
            key.__file__,
            value.__file__,
            _quantized_dispatch.__file__,
            dispatch.__file__,
            _compile_fx.__file__,
        )
        if file_name is not None
    )


def _normalized_rope_pattern(prefix: str) -> CallFunction:
    linear = CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        KeywordArg("sparse_input"),
        KeywordArg(f"{prefix}_weight_qdata"),
        KeywordArg(f"{prefix}_weight_scale"),
        None,
        KeywordArg("sparse_group_size"),
        _users=1,
    )
    reshaped = CallFunction(
        torch.ops.aten.reshape.default,
        linear,
        KeywordArg("sparse_attention_shape"),
        _users=1,
    )
    promoted = CallFunction(
        torch.ops.prims.convert_element_type.default,
        reshaped,
        torch.float32,
        _users=2,
    )
    squared = CallFunction(
        torch.ops.aten.pow.Tensor_Scalar,
        promoted,
        2,
        _users=1,
    )
    mean = CallFunction(
        torch.ops.aten.mean.dim,
        squared,
        [3],
        True,
        _users=1,
    )
    variance = CallFunction(
        torch.ops.aten.add.Scalar,
        mean,
        KeywordArg(f"{prefix}_norm_epsilon"),
        _users=1,
    )
    inverse_rms = CallFunction(torch.ops.aten.rsqrt.default, variance, _users=1)
    normalized = CallFunction(
        torch.ops.aten.mul.Tensor,
        promoted,
        inverse_rms,
        _users=1,
    )
    scaled = CallFunction(
        torch.ops.aten.mul.Tensor,
        normalized,
        KeywordArg(f"{prefix}_norm_weight"),
        _users=1,
    )
    rounded = CallFunction(
        torch.ops.prims.convert_element_type.default,
        scaled,
        torch.bfloat16,
        _users=2,
    )
    rotary = CallFunction(
        torch.ops.aten.slice.Tensor,
        rounded,
        3,
        0,
        KeywordArg("sparse_rotary_dim"),
        _users=2,
    )
    split = CallFunction(
        torch.ops.aten.split.Tensor,
        rotary,
        KeywordArg("sparse_half_rotary_dim"),
        -1,
        _users=2,
    )
    first = CallFunction(operator.getitem, split, 0, _users=1)
    second = CallFunction(operator.getitem, split, 1, _users=1)
    cos = CallFunction(
        torch.ops.prims.convert_element_type.default,
        KeywordArg("sparse_cos"),
        torch.bfloat16,
        _users=1,
    )
    cos = CallFunction(torch.ops.aten.unsqueeze.default, cos, 0, _users=1)
    cos = CallFunction(torch.ops.aten.unsqueeze.default, cos, 2, _users=1)
    direct = CallFunction(torch.ops.aten.mul.Tensor, rotary, cos, _users=1)
    negated_second = CallFunction(torch.ops.aten.neg.default, second, _users=1)
    rotated = CallFunction(
        torch.ops.aten.cat.default,
        [negated_second, first],
        -1,
        _users=1,
    )
    sin = CallFunction(
        torch.ops.prims.convert_element_type.default,
        KeywordArg("sparse_sin"),
        torch.bfloat16,
        _users=1,
    )
    sin = CallFunction(torch.ops.aten.unsqueeze.default, sin, 0, _users=1)
    sin = CallFunction(torch.ops.aten.unsqueeze.default, sin, 2, _users=1)
    rotated = CallFunction(torch.ops.aten.mul.Tensor, rotated, sin, _users=1)
    rotary_output = CallFunction(torch.ops.aten.add.Tensor, direct, rotated, _users=1)
    passthrough = CallFunction(
        torch.ops.aten.slice.Tensor,
        rounded,
        3,
        KeywordArg("sparse_rotary_dim"),
        _SLICE_END,
        _users=1,
    )
    return CallFunction(
        torch.ops.aten.cat.default,
        [rotary_output, passthrough],
        -1,
        _users=1,
    )


def _sparse_piper_projection_pattern() -> CallFunction:
    value = CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        KeywordArg("sparse_input"),
        KeywordArg("sparse_v_weight_qdata"),
        KeywordArg("sparse_v_weight_scale"),
        None,
        KeywordArg("sparse_group_size"),
        _users=1,
    )
    value = CallFunction(
        torch.ops.aten.reshape.default,
        value,
        KeywordArg("sparse_attention_shape"),
        _users=1,
    )
    return CallFunction(
        torch.ops.piper_kernels.sparse_piper_attention.default,
        _normalized_rope_pattern("sparse_q"),
        _normalized_rope_pattern("sparse_k"),
        value,
        KeywordArg("sparse_keep_blocks"),
        KeywordArg("sparse_head_offsets"),
        KeywordArg("sparse_suffix_start"),
        KeywordArg("sparse_valid_sequence_length"),
        KeywordArg("sparse_softmax_scale"),
        KeywordArg("sparse_routes_per_query"),
        KeywordArg("sparse_query_chunk_blocks"),
    )


def _static_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted > 0 else None


def _valid_sparse_piper_projection(match: Match) -> bool:  # noqa: PLR0911, PLR0912
    nodes = (
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
        "sparse_keep_blocks",
        "sparse_head_offsets",
    )
    if any(not isinstance(match.kwargs[name], torch.fx.Node) for name in nodes):
        return False
    metadata = {
        name: preparation_sharing.tensor_metadata(match.kwargs[name])  # type: ignore[arg-type]
        for name in nodes
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
    shape = match.kwargs["sparse_attention_shape"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return False
    dimensions = tuple(_static_int(value) for value in shape)
    if any(value is None for value in dimensions):
        return False
    batch, sequence_length, heads, head_dim = dimensions
    assert batch is not None
    assert sequence_length is not None
    assert heads is not None
    assert head_dim is not None
    if (
        input_value.ndim != 3
        or input_value.dtype is not torch.bfloat16
        or input_value.device.type != "cuda"
        or tuple(input_value.shape[:2]) != (batch, sequence_length)
        or head_dim != _HEAD_DIM
        or sequence_length < _QUERY_BLOCK_ROWS
        or sequence_length % _QUERY_BLOCK_ROWS
    ):
        return False
    target = AcceleratorTarget.from_device(input_value.device)
    if not target.is_cuda_capability(12, 0):
        return False

    input_features = _static_int(input_value.shape[-1])
    group_size = _static_int(match.kwargs["sparse_group_size"])
    if input_features is None or group_size is None or group_size < 1:
        return False
    for prefix in ("sparse_q", "sparse_k", "sparse_v"):
        weight = metadata[f"{prefix}_weight_qdata"]
        scale = metadata[f"{prefix}_weight_scale"]
        assert weight is not None
        assert scale is not None
        if (
            weight.dtype is not torch.int8
            or tuple(weight.shape) != (heads * head_dim, input_features)
            or scale.dtype is not torch.float32
            or tuple(scale.shape) != (heads * head_dim, 1)
            or weight.device != input_value.device
            or scale.device != input_value.device
        ):
            return False

    for name in ("sparse_q_norm_weight", "sparse_k_norm_weight"):
        norm = metadata[name]
        assert norm is not None
        if (
            norm.dtype is not torch.bfloat16
            or tuple(norm.shape) != (head_dim,)
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

    rotary_dim = _static_int(match.kwargs["sparse_rotary_dim"])
    half_rotary_dim = _static_int(match.kwargs["sparse_half_rotary_dim"])
    cos = metadata["sparse_cos"]
    sin = metadata["sparse_sin"]
    assert cos is not None
    assert sin is not None
    if (
        rotary_dim is None
        or half_rotary_dim is None
        or rotary_dim < 2
        or rotary_dim > head_dim
        or rotary_dim % 2
        or half_rotary_dim != rotary_dim // 2
        or cos.dtype is not torch.float32
        or sin.dtype is not torch.float32
        or tuple(cos.shape) != (sequence_length, rotary_dim)
        or sin.shape != cos.shape
        or cos.device != input_value.device
        or sin.device != input_value.device
    ):
        return False

    suffix_start = _static_int(match.kwargs["sparse_suffix_start"])
    valid_sequence_length = _static_int(match.kwargs["sparse_valid_sequence_length"])
    routes_per_query = _static_int(match.kwargs["sparse_routes_per_query"])
    query_chunk_blocks = _static_int(match.kwargs["sparse_query_chunk_blocks"])
    if (
        suffix_start is None
        or valid_sequence_length is None
        or routes_per_query is None
        or query_chunk_blocks is None
        or not _KEY_BLOCK_ROWS <= suffix_start <= valid_sequence_length <= sequence_length
        or suffix_start % _KEY_BLOCK_ROWS
        or valid_sequence_length <= sequence_length - _KEY_BLOCK_ROWS
        or routes_per_query < 1
        or query_chunk_blocks < 1
    ):
        return False
    keep_blocks = metadata["sparse_keep_blocks"]
    head_offsets = metadata["sparse_head_offsets"]
    assert keep_blocks is not None
    assert head_offsets is not None
    return (
        keep_blocks.dtype is torch.int32
        and tuple(keep_blocks.shape) == (heads,)
        and head_offsets.dtype is torch.int32
        and tuple(head_offsets.shape) == (heads + 1,)
        and keep_blocks.device == input_value.device
        and head_offsets.device == input_value.device
    )


def _replace_sparse_piper_projection(  # noqa: PLR0913, PLR0917
    match: Match,
    sparse_input: torch.fx.Node,
    sparse_q_weight_qdata: torch.fx.Node,
    sparse_q_weight_scale: torch.fx.Node,
    sparse_k_weight_qdata: torch.fx.Node,
    sparse_k_weight_scale: torch.fx.Node,
    sparse_v_weight_qdata: torch.fx.Node,
    sparse_v_weight_scale: torch.fx.Node,
    sparse_q_norm_weight: torch.fx.Node,
    sparse_k_norm_weight: torch.fx.Node,
    sparse_cos: torch.fx.Node,
    sparse_sin: torch.fx.Node,
    sparse_keep_blocks: torch.fx.Node,
    sparse_head_offsets: torch.fx.Node,
    sparse_group_size: int,
    sparse_attention_shape: list[int] | tuple[int, ...],
    sparse_q_norm_epsilon: float,
    sparse_k_norm_epsilon: float,
    sparse_suffix_start: int,
    sparse_valid_sequence_length: int,
    sparse_softmax_scale: float,
    sparse_routes_per_query: int,
    sparse_query_chunk_blocks: int,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    input_value = preparation_sharing.tensor_metadata(sparse_input)
    assert input_value is not None
    batch, sequence_length, heads, head_dim = sparse_attention_shape
    with graph.inserting_before(original):
        input_qdata, input_scale, _logical_dtype = _compile_fx.emit_prepared_input(
            graph,
            sparse_input,
            sparse_group_size,
            None,
            tuple(input_value.shape),
        )
        query_values = (
            input_value.new_empty(
                (batch, heads, sequence_length, head_dim),
                dtype=torch.int8,
            ),
            input_value.new_empty(
                (batch, heads, sequence_length // _QUERY_SCALE_ROWS),
                dtype=torch.float32,
            ),
            input_value.new_empty(
                (batch, heads, sequence_length // _QUERY_BLOCK_ROWS, head_dim),
                dtype=torch.float32,
            ),
        )
        query, query_scale, query_summary = _compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_sparse_piper_project_query.default,
            (
                input_qdata,
                input_scale,
                sparse_q_weight_qdata,
                sparse_q_weight_scale,
                sparse_q_norm_weight,
                sparse_cos,
                sparse_sin,
                sparse_valid_sequence_length,
                sparse_q_norm_epsilon,
                sparse_softmax_scale,
            ),
            query_values,
        )
        key_values = (
            input_value.new_empty(
                (batch, heads, sequence_length, head_dim),
                dtype=torch.int8,
            ),
            input_value.new_empty(
                (batch, heads, sequence_length // _KEY_BLOCK_ROWS),
                dtype=torch.float32,
            ),
            input_value.new_empty(
                (batch, heads, sequence_length // _KEY_BLOCK_ROWS, head_dim),
                dtype=torch.float32,
            ),
            input_value.new_empty(
                (batch, heads, sequence_length // _KEY_BLOCK_ROWS, head_dim),
                dtype=torch.float32,
            ),
        )
        key, key_scale, key_max, key_min = _compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_sparse_piper_project_key.default,
            (
                input_qdata,
                input_scale,
                sparse_k_weight_qdata,
                sparse_k_weight_scale,
                sparse_k_norm_weight,
                sparse_cos,
                sparse_sin,
                sparse_valid_sequence_length,
                sparse_k_norm_epsilon,
            ),
            key_values,
        )
        input_mean = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_dequantized_input_mean.default,
            args=(input_qdata, input_scale, sparse_valid_sequence_length),
        )
        input_mean.meta["val"] = input_value.new_empty(
            (batch, input_value.shape[-1]),
            dtype=torch.float32,
        )
        value_values = (
            input_value.new_empty(
                (batch, heads, head_dim, sequence_length),
                dtype=torch.int8,
            ),
            input_value.new_empty(
                (batch, heads, sequence_length // _KEY_BLOCK_ROWS, 1),
                dtype=torch.float32,
            ),
            input_value.new_empty((batch, heads, head_dim), dtype=torch.float32),
        )
        value, value_scale, value_mean = _compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_sparse_piper_project_value.default,
            (
                input_qdata,
                input_scale,
                input_mean,
                sparse_v_weight_qdata,
                sparse_v_weight_scale,
                sparse_valid_sequence_length,
            ),
            value_values,
        )
        replacement = graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
            args=(
                query,
                query_scale,
                query_summary,
                key,
                key_scale,
                key_max,
                key_min,
                value,
                value_scale,
                value_mean,
                sparse_keep_blocks,
                sparse_head_offsets,
                sparse_suffix_start,
                sparse_valid_sequence_length,
                sparse_routes_per_query,
                sparse_query_chunk_blocks,
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


_patterns = PatternMatcherPass("convrot_sparse_piper_projection")
register_graph_pattern(
    _sparse_piper_projection_pattern(),
    extra_check=_valid_sparse_piper_projection,
    pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
)(_replace_sparse_piper_projection)


def fold_sparse_piper_projection(graph: torch.fx.Graph) -> bool:
    """Replace one compatible materialized sparse-attention projection region."""
    changed = _patterns.apply(graph) > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold compatible ConvRot projections into sparse Piper preparation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            fold_sparse_piper_projection(graph)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            source_files(),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def _without_compiler_pass(
    options: Mapping[str, object] | None,
    compiler_pass: object,
) -> dict[str, object]:
    """Copy options while removing one pass by identity."""
    combined = dict(options) if options is not None else {}
    existing = combined.get(_POST_GRAD_PRE_PASS)
    if existing is compiler_pass:
        combined.pop(_POST_GRAD_PRE_PASS)
    elif isinstance(existing, (list, tuple)):
        remaining = tuple(item for item in existing if item is not compiler_pass)
        if not remaining:
            combined.pop(_POST_GRAD_PRE_PASS)
        elif len(remaining) == 1:
            combined[_POST_GRAD_PRE_PASS] = remaining[0]
        else:
            combined[_POST_GRAD_PRE_PASS] = remaining
    return combined


def convrot_sparse_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install sparse fusion before ordinary ConvRot graph optimizations."""
    combined = _without_compiler_pass(options, compile_pass)
    combined = _without_compiler_pass(combined, convrot_compile.compile_pass)
    combined = preparation_sharing.add_post_grad_pass(combined, compile_pass)
    return preparation_sharing.add_post_grad_pass(combined, convrot_compile.compile_pass)


__all__ = ["convrot_sparse_piper_compile_options"]
