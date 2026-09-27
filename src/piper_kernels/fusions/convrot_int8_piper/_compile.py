"""Fold ConvRot INT8 Q/K/V projections into dense Piper operand preparation."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from itertools import product

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass, get_hash_for_files
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true

from piper_kernels.attention.piper_attention import _quantized_dispatch
from piper_kernels.fusions.attention import _output as output_pipeline
from piper_kernels.fusions.convrot_int8_projection import output as projection_output
from piper_kernels.fusions.projected_qk import _compile as projected_compile
from piper_kernels.fusions.projected_qk import _pattern as projected_pattern
from piper_kernels.linear import _bias, _projection_views
from piper_kernels.linear import _compile_fx as linear_compile_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _compile as linear_compile
from piper_kernels.linear.convrot.int8 import _compile_fx, _ops
from piper_kernels.weights.convrot import _rotation

from . import _backend, _output_compile, _schedule, _validation, output, query
from . import key as key_projection
from . import value as value_projection

_COMPILE_PASS_VERSION = "convrot-int8-piper-output-compile-v7"
_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16)


def _source_files() -> tuple[str, ...]:
    """Invalidate cached rewrites when their metadata or execution contracts change."""
    return tuple(
        path
        for path in (
            __file__,
            _output_compile.__file__,
            _schedule.__file__,
            output.__file__,
            output_pipeline.__file__,
            projection_output.__file__,
            _validation.__file__,
            *_backend.source_files(),
            *_quantized_dispatch.source_files(),
            query.__file__,
            key_projection.__file__,
            value_projection.__file__,
            projected_compile.__file__,
            projected_pattern.__file__,
            _bias.__file__,
            _projection_views.__file__,
            linear_compile_fx.__file__,
            preparation_sharing.__file__,
            _compile_fx.__file__,
            _ops.__file__,
            _rotation.__file__,
        )
        if path is not None
    )


def _projected_pattern(
    dtype: torch.dtype | None,
    affine: bool,
    permute: bool,
    full_rotary: bool = False,
) -> CallFunction:
    """Match a BHSD projection, including RMSNorm/RoPE when a dtype is supplied."""
    projection = CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        KeywordArg("projection_input"),
        KeywordArg("weight_qdata"),
        KeywordArg("weight_scale"),
        KeywordArg("bias"),
        KeywordArg("group_size"),
        None,
        KeywordArg("input_scale"),
        _users=1,
    )
    if dtype is not None:
        transformed = projected_pattern.normalized_rope_pattern(
            projection,
            shape_name="projection_shape",
            norm_weight_name="norm_weight",
            norm_epsilon_name="norm_epsilon",
            cos_name="cos",
            sin_name="sin",
            rotary_dim_name="rotary_dim",
            half_rotary_dim_name="half_rotary_dim",
            activation_dtype=dtype,
            affine=affine,
            full_rotary=full_rotary,
        )
    else:
        transformed = CallFunction(
            torch.ops.aten.reshape.default, projection, KeywordArg("projection_shape"), _users=1
        )
    return (
        CallFunction(torch.ops.aten.permute.default, transformed, [0, 2, 1, 3], _users=1)
        if permute
        else CallFunction(torch.ops.aten.transpose.int, transformed, 1, 2, _users=1)
    )


def _projection_pattern(
    dtype: torch.dtype, affine: bool, permute: bool, full_rotary: bool = False
) -> CallFunction:
    transposed = _projected_pattern(dtype, affine, permute, full_rotary)
    return CallFunction(
        torch.ops.piper_kernels.piper_attention.default,
        transposed,
        KeywordArg("key"),
        KeywordArg("value"),
        KeywordArg("softmax_scale"),
        KeywordArg("is_causal"),
    )


def _same_shape(left: Sequence[int | torch.SymInt], right: Sequence[int | torch.SymInt]) -> bool:
    return len(left) == len(right) and all(
        statically_known_true(a == b) for a, b in zip(left, right, strict=True)
    )


def _valid_projected_tensor(  # noqa: PLR0911
    match: Match, output_node: torch.fx.Node, *, normalized: bool = True
) -> bool:
    names = ("projection_input", "weight_qdata", "weight_scale")
    if any(not isinstance(match.kwargs[name], torch.fx.Node) for name in names):
        return False
    metadata = {
        name: preparation_sharing.tensor_metadata(match.kwargs[name])  # type: ignore[arg-type]
        for name in names
    }
    input_value = metadata["projection_input"]
    weight = metadata["weight_qdata"]
    weight_scale = metadata["weight_scale"]
    if input_value is None or weight is None or weight_scale is None:
        return False
    output_value = preparation_sharing.tensor_metadata(output_node)
    if (
        output_value is None
        or output_value.ndim != 4
        or output_value.layout is not torch.strided
        or output_value.stride(-1) != 1
        or output_value.dtype is not input_value.dtype
        or input_value.ndim != 3
        or input_value.dtype not in _ACTIVATION_DTYPES
        or any(
            tensor.layout is not torch.strided or not tensor.is_contiguous()
            for tensor in (input_value, weight, weight_scale)
        )
    ):
        return False
    head_dim = projected_compile.static_int(output_value.shape[-1])
    input_features = projected_compile.static_int(weight.shape[1]) if weight.ndim == 2 else None
    output_features = projected_compile.static_int(weight.shape[0]) if weight.ndim == 2 else None
    group_size = projected_compile.static_int(match.kwargs["group_size"])
    if (
        head_dim not in (64, 128)
        or input_features is None
        or input_features < 1
        or output_features is None
        or output_features < 1
        or output_features % head_dim
        or group_size not in _rotation.SUPPORTED_GROUP_SIZES
        or input_features % group_size
        or weight.dtype is not torch.int8
        or input_value.shape[-1] != input_features
        or weight_scale.dtype is not torch.float32
        or tuple(weight_scale.shape) != (output_features, 1)
        or not _compile_fx.valid_input_scale(match.kwargs["input_scale"], input_value.device)
    ):
        return False
    heads = output_features // head_dim
    batch, sequence_length = input_value.shape[:2]
    if not _same_shape(output_value.shape, (batch, heads, sequence_length, head_dim)):
        return False
    if isinstance(sequence_length, int) and sequence_length < 1:
        return False
    if isinstance(batch, int) and batch < 1:
        return False
    if any(tensor.device != input_value.device for tensor in (weight, weight_scale, output_value)):
        return False
    bias_node = match.kwargs["bias"]
    if bias_node is not None:
        if not isinstance(bias_node, torch.fx.Node):
            return False
        bias = preparation_sharing.tensor_metadata(bias_node)
        if (
            bias is None
            or tuple(bias.shape) != (output_features,)
            or not _bias.is_supported_dtype(bias.dtype)
            or bias.device != input_value.device
            or bias.layout is not torch.strided
            or not bias.is_contiguous()
        ):
            return False
    if not normalized:
        return _backend.select_projection_backend(input_value, head_dim=head_dim) is not None
    return (
        projected_compile.valid_rmsnorm(
            match.kwargs.get("norm_weight"),
            match.kwargs["norm_epsilon"],
            head_dim=head_dim,
            device=input_value.device,
            supported_dtypes=(*_ACTIVATION_DTYPES, torch.float32),
        )
        and projected_compile.valid_rope_tables(
            match.kwargs["cos"],
            match.kwargs["sin"],
            match.kwargs.get("rotary_dim", head_dim),
            match.kwargs["half_rotary_dim"],
            sequence_length=sequence_length,
            head_dim=head_dim,
            device=input_value.device,
        )
        and _backend.select_projection_backend(input_value, head_dim=head_dim) is not None
    )


def _valid_projection(match: Match) -> bool:
    query_node = match.output_node().args[0]
    if not isinstance(query_node, torch.fx.Node) or not _valid_projected_tensor(match, query_node):
        return False
    query_value = preparation_sharing.tensor_metadata(query_node)
    key_node, value_node = match.kwargs["key"], match.kwargs["value"]
    if not isinstance(key_node, torch.fx.Node) or not isinstance(value_node, torch.fx.Node):
        return False
    key = preparation_sharing.tensor_metadata(key_node)
    value = preparation_sharing.tensor_metadata(value_node)
    assert query_value is not None
    if key is None or value is None:
        return False
    batch, heads, sequence_length, head_dim = query_value.shape
    if any(
        tensor.ndim != 4
        or tensor.layout is not torch.strided
        or tensor.dtype is not query_value.dtype
        or tensor.device != query_value.device
        or tensor.stride(-1) != 1
        for tensor in (key, value)
    ):
        return False
    return not (
        not _same_shape(key.shape, value.shape)
        or not _same_shape((key.shape[0], key.shape[3]), (batch, head_dim))
        or not statically_known_true(key.shape[1] > 0)
        or heads % key.shape[1] != 0
        or not statically_known_true(key.shape[2] > 0)
        or not isinstance(match.kwargs["is_causal"], bool)
        or (match.kwargs["is_causal"] and not _same_shape((sequence_length,), (key.shape[2],)))
        or projected_compile.positive_float(match.kwargs["softmax_scale"]) is None
    )


def _replace_projection(  # noqa: PLR0913, PLR0917
    match: Match,
    projection_input: torch.fx.Node,
    weight_qdata: torch.fx.Node,
    weight_scale: torch.fx.Node,
    bias: torch.fx.Node | None,
    group_size: int,
    input_scale: torch.fx.Node | None,
    norm_epsilon: float,
    cos: torch.fx.Node,
    sin: torch.fx.Node,
    key: torch.fx.Node,
    value: torch.fx.Node,
    softmax_scale: float,
    is_causal: bool,
    norm_weight: torch.fx.Node | None = None,
    **_unused: object,
) -> None:
    graph = match.graph
    original = match.output_node()
    input_value = preparation_sharing.tensor_metadata(projection_input)
    query_node = original.args[0]
    assert input_value is not None
    assert isinstance(query_node, torch.fx.Node)
    query_value = preparation_sharing.tensor_metadata(query_node)
    assert query_value is not None
    batch, heads, sequence_length, head_dim = query_value.shape
    storage_rows = (sequence_length + 63) // 64 * 64
    with graph.inserting_before(original):
        query_length = graph.call_function(torch.ops.aten.sym_size.int, args=(projection_input, 1))
        query_length.meta["val"] = sequence_length
        input_qdata, scales, _ = _compile_fx.emit_prepared_input(
            graph, projection_input, group_size, None, tuple(input_value.shape), input_scale
        )
        query_data, query_scale = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_int8_piper_project_query.default,
            (
                input_qdata,
                scales,
                weight_qdata,
                weight_scale,
                norm_weight,
                cos,
                sin,
                norm_epsilon,
                softmax_scale,
                bias,
            ),
            (
                input_value.new_empty((batch, heads, storage_rows, head_dim), dtype=torch.int8),
                input_value.new_empty((batch, heads, storage_rows // 32), dtype=torch.float32),
            ),
            kwargs={"head_dim": head_dim},
        )
        replacement = graph.call_function(
            torch.ops.piper_kernels.piper_attention_from_quantized_query.default,
            args=(query_data, query_scale, key, value, query_length, is_causal),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


_patterns = PatternMatcherPass("convrot_int8_piper_query")
for _dtype, _affine, _permute, _full_rotary in product(
    _ACTIVATION_DTYPES, (False, True), (False, True), (False, True)
):
    register_graph_pattern(
        _projection_pattern(_dtype, _affine, _permute, _full_rotary),
        extra_check=_valid_projection,
        pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
    )(_replace_projection)


_key_patterns = tuple(
    _projected_pattern(dtype, affine, permute, full_rotary)
    for dtype, affine, permute, full_rotary in product(
        _ACTIVATION_DTYPES, (False, True), (False, True), (False, True)
    )
)
_value_patterns = tuple(_projected_pattern(None, False, permute) for permute in (False, True))


def _context_match(node: torch.fx.Node, *, normalized: bool) -> Match | None:
    if len(node.users) != 1:
        return None
    for pattern in _key_patterns if normalized else _value_patterns:
        match = pattern.match(node)
        if isinstance(match, Match) and _valid_projected_tensor(match, node, normalized=normalized):
            return match
    return None


def _prepared_context_input(
    graph: torch.fx.Graph, arguments: dict, before: torch.fx.Node
) -> tuple[torch.fx.Node, torch.fx.Node]:
    input_node = arguments["projection_input"]
    preparation_args = (input_node, arguments["group_size"], None, arguments["input_scale"])
    prepared: dict[torch.fx.Node, dict[int, torch.fx.Node]] = {}
    for node in graph.nodes:
        if node is before:
            break
        if node.op != "call_function":
            continue
        if (
            node.target is torch.ops.piper_kernels.convrot_int8_prepare_input.default
            and node.args == preparation_args
        ):
            prepared[node] = {}
        elif (
            node.target is operator.getitem
            and isinstance(node.args[0], torch.fx.Node)
            and node.args[0] in prepared
            and isinstance(node.args[1], int)
        ):
            # The tuple and both elements must precede this attention. Its
            # other tuple users can appear later in the graph.
            parts = prepared[node.args[0]]
            parts[node.args[1]] = node
            if 0 in parts and 1 in parts:
                return parts[0], parts[1]
    input_value = preparation_sharing.tensor_metadata(input_node)
    assert input_value is not None
    data, scales, _ = _compile_fx.emit_prepared_input(
        graph,
        input_node,
        arguments["group_size"],
        None,
        tuple(input_value.shape),
        arguments["input_scale"],
    )
    return data, scales


def _emit_context_projection(
    graph: torch.fx.Graph, match: Match, *, normalized: bool, is_causal: bool, before: torch.fx.Node
) -> tuple[torch.fx.Node, ...]:
    arguments = match.kwargs
    input_node = arguments["projection_input"]
    assert isinstance(input_node, torch.fx.Node)
    input_value = preparation_sharing.tensor_metadata(input_node)
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    assert input_value is not None
    assert output_value is not None
    batch, heads, sequence, head_dim = output_value.shape
    storage = (sequence + 63) // 64 * 64
    prepared, scales = _prepared_context_input(graph, arguments, before)
    common_args = (prepared, scales, arguments["weight_qdata"], arguments["weight_scale"])
    if normalized:
        return linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_int8_piper_project_key.default,
            (
                *common_args,
                arguments.get("norm_weight"),
                arguments["cos"],
                arguments["sin"],
                arguments["norm_epsilon"],
                arguments["bias"],
            ),
            (
                input_value.new_empty((batch, heads, storage, head_dim), dtype=torch.int8),
                input_value.new_empty((batch, heads, storage // 64), dtype=torch.float32),
            ),
            kwargs={"head_dim": head_dim},
        )
    return linear_compile_fx.emit_tuple_result(
        graph,
        torch.ops.piper_kernels.convrot_int8_piper_project_value.default,
        (*common_args, arguments["bias"]),
        (
            input_value.new_empty((batch, heads, head_dim, storage), dtype=torch.int8),
            input_value.new_empty((batch, heads, storage), dtype=torch.float32),
            input_value.new_empty((batch, heads, storage), dtype=torch.float32),
            input_value.new_empty((batch, heads, head_dim), dtype=torch.float32),
        ),
        kwargs={"head_dim": head_dim, "is_causal": is_causal},
    )


def _fuse_context(graph: torch.fx.Graph) -> bool:
    changed = False
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target
            is not torch.ops.piper_kernels.piper_attention_from_quantized_query.default
        ):
            continue
        query_data, query_scale, key_node, value_node, query_length, is_causal = node.args
        if not isinstance(key_node, torch.fx.Node) or not isinstance(value_node, torch.fx.Node):
            continue
        key_match = _context_match(key_node, normalized=True)
        value_match = _context_match(value_node, normalized=False)
        if key_match is None or value_match is None:
            continue
        key_value = preparation_sharing.tensor_metadata(key_node)
        assert key_value is not None
        assert isinstance(is_causal, bool)
        with graph.inserting_before(node):
            key_data, key_scale = _emit_context_projection(
                graph, key_match, normalized=True, is_causal=is_causal, before=node
            )
            value_data, multiplier, logs, mean = _emit_context_projection(
                graph, value_match, normalized=False, is_causal=is_causal, before=node
            )
            key_length = graph.call_function(
                torch.ops.aten.sym_size.int, args=(key_match.kwargs["projection_input"], 1)
            )
            key_length.meta["val"] = key_value.shape[2]
            replacement = graph.call_function(
                torch.ops.piper_kernels.piper_attention_from_quantized.default,
                args=(
                    query_data,
                    query_scale,
                    key_data,
                    key_scale,
                    value_data,
                    multiplier,
                    logs,
                    mean,
                    query_length,
                    key_length,
                    is_causal,
                    key_value.dtype,
                ),
            )
        replacement.meta = node.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        # Upstream projection patterns may share inputs. Dead-code elimination
        # removes only the chains that no longer have an external consumer.
        changed = True
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        _projection_views.normalize_projection_views(graph)
        with _compile_fx.canonical_linear_calls(graph):
            changed = _patterns.apply(graph) > 0
            changed = _fuse_context(graph) or changed
        if changed:
            graph.eliminate_dead_code()
            graph.lint()

    def uuid(self) -> bytes:
        return get_hash_for_files(_source_files(), extra=_COMPILE_PASS_VERSION)


compile_pass = _CompilePass()


class _OutputCompilePass(CustomInferenceAwareGraphPass):
    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        with _compile_fx.canonical_linear_calls(graph):
            changed = _output_compile.fold_attention_output(graph)
        if changed:
            graph.eliminate_dead_code()
            graph.lint()

    def uuid(self) -> bytes:
        return get_hash_for_files(_source_files(), extra=f"{_COMPILE_PASS_VERSION}-output")


output_compile_pass = _OutputCompilePass()


def convrot_int8_piper_compile_options(
    options: Mapping[str, object] | None = None,
    *,
    fuse_output: bool = False,
) -> dict[str, object]:
    """Install Q/K/V fusion, with optional bounded attention-to-output projection."""
    if not isinstance(fuse_output, bool):
        raise TypeError("fuse_output must be boolean")
    passes = (compile_pass, output_compile_pass) if fuse_output else (compile_pass,)
    return preparation_sharing.add_ordered_post_grad_passes(
        options, (*passes, linear_compile.compile_pass)
    )


__all__ = ["convrot_int8_piper_compile_options"]
