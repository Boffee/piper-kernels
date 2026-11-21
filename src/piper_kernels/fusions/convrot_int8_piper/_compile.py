"""Fold ConvRot INT8 Q/K/V projections into dense Piper operand preparation."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.node import Argument

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    QKQuantizationGranularity,
    count_key_scales,
    count_query_scales,
)
from piper_kernels.attention.piper_attention import _quantized_dispatch
from piper_kernels.fusions.attention import _head_slices as head_slices
from piper_kernels.fusions.attention import _projected_qk_compile as projected_compile
from piper_kernels.fusions.attention import _residual_fold as residual_fold
from piper_kernels.linear import _bias, _projection_views
from piper_kernels.linear import _compile_fx as linear_compile_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _compile_fx
from piper_kernels.weights.convrot import _rotation

from . import _backend, _distributed_compile, _output_compile
from . import key as key_operators  # noqa: F401 - registers the K projection
from . import value as value_operators  # noqa: F401 - registers the V projection

_aten = torch.ops.aten
_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16)


@dataclass(frozen=True, slots=True)
class _MatchedProjection:
    """Validated BHSD ConvRot INT8 projection (with optional RMSNorm/RoPE)."""

    output_node: torch.fx.Node
    projection_input: torch.fx.Node
    weight_qdata: torch.fx.Node
    weight_scale: torch.fx.Node
    bias: torch.fx.Node | None
    group_size: int
    input_scale: torch.fx.Node | None
    norm_weight: torch.fx.Node | None = None
    norm_epsilon: Argument = None
    cos: torch.fx.Node | None = None
    sin: torch.fx.Node | None = None
    rotary_dim: Argument | None = None
    half_rotary_dim: Argument = None


def _qk_quantization(device: torch.device) -> QKQuantizationGranularity:
    """Emit Q/K scales in the granularity the device's quantized attention reads."""
    return _quantized_dispatch.select_qk_quantization(AcceleratorTarget.from_device(device))


def _same_shape(left: Sequence[int | torch.SymInt], right: Sequence[int | torch.SymInt]) -> bool:
    return len(left) == len(right) and all(
        statically_known_true(a == b) for a, b in zip(left, right, strict=True)
    )


def _match_bhsd_transpose(node: torch.fx.Node) -> torch.fx.Node | None:
    """Match ``permute(x, [0, 2, 1, 3])`` or ``transpose(x, 1, 2)``."""
    permute = linear_compile_fx.bind_call(node, _aten.permute.default)
    if permute is not None:
        dims = permute["dims"]
        source = permute["self"]
        if (
            isinstance(source, torch.fx.Node)
            and isinstance(dims, (list, tuple))
            and tuple(dims) == (0, 2, 1, 3)
        ):
            return source
        return None
    transpose = linear_compile_fx.bind_call(node, _aten.transpose.int)
    if transpose is not None:
        source = transpose["self"]
        if isinstance(source, torch.fx.Node) and (transpose["dim0"], transpose["dim1"]) in (
            (1, 2),
            (2, 1),
        ):
            return source
    return None


def _valid_projected_tensor(  # noqa: PLR0911
    matched: _MatchedProjection, *, normalized: bool = True
) -> bool:
    input_value = preparation_sharing.tensor_metadata(matched.projection_input)
    weight = preparation_sharing.tensor_metadata(matched.weight_qdata)
    weight_scale = preparation_sharing.tensor_metadata(matched.weight_scale)
    output_value = preparation_sharing.tensor_metadata(matched.output_node)
    if input_value is None or weight is None or weight_scale is None or output_value is None:
        return False
    if (
        output_value.ndim != 4
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
    group_size = projected_compile.static_int(matched.group_size)
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
        or not _compile_fx.valid_input_scale(matched.input_scale, input_value.device)
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
    if matched.bias is not None:
        bias = preparation_sharing.tensor_metadata(matched.bias)
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
    rotary_dim = head_dim if matched.rotary_dim is None else matched.rotary_dim
    return (
        projected_compile.valid_rmsnorm(
            matched.norm_weight,
            matched.norm_epsilon,
            head_dim=head_dim,
            device=input_value.device,
            supported_dtypes=(*_ACTIVATION_DTYPES, torch.float32),
        )
        and projected_compile.valid_rope_tables(
            matched.cos,
            matched.sin,
            rotary_dim,
            matched.half_rotary_dim,
            sequence_length=sequence_length,
            head_dim=head_dim,
            device=input_value.device,
        )
        and _backend.select_projection_backend(input_value, head_dim=head_dim) is not None
    )


def _match_bhsd_projection(node: object, *, normalized: bool) -> _MatchedProjection | None:  # noqa: PLR0911
    """Imperatively match one exclusive BHSD ConvRot INT8 projection chain."""
    if not isinstance(node, torch.fx.Node) or len(node.users) != 1:
        return None
    transformed = _match_bhsd_transpose(node)
    if transformed is None:
        return None
    if normalized:
        rope = projected_compile.match_normalized_rope(transformed, expected_users=1)
        if rope is None or rope.activation_dtype not in _ACTIVATION_DTYPES:
            return None
        linear_node = rope.projection_node
    else:
        if len(transformed.users) != 1:
            return None
        reshape = linear_compile_fx.bind_call(transformed, _aten.reshape.default)
        if reshape is None or not isinstance(reshape["self"], torch.fx.Node):
            return None
        linear_node = reshape["self"]
        if len(linear_node.users) != 1:
            return None
        rope = None

    linear_bound = linear_compile_fx.bind_call(
        linear_node, torch.ops.piper_kernels.convrot_int8_linear.default
    )
    if linear_bound is None or linear_bound["activation_fn"] is not None:
        return None
    projection_input = linear_bound["input"]
    weight_qdata = linear_bound["weight_qdata"]
    weight_scale = linear_bound["weight_scale"]
    bias = linear_bound["bias"]
    group_size = linear_bound["group_size"]
    input_scale = linear_bound["input_scale"]
    if (
        not isinstance(projection_input, torch.fx.Node)
        or not isinstance(weight_qdata, torch.fx.Node)
        or not isinstance(weight_scale, torch.fx.Node)
        or (bias is not None and not isinstance(bias, torch.fx.Node))
        or isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or (input_scale is not None and not isinstance(input_scale, torch.fx.Node))
    ):
        return None
    matched = _MatchedProjection(
        output_node=node,
        projection_input=projection_input,
        weight_qdata=weight_qdata,
        weight_scale=weight_scale,
        bias=bias,
        group_size=group_size,
        input_scale=input_scale,
        norm_weight=None if rope is None else rope.norm_weight,
        norm_epsilon=None if rope is None else rope.norm_epsilon,
        cos=None if rope is None else rope.cos,
        sin=None if rope is None else rope.sin,
        rotary_dim=None if rope is None else rope.rotary_dim,
        half_rotary_dim=None if rope is None else rope.half_rotary_dim,
    )
    return matched if _valid_projected_tensor(matched, normalized=normalized) else None


def _valid_query_attention(
    query_match: _MatchedProjection,
    key_arg: Argument,
    value_arg: Argument,
    softmax_scale: Argument,
    is_causal: Argument,
) -> bool:
    query_value = preparation_sharing.tensor_metadata(query_match.output_node)
    if not isinstance(key_arg, torch.fx.Node) or not isinstance(value_arg, torch.fx.Node):
        return False
    key = preparation_sharing.tensor_metadata(key_arg)
    value = preparation_sharing.tensor_metadata(value_arg)
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
        or not isinstance(is_causal, bool)
        or (is_causal and not _same_shape((sequence_length,), (key.shape[2],)))
        or projected_compile.positive_float(softmax_scale) is None
    )


def _fold_query_projections(graph: torch.fx.Graph) -> bool:
    changed = False
    for node in list(graph.nodes):
        bound = linear_compile_fx.bind_call(node, torch.ops.piper_kernels.piper_attention.default)
        if bound is None:
            continue
        query_match = _match_bhsd_projection(bound["query"], normalized=True)
        if query_match is None:
            continue
        key_arg = bound["key"]
        value_arg = bound["value"]
        softmax_scale = bound["scale"]
        is_causal = bound["is_causal"]
        if not _valid_query_attention(query_match, key_arg, value_arg, softmax_scale, is_causal):
            continue
        assert isinstance(key_arg, torch.fx.Node)
        assert isinstance(value_arg, torch.fx.Node)
        assert isinstance(is_causal, bool)
        input_value = preparation_sharing.tensor_metadata(query_match.projection_input)
        query_value = preparation_sharing.tensor_metadata(query_match.output_node)
        assert input_value is not None
        assert query_value is not None
        batch, heads, sequence_length, head_dim = query_value.shape
        storage_rows = (sequence_length + 63) // 64 * 64
        qk_quantization = _qk_quantization(input_value.device)
        num_scales = count_query_scales(storage_rows, qk_quantization)
        with graph.inserting_before(node):
            query_length = graph.call_function(
                _aten.sym_size.int, args=(query_match.projection_input, 1)
            )
            query_length.meta["val"] = sequence_length
            input_qdata, scales, _ = _compile_fx.emit_prepared_input(
                graph,
                query_match.projection_input,
                query_match.group_size,
                None,
                tuple(input_value.shape),
                query_match.input_scale,
            )
            query_data, query_scale = linear_compile_fx.emit_tuple_result(
                graph,
                torch.ops.piper_kernels.convrot_int8_piper_project_query.default,
                (
                    input_qdata,
                    scales,
                    query_match.weight_qdata,
                    query_match.weight_scale,
                    query_match.norm_weight,
                    query_match.cos,
                    query_match.sin,
                    query_match.norm_epsilon,
                    softmax_scale,
                    query_match.bias,
                ),
                (
                    input_value.new_empty((batch, heads, storage_rows, head_dim), dtype=torch.int8),
                    input_value.new_empty((batch, heads, num_scales), dtype=torch.float32),
                ),
                kwargs={"head_dim": head_dim, "qk_quantization": qk_quantization},
            )
            replacement = graph.call_function(
                torch.ops.piper_kernels.piper_attention_from_quantized_query.default,
                args=(query_data, query_scale, key_arg, value_arg, query_length, is_causal),
            )
        replacement.meta = node.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True
    return changed


def _prepared_context_input(
    graph: torch.fx.Graph, matched: _MatchedProjection, before: torch.fx.Node
) -> tuple[torch.fx.Node, torch.fx.Node]:
    input_node = matched.projection_input
    prepared: dict[torch.fx.Node, dict[int, torch.fx.Node]] = {}
    for node in graph.nodes:
        if node is before:
            break
        if node.op != "call_function":
            continue
        prep_bound = linear_compile_fx.bind_call(
            node, torch.ops.piper_kernels.convrot_int8_prepare_input.default
        )
        if (
            prep_bound is not None
            and prep_bound["input"] is input_node
            and prep_bound["group_size"] == matched.group_size
            and prep_bound["activation_fn"] is None
            and prep_bound["input_scale"] is matched.input_scale
        ):
            prepared[node] = {}
        elif (
            node.target is operator.getitem
            and not node.kwargs
            and len(node.args) == 2
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
        matched.group_size,
        None,
        tuple(input_value.shape),
        matched.input_scale,
    )
    return data, scales


def _emit_context_projection(
    graph: torch.fx.Graph,
    matched: _MatchedProjection,
    *,
    normalized: bool,
    is_causal: bool,
    before: torch.fx.Node,
) -> tuple[torch.fx.Node, ...]:
    input_value = preparation_sharing.tensor_metadata(matched.projection_input)
    output_value = preparation_sharing.tensor_metadata(matched.output_node)
    assert input_value is not None
    assert output_value is not None
    batch, heads, sequence, head_dim = output_value.shape
    storage = (sequence + 63) // 64 * 64
    chunks = (sequence + 1023) // 1024
    prepared, scales = _prepared_context_input(graph, matched, before)
    common_args = (prepared, scales, matched.weight_qdata, matched.weight_scale)
    if normalized:
        qk_quantization = _qk_quantization(input_value.device)
        return linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_int8_piper_project_key.default,
            (
                *common_args,
                matched.norm_weight,
                matched.cos,
                matched.sin,
                matched.norm_epsilon,
                matched.bias,
            ),
            (
                input_value.new_empty((batch, heads, storage, head_dim), dtype=torch.int8),
                input_value.new_empty(
                    (batch, heads, count_key_scales(storage, qk_quantization)),
                    dtype=torch.float32,
                ),
                input_value.new_empty((batch, heads, chunks, head_dim), dtype=torch.float32),
            ),
            kwargs={"head_dim": head_dim, "qk_quantization": qk_quantization},
        )
    return linear_compile_fx.emit_tuple_result(
        graph,
        torch.ops.piper_kernels.convrot_int8_piper_project_value.default,
        (*common_args, matched.bias),
        (
            input_value.new_empty((batch, heads, head_dim, storage), dtype=torch.int8),
            input_value.new_empty((batch, heads, storage), dtype=torch.float32),
            input_value.new_empty((batch, heads, storage), dtype=torch.float32),
            input_value.new_empty((batch, heads, chunks, head_dim), dtype=torch.float32),
        ),
        kwargs={"head_dim": head_dim, "is_causal": is_causal},
    )


def _fuse_context(graph: torch.fx.Graph) -> bool:
    changed = False
    for node in list(graph.nodes):
        bound = linear_compile_fx.bind_call(
            node, torch.ops.piper_kernels.piper_attention_from_quantized_query.default
        )
        if bound is None:
            continue
        query_data = bound["query"]
        query_scale = bound["query_scale"]
        key_node = bound["key"]
        value_node = bound["value"]
        query_length = bound["query_length"]
        is_causal = bound["is_causal"]
        if not isinstance(key_node, torch.fx.Node) or not isinstance(value_node, torch.fx.Node):
            continue
        key_match = _match_bhsd_projection(key_node, normalized=True)
        value_match = _match_bhsd_projection(value_node, normalized=False)
        if key_match is None or value_match is None:
            continue
        key_value = preparation_sharing.tensor_metadata(key_node)
        assert key_value is not None
        assert isinstance(is_causal, bool)
        with graph.inserting_before(node):
            key_data, key_scale, key_center = _emit_context_projection(
                graph, key_match, normalized=True, is_causal=is_causal, before=node
            )
            value_data, multiplier, logs, mean = _emit_context_projection(
                graph, value_match, normalized=False, is_causal=is_causal, before=node
            )
            key_length = graph.call_function(
                _aten.sym_size.int, args=(key_match.projection_input, 1)
            )
            key_length.meta["val"] = key_value.shape[2]
            replacement = graph.call_function(
                torch.ops.piper_kernels.piper_attention_from_quantized.default,
                args=(
                    query_data,
                    query_scale,
                    key_data,
                    key_scale,
                    key_center,
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
        changed = True
    return changed


def fold_attention_projections(graph: torch.fx.Graph) -> bool:
    """Fold ConvRot INT8 Q/K/V projections into dense Piper operand preparation."""
    changed = _projection_views.normalize_projection_views(graph)
    changed = linear_compile_fx.coalesce_input_casts(graph) or changed
    changed = _fold_query_projections(graph) or changed
    changed = _fuse_context(graph) or changed
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


_kernels = torch.ops.piper_kernels
# Which residual-add and K/V projection arguments hold rows per head, for head slices.
_HEAD_SLICING = head_slices.HeadSlicing(
    producers=(
        {_kernels.convrot_int8_piper_project_key.default: ("key", "key_scale", "key_center")},
        {
            _kernels.convrot_int8_piper_project_value.default: (
                "value",
                "multiplier",
                "log_scale",
                "value_mean",
            )
        },
    ),
    producer_rows=("weight_qdata", "weight_scale", "bias"),
    rows=("query_weight", "query_weight_scale", "query_bias"),
    columns="weight_qdata",
    once=("bias",),
    alignment="group_size",
    unsliceable=(),
    # Dense launches lose efficiency at half the heads, and at H3's shape the add has
    # room below the K projection's peak, so full-size windows add little peak.
    widened_window_rows="query_chunk_rows",
    slice_position=None,
)


def fold_attention_output(graph: torch.fx.Graph) -> bool:
    """Fold dense attention output projections, residual adds, and reduce-scatters."""
    changed = _output_compile.fold_attention_output(graph)
    if changed:
        # Drop the replaced attention, so the fused output alone reads its K and V.
        graph.eliminate_dead_code()
    changed = (
        residual_fold.fold_residual_adds(
            graph,
            _kernels.convrot_int8_piper_projected_query_attention_output.default,
            _kernels.convrot_int8_piper_projected_query_attention_residual_add_.default,
            _HEAD_SLICING,
        )
        or changed
    )
    changed = _distributed_compile.fold_distributed_attention_output(graph) or changed
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__ = ["fold_attention_output", "fold_attention_projections"]
