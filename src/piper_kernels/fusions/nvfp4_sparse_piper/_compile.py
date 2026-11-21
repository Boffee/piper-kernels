"""NVFP4 and ConvRot NVFP4 projection folding for sparse Piper attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels.attention.kernels.sparse_piper import layout
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    SUPPORTED_DTYPES,
    is_valid_routing_mode,
)
from piper_kernels.fusions.sparse_piper import _compile as sparse_piper_compile
from piper_kernels.linear import _compile_fx as linear_compile_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear import _projection_views as projection_views
from piper_kernels.linear.nvfp4 import _chunking as nvfp4_chunking
from piper_kernels.linear.nvfp4 import _compile_fx

from . import _output_compile, _validation
from . import key as key_operators  # noqa: F401 - registers the K projection
from . import value as value_operators  # noqa: F401 - registers the V projections

_aten = torch.ops.aten
_TILE_ROWS = layout.TILE_ROWS
_NVFP4_LINEAR = "piper_kernels::nvfp4_linear"
_CONVROT_NVFP4_LINEAR = "piper_kernels::convrot_nvfp4_linear"
_NVFP4_LINEAR_PREPARED = "piper_kernels::nvfp4_linear_prepared"


@dataclass(frozen=True, slots=True)
class _MatchedNvfp4Projection:
    """One validated prepared or semantic NVFP4 / ConvRot NVFP4 projection."""

    sequence_length: int | torch.SymInt
    heads: int
    device: torch.device
    prepared: _compile_fx.PreparedLinearNodes | None = None
    semantic: _compile_fx.SemanticLinearNodes | None = None
    group_size: int | None = None


def _parse_prepared_projection(  # noqa: PLR0911
    node: torch.fx.Node,
    *,
    head_dim: int,
    name: str,
) -> _MatchedNvfp4Projection | None:
    reshape_bound = linear_compile_fx.bind_call(node, _aten.reshape.default)
    if reshape_bound is None:
        return None
    prepared_node = reshape_bound["self"]
    if (
        not isinstance(prepared_node, torch.fx.Node)
        or len(prepared_node.users) != 1
        or linear_compile_fx.schema_name(prepared_node) != _NVFP4_LINEAR_PREPARED
    ):
        return None
    operands = _compile_fx.PreparedLinearNodes.from_call(prepared_node)
    if operands is None or operands.logical_dtype not in SUPPORTED_DTYPES:
        return None
    input_qdata = preparation_sharing.tensor_metadata(operands.input_qdata)
    input_scale = preparation_sharing.tensor_metadata(operands.input_scale)
    input_global = preparation_sharing.tensor_metadata(operands.input_per_tensor_scale)
    weight = preparation_sharing.tensor_metadata(operands.weight_qdata)
    weight_scale = preparation_sharing.tensor_metadata(operands.weight_scale)
    global_valid, global_scale = sparse_piper_compile.optional_tensor_metadata(
        operands.weight_per_tensor_scale
    )
    bias_valid, bias = sparse_piper_compile.optional_tensor_metadata(operands.bias)
    if (
        input_qdata is None
        or input_scale is None
        or input_global is None
        or weight is None
        or weight_scale is None
        or not global_valid
        or not bias_valid
    ):
        return None
    try:
        sequence_length, heads = _validation.validate_projection(
            input_qdata,
            input_scale,
            input_global,
            weight,
            weight_scale,
            global_scale,
            bias,
            nvfp4_chunking.DEFAULT_CHUNK_ROWS,
            name,
            head_dim,
        )
    except ValueError:
        return None
    linear_shape = reshape_bound["shape"]
    linear_sequence_length = (
        sparse_piper_compile.integer_scalar_metadata(linear_shape[1])
        if isinstance(linear_shape, (list, tuple)) and len(linear_shape) == 3
        else None
    )
    output_features = heads * head_dim
    if (
        not isinstance(linear_shape, (list, tuple))
        or len(linear_shape) != 3
        or sparse_piper_compile.static_int(linear_shape[0]) != 1
        or linear_sequence_length is None
        or preparation_sharing.dimension_key(linear_sequence_length)
        != preparation_sharing.dimension_key(sequence_length)
        or sparse_piper_compile.static_int(linear_shape[2]) != output_features
    ):
        return None
    return _MatchedNvfp4Projection(
        sequence_length=sequence_length,
        heads=heads,
        device=input_qdata.device,
        prepared=operands,
    )


def _parse_semantic_projection(  # noqa: PLR0911
    node: torch.fx.Node,
    *,
    head_dim: int,
    name: str,
) -> _MatchedNvfp4Projection | None:
    op_name = linear_compile_fx.schema_name(node)
    if op_name == _NVFP4_LINEAR:
        linear = _compile_fx.SemanticLinearNodes.from_call(node)
        if linear is None:
            return None
        validated = _compile_fx.validated_semantic_linear(linear, name)
        group_size: int | None = None
    elif op_name == _CONVROT_NVFP4_LINEAR:
        from piper_kernels.linear.convrot.nvfp4 import (  # noqa: PLC0415
            _compile_fx as convrot_nvfp4_compile_fx,
        )

        convrot = convrot_nvfp4_compile_fx.SemanticLinearNodes.from_call(node)
        if convrot is None:
            return None
        validated = convrot_nvfp4_compile_fx.validated_semantic_linear(convrot, name)
        linear = convrot.linear
        group_size = convrot.group_size
    else:
        return None

    if validated is None:
        return None
    input_value, shape = validated
    output_features = sparse_piper_compile.static_int(shape.output_features)
    if (
        input_value.ndim != 3
        or sparse_piper_compile.static_int(input_value.shape[0]) != 1
        or input_value.dtype not in SUPPORTED_DTYPES
        or not input_value.is_contiguous()
        or output_features is None
        or output_features % head_dim
    ):
        return None
    sequence_length = input_value.shape[1]
    if isinstance(sequence_length, int) and sequence_length < _TILE_ROWS:
        return None
    return _MatchedNvfp4Projection(
        sequence_length=sequence_length,
        heads=output_features // head_dim,
        device=input_value.device,
        semantic=linear,
        group_size=group_size,
    )


def _parse_projection(
    node: torch.fx.Node,
    *,
    head_dim: int,
    name: str,
) -> _MatchedNvfp4Projection | None:
    prepared = _parse_prepared_projection(node, head_dim=head_dim, name=name)
    if prepared is not None:
        return prepared
    return _parse_semantic_projection(node, head_dim=head_dim, name=name)


def _valid_projections(  # noqa: PLR0911
    matched: sparse_piper_compile.SparsePiperProjectionMatch,
) -> tuple[_MatchedNvfp4Projection, _MatchedNvfp4Projection, _MatchedNvfp4Projection, int] | None:
    routing_mode = matched.routing_mode
    if (
        isinstance(routing_mode, bool)
        or not isinstance(routing_mode, int)
        or not is_valid_routing_mode(routing_mode)
    ):
        return None
    head_dim = sparse_piper_compile.attention_head_dim(matched.output_node)
    if head_dim is None:
        return None
    parsed = tuple(
        _parse_projection(node, head_dim=head_dim, name=f"sparse_{prefix} compiler projection")
        for prefix, node in (
            ("q", matched.query_projection),
            ("k", matched.key_projection),
            ("v", matched.value_projection),
        )
    )
    if any(item is None for item in parsed):
        return None
    q_proj, k_proj, v_proj = cast(
        tuple[_MatchedNvfp4Projection, _MatchedNvfp4Projection, _MatchedNvfp4Projection], parsed
    )
    sequence_length = q_proj.sequence_length
    heads = q_proj.heads
    device = q_proj.device
    if any(
        preparation_sharing.dimension_key(proj.sequence_length)
        != preparation_sharing.dimension_key(sequence_length)
        or proj.heads != heads
        or proj.device != device
        for proj in (k_proj, v_proj)
    ):
        return None
    if not sparse_piper_compile.valid_sparse_piper_attention(
        matched,
        batch=1,
        sequence_length=sequence_length,
        heads=heads,
        device=device,
        head_dim=head_dim,
        tile_rows=_TILE_ROWS,
    ):
        return None
    if (
        matched.coarse_node is not None
        and not sparse_piper_compile.valid_sparse_piper_coarse_residual(matched)
    ):
        return None
    return q_proj, k_proj, v_proj, head_dim


def _emit_prepared_storage(
    graph: torch.fx.Graph,
    projection: _MatchedNvfp4Projection,
    preparations: dict[tuple[object, ...], _compile_fx.PreparedInputNodes],
) -> tuple[torch.fx.Node | None, ...]:
    if projection.prepared is not None:
        return projection.prepared.storage_arguments()
    semantic = projection.semantic
    assert semantic is not None
    key = (
        semantic.input,
        projection.group_size,
        semantic.activation_per_tensor_scale,
        semantic.dynamic_activation_scale,
        semantic.high_first,
    )
    if key not in preparations:
        if projection.group_size is None:
            preparations[key] = _compile_fx.emit_prepared_input(
                graph,
                semantic.input,
                semantic.activation_per_tensor_scale,
                semantic.dynamic_activation_scale,
                high_first=semantic.high_first,
            )
        else:
            from piper_kernels.linear.convrot.nvfp4 import (  # noqa: PLC0415
                _compile_fx as convrot_nvfp4_compile_fx,
            )

            preparations[key] = convrot_nvfp4_compile_fx.emit_prepared_input(
                graph,
                convrot_nvfp4_compile_fx.SemanticLinearNodes(semantic, projection.group_size),
            )
    input_qdata, input_scale, input_per_tensor_scale, _ = preparations[key]
    return (
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        semantic.weight_qdata,
        semantic.weight_scale,
        semantic.weight_per_tensor_scale,
        semantic.bias,
    )


def _prepare_semantic_coarse_gate(
    graph: torch.fx.Graph,
    coarse_gate: Argument,
    head_dim: int,
    preparations: dict[tuple[object, ...], _compile_fx.PreparedInputNodes],
) -> Argument:
    """Lower a semantic coarse gate to prepared form so output folding can defer it."""
    if not isinstance(coarse_gate, torch.fx.Node):
        return coarse_gate
    linear_node = sparse_piper_compile.unwrap_shape_only_views(coarse_gate)
    if linear_node is None or linear_node is coarse_gate:
        return coarse_gate
    parsed = _parse_semantic_projection(
        linear_node, head_dim=head_dim, name="NVFP4 sparse coarse gate"
    )
    if parsed is None or parsed.semantic is None:
        return coarse_gate
    storage = _emit_prepared_storage(graph, parsed, preparations)
    input_qdata, input_scale, input_per_tensor_scale = storage[:3]
    assert isinstance(input_qdata, torch.fx.Node)
    assert isinstance(input_scale, torch.fx.Node)
    assert isinstance(input_per_tensor_scale, torch.fx.Node)
    input_value = preparation_sharing.tensor_metadata(parsed.semantic.input)
    weight_value = preparation_sharing.tensor_metadata(parsed.semantic.weight_qdata)
    assert input_value is not None
    assert weight_value is not None
    prepared_linear = graph.call_function(
        torch.ops.piper_kernels.nvfp4_linear_prepared.default,
        args=(
            input_qdata,
            input_scale,
            input_per_tensor_scale,
            parsed.semantic.weight_qdata,
            parsed.semantic.weight_scale,
            parsed.semantic.weight_per_tensor_scale,
            parsed.semantic.bias,
            input_value.dtype,
        ),
    )
    prepared_linear.meta["val"] = input_value.new_empty(
        (input_qdata.meta["val"].shape[0], weight_value.shape[0])
    )
    reshaped = graph.call_function(
        _aten.reshape.default,
        args=(prepared_linear, sparse_piper_compile.view_shape_argument(coarse_gate)),
    )
    reshaped.meta = coarse_gate.meta.copy()
    return reshaped


def _replace_projection(
    graph: torch.fx.Graph,
    matched: sparse_piper_compile.SparsePiperProjectionMatch,
    q_proj: _MatchedNvfp4Projection,
    k_proj: _MatchedNvfp4Projection,
    v_proj: _MatchedNvfp4Projection,
    head_dim: int,
) -> None:
    original = matched.output_node
    sequence_length = q_proj.sequence_length
    heads = q_proj.heads
    storage_sequence_length = ((sequence_length + _TILE_ROWS - 1) // _TILE_ROWS) * _TILE_ROWS
    attention_shape = matched.attention_shape
    assert isinstance(attention_shape, (list, tuple))
    logical_sequence_length = sparse_piper_compile.integer_scalar_argument(attention_shape[1])
    assert logical_sequence_length is not None
    block_length_arguments = () if matched.block_lengths is None else (matched.block_lengths,)
    output_value = preparation_sharing.tensor_metadata(original)
    assert output_value is not None
    query_values, key_values, value_values = sparse_piper_compile.sparse_projection_metadata(
        output_value,
        batch=1,
        heads=heads,
        storage_sequence_length=storage_sequence_length,
        head_dim=head_dim,
        mean_pool_routing=bool(matched.routing_mode == _MEAN_ROUTING),
        value_shape=(1, heads, head_dim, storage_sequence_length),
    )
    with graph.inserting_before(original):
        preparations: dict[tuple[object, ...], _compile_fx.PreparedInputNodes] = {}
        q_storage = _emit_prepared_storage(graph, q_proj, preparations)
        k_storage = _emit_prepared_storage(graph, k_proj, preparations)
        v_storage = _emit_prepared_storage(graph, v_proj, preparations)
        prepared_coarse_gate = _prepare_semantic_coarse_gate(
            graph, matched.coarse_gate, head_dim, preparations
        )
        query_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default,
            (
                *q_storage,
                matched.q_norm_weight,
                matched.cos,
                matched.sin,
                matched.q_norm_epsilon,
                matched.softmax_scale,
                nvfp4_chunking.DEFAULT_CHUNK_ROWS,
                matched.routing_mode,
                *block_length_arguments,
            ),
            query_values,
            kwargs={"head_dim": head_dim},
        )
        key_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default,
            (
                *k_storage,
                matched.k_norm_weight,
                matched.cos,
                matched.sin,
                matched.k_norm_epsilon,
                nvfp4_chunking.DEFAULT_CHUNK_ROWS,
                matched.routing_mode,
                *block_length_arguments,
            ),
            key_values,
            kwargs={"head_dim": head_dim},
        )
        value_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default,
            (*v_storage, nvfp4_chunking.DEFAULT_CHUNK_ROWS, *block_length_arguments),
            value_values,
            kwargs={"head_dim": head_dim},
        )
        replacement = sparse_piper_compile.emit_quantized_sparse_piper_attention(
            graph,
            (*query_nodes, *key_nodes, *value_nodes),
            ratio_profile=matched.ratio_profile,
            sparse_key_blocks=matched.sparse_key_blocks,
            logical_sequence_length=logical_sequence_length,
            routing_mode=matched.routing_mode,
            block_lengths=matched.block_lengths,
            sparse_query_blocks=matched.sparse_query_blocks,
            force_tile_ids=matched.force_tile_ids,
            force_tile_counts=matched.force_tile_counts,
            coarse_gate=prepared_coarse_gate,
            coarse_scale=matched.coarse_scale,
            coarse_key_blocks=matched.coarse_key_blocks,
            output_dtype=output_value.dtype,
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    graph.erase_node(original)


def _fold_projection(graph: torch.fx.Graph) -> bool:
    changed = False
    for node in list(graph.nodes):
        matched = sparse_piper_compile.match_sparse_piper_projection(node)
        if matched is None:
            continue
        validated = _valid_projections(matched)
        if validated is None:
            continue
        q_proj, k_proj, v_proj, head_dim = validated
        _replace_projection(graph, matched, q_proj, k_proj, v_proj, head_dim)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


def fold_sparse_attention(graph: torch.fx.Graph) -> bool:
    """Fold NVFP4 / ConvRot NVFP4 sparse-Piper projections and output projections."""
    changed = projection_views.normalize_projection_views(graph)
    changed = _fold_projection(graph) or changed
    changed = _output_compile._fold_attention_output(graph) or changed
    return changed


__all__ = ["fold_sparse_attention"]
