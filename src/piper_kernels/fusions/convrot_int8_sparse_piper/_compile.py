"""ConvRot INT8 projection folding for sparse Piper attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.attention.sparse_piper_attention import _backend as attention_backend
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    SUPPORTED_DTYPES,
    is_valid_routing_mode,
)
from piper_kernels.fusions.attention import _head_slices as head_slices
from piper_kernels.fusions.attention import _residual_fold as residual_fold
from piper_kernels.fusions.sparse_piper import _compile as sparse_piper_compile
from piper_kernels.linear import _bias
from piper_kernels.linear import _compile_fx as linear_compile_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear import _projection_views as projection_views
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8 import _compile_fx

from . import _backend, _distributed_compile, _layout, _output_compile
from . import key as key_operators  # noqa: F401 - registers the K projection
from . import value as value_operators  # noqa: F401 - registers the V projections

_TILE_ROWS = _layout.TILE_ROWS
_LINEAR_TARGET = torch.ops.piper_kernels.convrot_int8_linear.default


@dataclass(frozen=True, slots=True)
class _BoundInt8Projection:
    """One schema-bound unactivated ConvRot INT8 linear projection."""

    node: torch.fx.Node
    input: torch.fx.Node
    weight_qdata: torch.fx.Node
    weight_scale: torch.fx.Node
    bias: torch.fx.Node | None
    group_size: int
    input_scale: torch.fx.Node | None


type _SemanticGateProjection = tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node | None,
    torch.fx.Node | None,
]


def _parse_int8_projection(node: torch.fx.Node) -> _BoundInt8Projection | None:
    bound = linear_compile_fx.bind_call(node, _LINEAR_TARGET)
    if bound is None or bound["activation_fn"] is not None:
        return None
    input_node = bound["input"]
    weight_qdata = bound["weight_qdata"]
    weight_scale = bound["weight_scale"]
    bias = bound["bias"]
    group_size = bound["group_size"]
    input_scale = bound["input_scale"]
    if (
        not isinstance(input_node, torch.fx.Node)
        or not isinstance(weight_qdata, torch.fx.Node)
        or not isinstance(weight_scale, torch.fx.Node)
        or (bias is not None and not isinstance(bias, torch.fx.Node))
        or isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or (input_scale is not None and not isinstance(input_scale, torch.fx.Node))
    ):
        return None
    return _BoundInt8Projection(
        node=node,
        input=input_node,
        weight_qdata=weight_qdata,
        weight_scale=weight_scale,
        bias=bias,
        group_size=group_size,
        input_scale=input_scale,
    )


def _semantic_gate_projection(
    gate: torch.fx.Node,
    sparse_input: torch.fx.Node,
    group_size: int,
) -> _SemanticGateProjection | None:
    """Match a reshaped ConvRot INT8 gate derived from the shared sparse input."""
    linear = sparse_piper_compile.unwrap_shape_only_views(gate)
    if linear is None or linear is gate:
        return None
    parsed = _parse_int8_projection(linear)
    if parsed is None or parsed.input is not sparse_input or parsed.group_size != group_size:
        return None
    gate_value = preparation_sharing.tensor_metadata(gate)
    weight_value = preparation_sharing.tensor_metadata(parsed.weight_qdata)
    scale_value = preparation_sharing.tensor_metadata(parsed.weight_scale)
    if (
        gate_value is None
        or weight_value is None
        or scale_value is None
        or gate_value.dtype not in SUPPORTED_DTYPES
        or weight_value.dtype is not torch.int8
        or weight_value.ndim != 2
        or scale_value.dtype is not torch.float32
        or tuple(scale_value.shape) != (weight_value.shape[0], 1)
        or not _compile_fx.valid_input_scale(parsed.input_scale, weight_value.device)
    ):
        return None
    return linear, parsed.weight_qdata, parsed.weight_scale, parsed.bias, parsed.input_scale


def _valid_sparse_piper_projection(  # noqa: PLR0911, PLR0912
    matched: sparse_piper_compile.SparsePiperProjectionMatch,
    q_proj: _BoundInt8Projection,
    k_proj: _BoundInt8Projection,
    v_proj: _BoundInt8Projection,
) -> bool:
    routing_mode = matched.routing_mode
    if (
        isinstance(routing_mode, bool)
        or not isinstance(routing_mode, int)
        or not is_valid_routing_mode(routing_mode)
    ):
        return False
    if (
        k_proj.input is not q_proj.input
        or v_proj.input is not q_proj.input
        or k_proj.group_size != q_proj.group_size
        or v_proj.group_size != q_proj.group_size
    ):
        return False
    input_value = preparation_sharing.tensor_metadata(q_proj.input)
    projections = (q_proj, k_proj, v_proj)
    weights = tuple(preparation_sharing.tensor_metadata(p.weight_qdata) for p in projections)
    scales = tuple(preparation_sharing.tensor_metadata(p.weight_scale) for p in projections)
    if input_value is None or any(value is None for value in (*weights, *scales)):
        return False
    if any(
        value.layout is not torch.strided or not value.is_contiguous()
        for value in (input_value, *weights, *scales)
        if value is not None
    ):
        return False
    if any(
        not _compile_fx.valid_input_scale(p.input_scale, input_value.device) for p in projections
    ):
        return False
    if input_value.ndim != 3:
        return False
    input_features = sparse_piper_compile.static_int(input_value.shape[-1])
    q_weight = weights[0]
    assert q_weight is not None
    output_features = sparse_piper_compile.static_int(q_weight.shape[0])
    head_dim = sparse_piper_compile.attention_head_dim(matched.output_node)
    if (
        input_features is None
        or output_features is None
        or head_dim is None
        or output_features % head_dim
    ):
        return False
    batch, sequence_length = input_value.shape[:2]
    heads = output_features // head_dim
    if input_value.dtype not in SUPPORTED_DTYPES or (
        isinstance(sequence_length, int) and sequence_length < _TILE_ROWS
    ):
        return False
    if (
        _backend.select_projection_backend(input_value, head_dim=head_dim) is None
        or linear_backend.select_linear_backend(input_value) is None
        or attention_backend.select_attention_backend(input_value) is None
    ):
        return False
    if q_proj.group_size < 1:
        return False

    for proj, weight, scale in zip(projections, weights, scales, strict=True):
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
        if proj.bias is not None:
            bias = preparation_sharing.tensor_metadata(proj.bias)
            if (
                bias is None
                or bias.shape != (output_features,)
                or not _bias.is_supported_dtype(bias.dtype)
                or bias.device != input_value.device
                or bias.layout is not torch.strided
                or not bias.is_contiguous()
            ):
                return False

    if not sparse_piper_compile.valid_sparse_piper_attention(
        matched,
        batch=batch,
        sequence_length=sequence_length,
        heads=heads,
        device=input_value.device,
        head_dim=head_dim,
        tile_rows=_TILE_ROWS,
    ):
        return False
    if matched.coarse_node is not None:
        return sparse_piper_compile.valid_sparse_piper_coarse_residual(matched)
    return True


def _replace_sparse_piper_projection(
    graph: torch.fx.Graph,
    matched: sparse_piper_compile.SparsePiperProjectionMatch,
    q_proj: _BoundInt8Projection,
    k_proj: _BoundInt8Projection,
    v_proj: _BoundInt8Projection,
) -> None:
    original = matched.output_node
    sparse_input = q_proj.input
    sparse_group_size = q_proj.group_size
    input_value = preparation_sharing.tensor_metadata(sparse_input)
    assert input_value is not None
    batch, sequence_length = input_value.shape[:2]
    q_weight_value = preparation_sharing.tensor_metadata(q_proj.weight_qdata)
    assert q_weight_value is not None
    head_dim = sparse_piper_compile.attention_head_dim(original)
    assert head_dim is not None
    heads = q_weight_value.shape[0] // head_dim
    storage_sequence_length = _layout.padded_sequence_length(sequence_length)
    value_format = attention_backend.require_attention_backend(input_value).value_format
    block_length_arguments = () if matched.block_lengths is None else (matched.block_lengths,)
    query_values, key_values, value_values = sparse_piper_compile.sparse_projection_metadata(
        input_value,
        batch=batch,
        heads=heads,
        storage_sequence_length=storage_sequence_length,
        head_dim=head_dim,
        mean_pool_routing=bool(matched.routing_mode == _MEAN_ROUTING),
        value_shape=value_format.shape(batch, heads, storage_sequence_length, head_dim),
    )
    with graph.inserting_before(original):
        logical_sequence_length = graph.call_function(
            torch.ops.aten.sym_size.int, args=(sparse_input, 1)
        )
        logical_sequence_length.meta["val"] = sequence_length
        preparations: dict[torch.fx.Node | None, _compile_fx.PreparedInputNodes] = {}

        def prepare(input_scale: torch.fx.Node | None) -> _compile_fx.PreparedInputNodes:
            if input_scale not in preparations:
                preparations[input_scale] = _compile_fx.emit_prepared_input(
                    graph,
                    sparse_input,
                    sparse_group_size,
                    None,
                    tuple(input_value.shape),
                    input_scale,
                )
            return preparations[input_scale]

        prepared_coarse_gate = matched.coarse_gate
        if isinstance(prepared_coarse_gate, torch.fx.Node):
            gate_projection = _semantic_gate_projection(
                prepared_coarse_gate, sparse_input, sparse_group_size
            )
            if gate_projection is not None:
                gate_linear, gate_wq, gate_ws, gate_bias, gate_input_scale = gate_projection
                gate_input, gate_scales, _ = prepare(gate_input_scale)
                prepared_gate_linear = graph.call_function(
                    torch.ops.piper_kernels.convrot_int8_linear_prepared.default,
                    args=(
                        gate_input,
                        gate_scales,
                        gate_wq,
                        gate_ws,
                        gate_bias,
                        input_value.dtype,
                    ),
                )
                prepared_gate_linear.meta = gate_linear.meta.copy()
                prepared_gate_linear.meta.pop("eager_input_vals", None)
                prepared_coarse_gate = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(
                        prepared_gate_linear,
                        sparse_piper_compile.view_shape_argument(prepared_coarse_gate),
                    ),
                )
                prepared_coarse_gate.meta = matched.coarse_gate.meta.copy()  # type: ignore[union-attr]

        q_input, q_scales, _ = prepare(q_proj.input_scale)
        query_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_int8_sparse_piper_project_query.default,
            (
                q_input,
                q_scales,
                q_proj.weight_qdata,
                q_proj.weight_scale,
                matched.q_norm_weight,
                matched.cos,
                matched.sin,
                matched.q_norm_epsilon,
                matched.softmax_scale,
                matched.routing_mode,
                *block_length_arguments,
            ),
            query_values,
            kwargs={"head_dim": head_dim, "bias": q_proj.bias},
        )
        k_input, k_scales, _ = prepare(k_proj.input_scale)
        key_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_int8_sparse_piper_project_key.default,
            (
                k_input,
                k_scales,
                k_proj.weight_qdata,
                k_proj.weight_scale,
                matched.k_norm_weight,
                matched.cos,
                matched.sin,
                matched.k_norm_epsilon,
                matched.routing_mode,
                *block_length_arguments,
            ),
            key_values,
            kwargs={"head_dim": head_dim, "bias": k_proj.bias},
        )
        v_input, v_scales, _ = prepare(v_proj.input_scale)
        value_nodes = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_int8_sparse_piper_project_value.default,
            (
                v_input,
                v_scales,
                v_proj.weight_qdata,
                v_proj.weight_scale,
                matched.block_lengths,
                head_dim,
            ),
            value_values,
            kwargs={"bias": v_proj.bias, "value_format": int(value_format)},
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
            output_dtype=original.meta["val"].dtype,
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    graph.erase_node(original)


def _fold_sparse_piper_projection(graph: torch.fx.Graph) -> bool:
    """Replace compatible materialized sparse-attention projection regions."""
    changed = False
    for node in list(graph.nodes):
        matched = sparse_piper_compile.match_sparse_piper_projection(node)
        if matched is None:
            continue
        q_proj = _parse_int8_projection(matched.query_projection)
        k_proj = _parse_int8_projection(matched.key_projection)
        v_proj = _parse_int8_projection(matched.value_projection)
        if q_proj is None or k_proj is None or v_proj is None:
            continue
        if not _valid_sparse_piper_projection(matched, q_proj, k_proj, v_proj):
            continue
        _replace_sparse_piper_projection(graph, matched, q_proj, k_proj, v_proj)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


_ops = torch.ops.piper_kernels
# Which residual-add and K/V projection arguments hold rows per head, for head slices.
_HEAD_SLICING = head_slices.HeadSlicing(
    producers=(
        {
            _ops.convrot_int8_sparse_piper_project_key.default: (
                "key",
                "key_scale",
                "key_center",
                "key_center_scale",
                "key_summary",
                "key_aux",
            )
        },
        {
            _ops.convrot_int8_sparse_piper_project_value.default: (
                "value",
                "value_scale_multiplier",
                "value_mean",
            ),
        },
    ),
    producer_rows=("weight_qdata", "weight_scale", "bias"),
    rows=(
        "query_weight_qdata",
        "query_weight_scale",
        "query_bias",
        "gate_weight_qdata",
        "gate_weight_scale",
        "gate_bias",
    ),
    columns="weight_qdata",
    once=("bias",),
    alignment="group_size",
    # A materialized coarse gate holds heads along its third dimension.
    unsliceable=("coarse_gate",),
    # At H3's shape, the add's routing scores and coarse-gate windows bring it up to the
    # K projection's peak, so full-size windows would raise the peak for little speed.
    widened_window_rows=None,
    # Keep budgets are rounded across all heads, so each slice takes its heads' share.
    slice_position=("head_slice_index", "num_head_slices"),
)


def fold_sparse_attention(graph: torch.fx.Graph) -> bool:
    """Fold ConvRot INT8 projections, output projections, residuals, and collectives."""
    changed = projection_views.normalize_projection_views(graph)
    changed = _fold_sparse_piper_projection(graph) or changed
    changed = _output_compile._fold_attention_output(graph) or changed
    post_changed = residual_fold.fold_residual_adds(
        graph,
        _ops.convrot_int8_sparse_piper_projected_query_attention_output.default,
        _ops.convrot_int8_sparse_piper_projected_query_attention_residual_add_.default,
        _HEAD_SLICING,
    )
    post_changed = _distributed_compile.fold_distributed_attention_output(graph) or post_changed
    post_changed = _distributed_compile.fold_prepare_gather_project_kv(graph) or post_changed
    if post_changed:
        # Drop the operands that only the replaced output read.
        graph.eliminate_dead_code()
        graph.lint()
    return changed or post_changed


__all__ = ["fold_sparse_attention"]
