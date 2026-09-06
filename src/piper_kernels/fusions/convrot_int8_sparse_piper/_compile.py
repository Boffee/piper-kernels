"""ConvRot INT8 projection folding for sparse Piper attention."""

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

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_kernels,
)
from piper_kernels.attention.sparse_piper_attention import (
    _coarse_dispatch,
    _quantized_dispatch,
    _routing,
    _summaries,
    coarse,
    dispatch,
    residual,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    is_valid_routing_mode,
)
from piper_kernels.fusions.convrot_int8_sage_qk import triton as convrot_int8_sage_qk
from piper_kernels.fusions.projected_qk import triton as projected_qk
from piper_kernels.fusions.sparse_piper import _compile as sparse_piper_compile
from piper_kernels.fusions.sparse_piper import _output as sparse_piper_output
from piper_kernels.fusions.sparse_piper import _pattern as sparse_piper_pattern
from piper_kernels.linear import _bias
from piper_kernels.linear import _compile_fx as linear_compile_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _compile as convrot_int8_compile
from piper_kernels.linear.convrot.int8 import _compile_fx

from . import _layout, _output_compile, key, output, query, value

_COMPILE_PASS_VERSION = "convrot-int8-sparse-piper-compile-v23"
_HEAD_DIM = _layout.HEAD_DIM
_TILE_ROWS = _layout.TILE_ROWS
_QUERY_SCALE_ROWS = _layout.QUERY_SCALE_ROWS

type _SemanticGateProjection = tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node | None,
]


def _source_files() -> tuple[str, ...]:
    """Return every source file whose changes invalidate this graph rewrite."""
    return tuple(
        file_name
        for file_name in (
            __file__,
            _bias.__file__,
            _layout.__file__,
            _output_compile.__file__,
            qk_quantization.__file__,
            sparse_piper_kernels.__file__,
            convrot_int8_sage_qk.__file__,
            *sparse_piper_compile.source_files(),
            sparse_piper_output.__file__,
            sparse_piper_pattern.__file__,
            linear_compile_fx.__file__,
            projected_qk.__file__,
            query.__file__,
            key.__file__,
            output.__file__,
            value.__file__,
            _coarse_dispatch.__file__,
            _quantized_dispatch.__file__,
            coarse.__file__,
            dispatch.__file__,
            _routing.__file__,
            _summaries.__file__,
            residual.__file__,
            _compile_fx.__file__,
        )
        if file_name is not None
    )


def _linear_pattern(prefix: str) -> CallFunction:
    return CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        KeywordArg("sparse_input"),
        KeywordArg(f"{prefix}_weight_qdata"),
        KeywordArg(f"{prefix}_weight_scale"),
        None,
        KeywordArg("sparse_group_size"),
        _users=1,
    )


def _semantic_gate_projection(
    gate: torch.fx.Node,
    sparse_input: torch.fx.Node,
    group_size: int,
) -> _SemanticGateProjection | None:
    """Match a reshaped ConvRot INT8 gate derived from the shared sparse input."""
    linear = sparse_piper_compile.unwrap_shape_only_views(gate)
    if (
        not isinstance(linear, torch.fx.Node)
        or linear is gate
        or linear.target is not torch.ops.piper_kernels.convrot_int8_linear.default
        or linear.kwargs
        or len(linear.args) not in (5, 6)
    ):
        return None
    arguments = (*linear.args, None) if len(linear.args) == 5 else linear.args
    input_node, weight_qdata, weight_scale, bias, linear_group_size, activation_fn = arguments
    if (
        input_node is not sparse_input
        or linear_group_size != group_size
        or activation_fn is not None
        or not isinstance(weight_qdata, torch.fx.Node)
        or not isinstance(weight_scale, torch.fx.Node)
        or (bias is not None and not isinstance(bias, torch.fx.Node))
    ):
        return None
    gate_value = preparation_sharing.tensor_metadata(gate)
    weight_value = preparation_sharing.tensor_metadata(weight_qdata)
    scale_value = preparation_sharing.tensor_metadata(weight_scale)
    if (
        gate_value is None
        or weight_value is None
        or scale_value is None
        or gate_value.dtype is not torch.bfloat16
        or weight_value.dtype is not torch.int8
        or weight_value.ndim != 2
        or scale_value.dtype is not torch.float32
        or tuple(scale_value.shape) != (weight_value.shape[0], 1)
    ):
        return None
    return linear, weight_qdata, weight_scale, bias


def _valid_sparse_piper_projection(match: Match) -> bool:  # noqa: PLR0911
    if not is_valid_routing_mode(match.kwargs["sparse_routing_mode"]):
        return False
    nodes = (
        "sparse_input",
        "sparse_q_weight_qdata",
        "sparse_q_weight_scale",
        "sparse_k_weight_qdata",
        "sparse_k_weight_scale",
        "sparse_v_weight_qdata",
        "sparse_v_weight_scale",
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
    if input_value.ndim != 3:
        return False
    input_features = sparse_piper_compile.static_int(input_value.shape[-1])
    q_weight = metadata["sparse_q_weight_qdata"]
    assert q_weight is not None
    output_features = sparse_piper_compile.static_int(q_weight.shape[0])
    if input_features is None or output_features is None or output_features % _HEAD_DIM:
        return False
    batch, sequence_length = input_value.shape[:2]
    heads = output_features // _HEAD_DIM
    if (
        input_value.dtype is not torch.bfloat16
        or input_value.device.type != "cuda"
        or (isinstance(sequence_length, int) and sequence_length < _TILE_ROWS)
    ):
        return False
    target = AcceleratorTarget.from_device(input_value.device)
    if not target.is_cuda_capability(12, 0):
        return False

    group_size = sparse_piper_compile.static_int(match.kwargs["sparse_group_size"])
    if input_features is None or group_size is None or group_size < 1:
        return False
    for prefix in ("sparse_q", "sparse_k", "sparse_v"):
        weight = metadata[f"{prefix}_weight_qdata"]
        scale = metadata[f"{prefix}_weight_scale"]
        assert weight is not None
        assert scale is not None
        if (
            weight.dtype is not torch.int8
            or tuple(weight.shape) != (heads * _HEAD_DIM, input_features)
            or scale.dtype is not torch.float32
            or tuple(scale.shape) != (heads * _HEAD_DIM, 1)
            or weight.device != input_value.device
            or scale.device != input_value.device
        ):
            return False

    return sparse_piper_compile.valid_sparse_piper_attention(
        match,
        batch=batch,
        sequence_length=sequence_length,
        heads=heads,
        device=input_value.device,
        head_dim=_HEAD_DIM,
        tile_rows=_TILE_ROWS,
    )


def _valid_sparse_piper_coarse_residual_projection(
    match: Match,
) -> bool:
    """Validate the extra operands and routing policy of a coarse residual."""
    if not _valid_sparse_piper_projection(match):
        return False
    return sparse_piper_compile.valid_sparse_piper_coarse_residual(match)


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
    sparse_head_keep_ratio_units: list[int],
    sparse_group_size: int,
    sparse_q_norm_epsilon: float,
    sparse_k_norm_epsilon: float,
    sparse_key_blocks: Argument,
    sparse_softmax_scale: float,
    sparse_routing_mode: int,
    sparse_block_lengths: torch.fx.Node | None = None,
    sparse_query_blocks: Argument | None = None,
    coarse_gate: torch.fx.Node | None = None,
    coarse_key_blocks: Argument | None = None,
    coarse_scale: float | None = None,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    mean_pool_routing = sparse_routing_mode == _MEAN_ROUTING
    input_value = preparation_sharing.tensor_metadata(sparse_input)
    assert input_value is not None
    batch, sequence_length = input_value.shape[:2]
    q_weight_value = preparation_sharing.tensor_metadata(sparse_q_weight_qdata)
    assert q_weight_value is not None
    heads = q_weight_value.shape[0] // _HEAD_DIM
    head_dim = _HEAD_DIM
    storage_sequence_length = _layout.padded_sequence_length(sequence_length)
    block_length_arguments = () if sparse_block_lengths is None else (sparse_block_lengths,)
    with graph.inserting_before(original):
        logical_sequence_length = graph.call_function(
            torch.ops.aten.sym_size.int,
            args=(sparse_input, 1),
        )
        logical_sequence_length.meta["val"] = sequence_length
        input_qdata, input_scale, _logical_dtype = _compile_fx.emit_prepared_input(
            graph,
            sparse_input,
            sparse_group_size,
            None,
            tuple(input_value.shape),
        )
        prepared_coarse_gate = coarse_gate
        if coarse_gate is not None:
            gate_projection = _semantic_gate_projection(
                coarse_gate,
                sparse_input,
                sparse_group_size,
            )
            if gate_projection is not None:
                gate_linear, gate_weight_qdata, gate_weight_scale, gate_bias = gate_projection
                prepared_gate_linear = graph.call_function(
                    torch.ops.piper_kernels.convrot_int8_linear_prepared.default,
                    args=(
                        input_qdata,
                        input_scale,
                        gate_weight_qdata,
                        gate_weight_scale,
                        gate_bias,
                        torch.bfloat16,
                    ),
                )
                prepared_gate_linear.meta = gate_linear.meta.copy()
                prepared_coarse_gate = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(prepared_gate_linear, coarse_gate.args[1]),
                )
                prepared_coarse_gate.meta = coarse_gate.meta.copy()
        query_values = (
            input_value.new_empty(
                (batch, heads, storage_sequence_length, head_dim),
                dtype=torch.int8,
            ),
            input_value.new_empty(
                (batch, heads, storage_sequence_length // _QUERY_SCALE_ROWS),
                dtype=torch.float32,
            ),
            input_value.new_empty(
                (batch, heads, storage_sequence_length // _TILE_ROWS, head_dim),
                dtype=torch.float32,
            ),
        )
        query, query_scale, query_summary = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_int8_sparse_piper_project_query.default,
            (
                input_qdata,
                input_scale,
                sparse_q_weight_qdata,
                sparse_q_weight_scale,
                sparse_q_norm_weight,
                sparse_cos,
                sparse_sin,
                sparse_q_norm_epsilon,
                sparse_softmax_scale,
                sparse_routing_mode,
                *block_length_arguments,
            ),
            query_values,
        )
        key_values = (
            input_value.new_empty(
                (batch, heads, storage_sequence_length, head_dim),
                dtype=torch.int8,
            ),
            input_value.new_empty(
                (batch, heads, storage_sequence_length // _TILE_ROWS),
                dtype=torch.float32,
            ),
            input_value.new_empty(
                (batch, heads, storage_sequence_length // _TILE_ROWS, head_dim),
                dtype=torch.float32,
            ),
            (
                input_value.new_empty((batch, heads, 0, head_dim), dtype=torch.float32)
                if mean_pool_routing
                else input_value.new_empty(
                    (batch, heads, storage_sequence_length // _TILE_ROWS, head_dim),
                    dtype=torch.float32,
                )
            ),
        )
        key_arguments = (
            input_qdata,
            input_scale,
            sparse_k_weight_qdata,
            sparse_k_weight_scale,
            sparse_k_norm_weight,
            sparse_cos,
            sparse_sin,
            sparse_k_norm_epsilon,
        )
        key, key_scale, key_summary, key_aux = linear_compile_fx.emit_tuple_result(
            graph,
            torch.ops.piper_kernels.convrot_int8_sparse_piper_project_key.default,
            (*key_arguments, sparse_routing_mode, *block_length_arguments),
            key_values,
        )
        input_mean = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_dequantized_input_mean.default,
            args=(input_qdata, input_scale, *block_length_arguments),
        )
        input_mean.meta["val"] = input_value.new_empty(
            (batch, input_value.shape[-1]),
            dtype=torch.float32,
        )
        value_values = (
            input_value.new_empty(
                (batch, heads, head_dim, storage_sequence_length),
                dtype=torch.int8,
            ),
            input_value.new_empty(
                (batch, heads, storage_sequence_length // _TILE_ROWS, 1),
                dtype=torch.float32,
            ),
            input_value.new_empty((batch, heads, head_dim), dtype=torch.float32),
        )
        with_coarse_residual = coarse_gate is not None
        if with_coarse_residual:
            value_values = (
                *value_values,
                input_value.new_empty(
                    (batch, heads, storage_sequence_length // _TILE_ROWS, head_dim),
                    dtype=torch.float32,
                ),
            )
        value_projection = linear_compile_fx.emit_tuple_result(
            graph,
            (
                torch.ops.piper_kernels.convrot_int8_sparse_piper_project_value_with_block_means.default
                if with_coarse_residual
                else torch.ops.piper_kernels.convrot_int8_sparse_piper_project_value.default
            ),
            (
                input_qdata,
                input_scale,
                input_mean,
                sparse_v_weight_qdata,
                sparse_v_weight_scale,
                *block_length_arguments,
            ),
            value_values,
        )
        value, value_scale_multiplier, value_mean = value_projection[:3]
        attention_arguments = (
            query,
            query_scale,
            query_summary,
            key,
            key_scale,
            key_summary,
            key_aux,
            value,
            value_scale_multiplier,
            value_mean,
        )
        replacement = sparse_piper_compile.emit_quantized_sparse_piper_attention(
            graph,
            attention_arguments,
            head_keep_ratio_units=sparse_head_keep_ratio_units,
            sparse_key_blocks=sparse_key_blocks,
            logical_sequence_length=logical_sequence_length,
            routing_mode=sparse_routing_mode,
            block_lengths=sparse_block_lengths,
            sparse_query_blocks=sparse_query_blocks,
            block_mean=value_projection[3] if with_coarse_residual else None,
            coarse_gate=prepared_coarse_gate,
            coarse_scale=coarse_scale,
            coarse_key_blocks=coarse_key_blocks,
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


_patterns = PatternMatcherPass("convrot_int8_sparse_piper_projection")
for _with_coarse in (True, False):
    for _with_block_lengths in (True, False):
        for _with_sparse_query_blocks in (True, False):
            register_graph_pattern(
                sparse_piper_pattern.sparse_piper_projection_pattern(
                    _linear_pattern,
                    with_block_lengths=_with_block_lengths,
                    with_coarse=_with_coarse,
                    with_sparse_query_blocks=_with_sparse_query_blocks,
                ),
                extra_check=(
                    _valid_sparse_piper_coarse_residual_projection
                    if _with_coarse
                    else _valid_sparse_piper_projection
                ),
                pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
            )(_replace_sparse_piper_projection)


def _fold_sparse_piper_projection(graph: torch.fx.Graph) -> bool:
    """Replace one compatible materialized sparse-attention projection region."""
    changed = _patterns.apply(graph) > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold compatible ConvRot INT8 projections into sparse Piper preparation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            _fold_sparse_piper_projection(graph)
            _output_compile._fold_attention_output(graph)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            _source_files(),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def convrot_int8_sparse_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install sparse fusion before ordinary ConvRot INT8 graph optimizations."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (compile_pass, convrot_int8_compile.compile_pass),
    )


__all__ = ["convrot_int8_sparse_piper_compile_options"]
