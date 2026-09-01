"""Compiler folding for a bounded-workspace NVFP4 SwiGLU FFN."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass

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

from piper_kernels.fusions.swiglu_ffn import _compile as swiglu_ffn_compile
from piper_kernels.fusions.swiglu_ffn import _pattern as swiglu_ffn_pattern
from piper_kernels.fusions.swiglu_ffn import triton as swiglu_ffn_triton
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _compile as nvfp4_compile
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation

from . import _core
from . import triton as ffn_backend

_COMPILE_PASS_VERSION = "nvfp4-swiglu-ffn-compile-v1"


@dataclass(frozen=True, slots=True)
class _MatchedFfn:
    up: nvfp4_compile_fx.SemanticLinearNodes
    down: nvfp4_compile_fx.PreparedLinearNodes
    down_activation_per_tensor_scale: torch.fx.Node | None
    down_dynamic_activation_scale: bool

    @classmethod
    def from_match(cls, match: Match) -> _MatchedFfn | None:
        up_calls = [
            node
            for node in match.nodes
            if node.op == "call_function"
            and node.target == torch.ops.piper_kernels.nvfp4_linear.default
        ]
        down_calls = [
            node
            for node in match.nodes
            if node.op == "call_function"
            and node.target == torch.ops.piper_kernels.nvfp4_linear_prepared.default
        ]
        if len(up_calls) != 1 or len(down_calls) != 1:
            return None
        up = nvfp4_compile_fx.SemanticLinearNodes.from_call(up_calls[0])
        down = nvfp4_compile_fx.PreparedLinearNodes.from_call(down_calls[0])
        if up is None or down is None:
            return None
        prepared_getitem = down.input_qdata
        if (
            prepared_getitem.op != "call_function"
            or prepared_getitem.target != operator.getitem
            or len(prepared_getitem.args) != 2
            or prepared_getitem.args[1] != 0
            or not isinstance(prepared_getitem.args[0], torch.fx.Node)
        ):
            return None
        prepared = prepared_getitem.args[0]
        if (
            prepared.op != "call_function"
            or prepared.target != torch.ops.piper_kernels.nvfp4_prepare_input.default
            or prepared.kwargs
            or len(prepared.args) != 4
            or prepared.args[0] is not up_calls[0]
            or prepared.args[3] != "swiglu"
        ):
            return None
        activation_scale, dynamic = prepared.args[1:3]
        if (
            activation_scale is not None and not isinstance(activation_scale, torch.fx.Node)
        ) or not isinstance(dynamic, bool):
            return None
        return cls(up, down, activation_scale, dynamic)

    def arguments(self) -> tuple[Argument, ...]:
        """Return custom-op operands in canonical up/down order."""
        return (
            self.up.input,
            self.up.weight_qdata,
            self.up.weight_scale,
            self.up.weight_per_tensor_scale,
            self.up.activation_per_tensor_scale,
            self.up.bias,
            self.up.dynamic_activation_scale,
            self.down.weight_qdata,
            self.down.weight_scale,
            self.down.weight_per_tensor_scale,
            self.down_activation_per_tensor_scale,
            self.down.bias,
            self.down_dynamic_activation_scale,
        )


def _normalized_ffn_pattern(*, reshape_output: bool) -> CallFunction:
    """Match the stable graph produced by NVFP4 activation folding."""
    packed = CallFunction(
        torch.ops.piper_kernels.nvfp4_linear.default,
        KeywordArg("ffn_input"),
        KeywordArg("up_weight_qdata"),
        KeywordArg("up_weight_scale"),
        KeywordArg("up_weight_per_tensor_scale"),
        KeywordArg("up_activation_per_tensor_scale"),
        KeywordArg("up_bias"),
        KeywordArg("up_dynamic_activation_scale"),
        _users=1,
    )
    prepared = CallFunction(
        torch.ops.piper_kernels.nvfp4_prepare_input.default,
        packed,
        KeywordArg("down_activation_per_tensor_scale"),
        KeywordArg("down_dynamic_activation_scale"),
        "swiglu",
        _users=3,
    )
    projected_arguments = (
        CallFunction(operator.getitem, prepared, 0, _users=1),
        CallFunction(operator.getitem, prepared, 1, _users=1),
        CallFunction(operator.getitem, prepared, 2, _users=1),
        KeywordArg("down_weight_qdata"),
        KeywordArg("down_weight_scale"),
        KeywordArg("down_weight_per_tensor_scale"),
        KeywordArg("down_bias"),
        KeywordArg("logical_dtype"),
    )
    projected = (
        CallFunction(
            torch.ops.piper_kernels.nvfp4_linear_prepared.default,
            *projected_arguments,
            _users=1,
        )
        if reshape_output
        else CallFunction(
            torch.ops.piper_kernels.nvfp4_linear_prepared.default,
            *projected_arguments,
        )
    )
    if not reshape_output:
        return projected
    return CallFunction(
        torch.ops.aten.reshape.default,
        projected,
        KeywordArg("output_shape"),
    )


def _valid_normalized_ffn(match: Match) -> bool:
    operands = _MatchedFfn.from_match(match)
    if operands is None:
        return False
    validated_up = nvfp4_compile_fx.validated_semantic_linear(
        operands.up,
        "NVFP4 FFN compiler up projection",
    )
    down_shape = nvfp4_compile_fx.validated_prepared_linear(
        operands.down,
        "NVFP4 FFN compiler down projection",
    )
    if validated_up is None or down_shape is None:
        return False
    input_value, up_shape = validated_up
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    if output_value is None:
        return False
    down_activation_scale = (
        None
        if operands.down_activation_per_tensor_scale is None
        else preparation_sharing.tensor_metadata(operands.down_activation_per_tensor_scale)
    )
    if operands.down_activation_per_tensor_scale is not None and down_activation_scale is None:
        return False
    try:
        nvfp4_validation.validate_activation_scale(
            down_activation_scale,
            operands.down_dynamic_activation_scale,
            input_value.device,
            "NVFP4 FFN compiler down projection",
        )
    except ValueError:
        return False
    return bool(
        input_value.dtype is torch.bfloat16
        and operands.down.logical_dtype is input_value.dtype
        and output_value.dtype is input_value.dtype
        and output_value.device == input_value.device
        and output_value.ndim == input_value.ndim
        and all(
            preparation_sharing.dimension_key(output_dimension)
            == preparation_sharing.dimension_key(input_dimension)
            for output_dimension, input_dimension in zip(
                output_value.shape[:-1],
                input_value.shape[:-1],
                strict=True,
            )
        )
        and preparation_sharing.dimension_key(output_value.shape[-1])
        == preparation_sharing.dimension_key(down_shape.output_features)
        and preparation_sharing.dimension_key(up_shape.rows)
        == preparation_sharing.dimension_key(down_shape.rows)
        and preparation_sharing.dimension_key(up_shape.output_features)
        == preparation_sharing.dimension_key(2 * down_shape.input_features)
    )


def _valid_gated_updates(match: Match) -> bool:
    return swiglu_ffn_compile.valid_gated_updates(match, _valid_normalized_ffn)


def _replace_normalized_ffn(match: Match, **_unused: object) -> None:
    original = match.output_node()
    graph = match.graph
    operands = _MatchedFfn.from_match(match)
    assert operands is not None
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.nvfp4_swiglu_ffn.default,
            args=(*operands.arguments(), ffn_backend._DEFAULT_CHUNK_ROWS),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


def _replace_normalized_ffn_gated_updates(match: Match, **_unused: object) -> None:
    original = match.output_node()
    graph = match.graph
    operands = _MatchedFfn.from_match(match)
    assert operands is not None
    python_indexing = swiglu_ffn_compile.uses_python_indexing(match)
    with graph.inserting_before(original):
        mutation = graph.call_function(
            torch.ops.piper_kernels.nvfp4_swiglu_ffn_gated_updates_.default,
            args=(
                *operands.arguments(),
                match.kwargs["base"],
                match.kwargs["reusable_update"],
                match.kwargs["update_gate"],
                match.kwargs["ffn_gate"],
                match.kwargs["gate_indices"],
                python_indexing,
                ffn_backend._DEFAULT_CHUNK_ROWS,
            ),
        )
    mutation.meta["val"] = None
    reusable_update = match.kwargs["reusable_update"]
    assert isinstance(reusable_update, torch.fx.Node)
    original.replace_all_uses_with(reusable_update)
    match.erase_nodes()


_gated_updates_patterns = PatternMatcherPass("nvfp4_swiglu_ffn_gated_updates")
_patterns = PatternMatcherPass("nvfp4_swiglu_ffn")
for _reshape_output in (False, True):
    _ffn_pattern = _normalized_ffn_pattern(reshape_output=_reshape_output)
    for _use_aten_index in (False, True):
        register_graph_pattern(
            swiglu_ffn_pattern.gated_updates_pattern(
                _ffn_pattern,
                use_aten_index=_use_aten_index,
            ),
            extra_check=_valid_gated_updates,
            pass_dict=_gated_updates_patterns,  # pyright: ignore[reportArgumentType]
        )(_replace_normalized_ffn_gated_updates)
    register_graph_pattern(
        _ffn_pattern,
        extra_check=_valid_normalized_ffn,
        pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
    )(_replace_normalized_ffn)


def _fold_chunked_ffn(graph: torch.fx.Graph) -> bool:
    changes = _gated_updates_patterns.apply(graph)
    changes += _patterns.apply(graph)
    changed = changes > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold normalized NVFP4 SwiGLU FFNs after ordinary NVFP4 rewriting."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            _fold_chunked_ffn(graph)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            tuple(
                file_name
                for file_name in (
                    __file__,
                    _core.__file__,
                    ffn_backend.__file__,
                    nvfp4_compile_fx.__file__,
                    nvfp4_validation.__file__,
                    swiglu_ffn_compile.__file__,
                    swiglu_ffn_pattern.__file__,
                    swiglu_ffn_triton.__file__,
                )
                if file_name is not None
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def nvfp4_swiglu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install chunked FFN folding immediately after ordinary NVFP4 folding."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (nvfp4_compile.compile_pass, compile_pass),
    )


__all__ = ["nvfp4_swiglu_ffn_compile_options"]
