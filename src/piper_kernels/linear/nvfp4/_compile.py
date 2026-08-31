"""NVFP4 inference graph optimizations for Inductor."""

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

from piper_kernels.linear import _input_activation_compile as input_activation_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _compile_fx, _ops, _validation

_COMPILE_PASS_VERSION = "nvfp4-compile-v2"
type _PreparedInputNodes = _compile_fx.PreparedInputNodes


class _PreparationRule:
    """Describe how compatible semantic NVFP4 linears share prepared inputs."""

    linear_target = torch.ops.piper_kernels.nvfp4_linear.default

    def match_key(
        self,
        node: torch.fx.Node,
    ) -> preparation_sharing.PreparationMatchKey | None:
        operands = _compile_fx.SemanticLinearNodes.from_call(node)
        if operands is None:
            return None
        validated = _compile_fx.validated_semantic_linear(operands, "NVFP4 compiler linear")
        if validated is None:
            return None
        input_value, shape = validated
        # Exact FX identity prevents grouping across functionalized mutations.
        family_key = (
            operands.input,
            preparation_sharing.dimension_key(shape.input_features),
            input_value.dtype,
        )
        preparation_key = (
            operands.activation_per_tensor_scale,
            operands.dynamic_activation_scale,
        )
        return family_key, preparation_key

    def prepare(
        self,
        graph: torch.fx.Graph,
        first: torch.fx.Node,
    ) -> _PreparedInputNodes:
        operands = _compile_fx.SemanticLinearNodes.from_call(first)
        assert operands is not None
        return _compile_fx.emit_prepared_input(
            graph,
            operands.input,
            operands.activation_per_tensor_scale,
            operands.dynamic_activation_scale,
        )

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: _PreparedInputNodes,
    ) -> torch.fx.Node:
        operands = _compile_fx.SemanticLinearNodes.from_call(node)
        assert operands is not None
        return _compile_fx.emit_prepared_linear(graph, prepared, operands)


_PREPARATION_RULES = (_PreparationRule(),)


_input_activation_patterns = PatternMatcherPass("nvfp4_input_activations")


def _linear_pattern(input_pattern: CallFunction) -> CallFunction:
    return CallFunction(
        torch.ops.piper_kernels.nvfp4_linear.default,
        input_pattern,
        KeywordArg("weight_qdata"),
        KeywordArg("weight_scale"),
        KeywordArg("weight_per_tensor_scale"),
        KeywordArg("activation_per_tensor_scale"),
        KeywordArg("bias"),
        KeywordArg("dynamic_activation_scale"),
    )


def _activation_input_features(match: Match) -> int | torch.SymInt | None:
    operands = _compile_fx.SemanticLinearNodes.from_call(match.output_node())
    if operands is None:
        return None
    validated = _compile_fx.validated_semantic_linear(
        operands,
        "NVFP4 activated compiler linear",
    )
    if validated is None:
        return None
    _, shape = validated
    return shape.input_features


def _valid_packed_swiglu(match: Match, *, promote_gate: bool | None) -> bool:
    input_features = _activation_input_features(match)
    return bool(
        input_features is not None
        and input_activation_compile.valid_packed_swiglu(
            match,
            promote_gate=promote_gate,
            input_features=input_features,
        )
    )


def _valid_gelu_tanh(match: Match, *, promote_input: bool) -> bool:
    input_features = _activation_input_features(match)
    return bool(
        input_features is not None
        and input_activation_compile.valid_gelu_tanh(
            match,
            promote_input=promote_input,
            input_features=input_features,
        )
    )


def _replace_input_activation_and_linear(
    match: Match,
    input_node: torch.fx.Node,
    activation_fn: str,
) -> None:
    original = match.output_node()
    graph = match.graph
    operands = _compile_fx.SemanticLinearNodes.from_call(original)
    assert operands is not None
    with graph.inserting_before(original):
        prepared = _compile_fx.emit_prepared_input(
            graph,
            input_node,
            operands.activation_per_tensor_scale,
            operands.dynamic_activation_scale,
            activation_fn,
        )
        replacement = _compile_fx.emit_prepared_linear(graph, prepared, operands)
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


def _replace_packed_swiglu(
    match: Match,
    packed: torch.fx.Node,
    **_unused: object,
) -> None:
    _replace_input_activation_and_linear(
        match,
        packed,
        "swiglu",
    )


for _promote_gate in (None, False, True):
    for _reverse_multiply in (False, True):
        register_graph_pattern(
            input_activation_compile.packed_swiglu_pattern(
                _linear_pattern,
                promote_gate=_promote_gate,
                reverse_multiply=_reverse_multiply,
            ),
            extra_check=lambda match, promote_gate=_promote_gate: _valid_packed_swiglu(
                match,
                promote_gate=promote_gate,
            ),
            pass_dict=_input_activation_patterns,  # pyright: ignore[reportArgumentType]
        )(_replace_packed_swiglu)


def _replace_gelu_tanh(
    match: Match,
    input: torch.fx.Node,  # noqa: A002 - pattern keyword
    **_unused: object,
) -> None:
    _replace_input_activation_and_linear(
        match,
        input,
        "gelu_tanh",
    )


for _promote_input in (False, True):
    register_graph_pattern(
        input_activation_compile.gelu_tanh_pattern(
            _linear_pattern,
            promote_input=_promote_input,
        ),
        extra_check=lambda match, promote_input=_promote_input: _valid_gelu_tanh(
            match,
            promote_input=promote_input,
        ),
        pass_dict=_input_activation_patterns,  # pyright: ignore[reportArgumentType]
    )(_replace_gelu_tanh)


def _fold_input_activations(graph: torch.fx.Graph) -> bool:
    changed = _input_activation_patterns.apply(graph) > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold input activations before sharing compatible NVFP4 preparation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        _fold_input_activations(graph)
        preparation_sharing.share_preparation(graph, _PREPARATION_RULES)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            (
                __file__,
                input_activation_compile.__file__,
                preparation_sharing.__file__,
                _compile_fx.__file__,
                _ops.__file__,
                _validation.__file__,
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def nvfp4_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return Inductor options that share compatible NVFP4 preparation."""
    return preparation_sharing.add_post_grad_pass(options, compile_pass)


__all__ = ["nvfp4_compile_options"]
