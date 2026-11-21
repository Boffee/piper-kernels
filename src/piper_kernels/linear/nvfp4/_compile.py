"""NVFP4 inference graph optimizations for Inductor."""

from __future__ import annotations

import torch

from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _input_activation_compile as input_activation_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear import _projection_views as projection_views

from . import (
    _compile_fx,
    _ops,  # noqa: F401 - registers nvfp4_linear
)

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
            operands.high_first,
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
            high_first=operands.high_first,
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


def _fold_gelu_tanh_inputs(graph: torch.fx.Graph) -> bool:
    """Fold exclusive GELU-tanh decompositions into prepared NVFP4 inputs."""
    changed = False
    for node in list(graph.nodes):
        operands = _compile_fx.SemanticLinearNodes.from_call(node)
        if operands is None:
            continue
        gelu = input_activation_compile.match_gelu_tanh_input(operands.input)
        if gelu is None:
            continue
        source, logical_dtype = gelu
        validated = _compile_fx.validated_semantic_linear(
            operands,
            "NVFP4 activated compiler linear",
        )
        if validated is None:
            continue
        _, shape = validated
        if not input_activation_compile.valid_gelu_tanh_input(
            source,
            logical_dtype,
            input_features=shape.input_features,
        ):
            continue
        with graph.inserting_before(node):
            prepared = _compile_fx.emit_prepared_input(
                graph,
                source,
                operands.activation_per_tensor_scale,
                operands.dynamic_activation_scale,
                "gelu_tanh",
                operands.high_first,
            )
            replacement = _compile_fx.emit_prepared_linear(graph, prepared, operands)
        replacement.meta = node.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


def fold_linear(graph: torch.fx.Graph) -> bool:
    """Normalize projection views, coalesce input casts, fold GELU-tanh, and share preparation."""
    changed = projection_views.normalize_projection_views(graph)
    changed = linear_fx.coalesce_input_casts(graph) or changed
    changed = _fold_gelu_tanh_inputs(graph) or changed
    changed = preparation_sharing.share_preparation(graph, _PREPARATION_RULES) or changed
    return changed


__all__ = ["fold_linear"]
