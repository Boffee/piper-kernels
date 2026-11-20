"""ConvRot NVFP4 inference graph optimizations for Inductor."""

from __future__ import annotations

import torch

from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear import _projection_views as projection_views
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import (
    _ops as nvfp4_ops,  # noqa: F401 - registers nvfp4_linear_prepared
)

from . import (
    _compile_fx,
    _ops,  # noqa: F401 - registers convrot_nvfp4_linear
)

type _PreparedInputNodes = _compile_fx.PreparedInputNodes


class _PreparationRule:
    """Describe how compatible ConvRot NVFP4 linears share preparation."""

    linear_target = torch.ops.piper_kernels.convrot_nvfp4_linear.default

    def match_key(
        self,
        node: torch.fx.Node,
    ) -> preparation_sharing.PreparationMatchKey | None:
        operands = _compile_fx.SemanticLinearNodes.from_call(node)
        if operands is None:
            return None
        validated = _compile_fx.validated_semantic_linear(
            operands,
            "ConvRot NVFP4 compiler linear",
        )
        if validated is None:
            return None
        input_value, shape = validated
        # Exact FX identity prevents grouping across functionalized mutations.
        family_key = (
            operands.linear.input,
            operands.group_size,
            preparation_sharing.dimension_key(shape.input_features),
            input_value.dtype,
        )
        preparation_key = (
            operands.linear.activation_per_tensor_scale,
            operands.linear.dynamic_activation_scale,
            operands.linear.high_first,
        )
        return family_key, preparation_key

    def prepare(
        self,
        graph: torch.fx.Graph,
        first: torch.fx.Node,
    ) -> _PreparedInputNodes:
        operands = _compile_fx.SemanticLinearNodes.from_call(first)
        assert operands is not None
        return _compile_fx.emit_prepared_input(graph, operands)

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: _PreparedInputNodes,
    ) -> torch.fx.Node:
        operands = _compile_fx.SemanticLinearNodes.from_call(node)
        assert operands is not None
        return nvfp4_compile_fx.emit_prepared_linear(graph, prepared, operands.linear)


_PREPARATION_RULES = (_PreparationRule(),)


def fold_linear(graph: torch.fx.Graph) -> bool:
    """Normalize projection views and share compatible ConvRot NVFP4 preparation."""
    changed = projection_views.normalize_projection_views(graph)
    return preparation_sharing.share_preparation(graph, _PREPARATION_RULES) or changed


__all__ = ["fold_linear"]
