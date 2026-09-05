"""ConvRot NVFP4 inference graph optimizations for Inductor."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)

from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot import _rotation as convrot_rotation
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation

from . import _compile_fx, _ops
from . import triton as convrot_nvfp4_triton

_COMPILE_PASS_VERSION = "convrot-nvfp4-compile-v4"
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


class _CompilePass(CustomInferenceAwareGraphPass):
    """Share compatible ConvRot NVFP4 input preparation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            preparation_sharing.share_preparation(graph, _PREPARATION_RULES)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            (
                __file__,
                preparation_sharing.__file__,
                convrot_rotation.__file__,
                _compile_fx.__file__,
                _ops.__file__,
                convrot_nvfp4_triton.__file__,
                nvfp4_compile_fx.__file__,
                nvfp4_layout.__file__,
                nvfp4_ops.__file__,
                nvfp4_validation.__file__,
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def convrot_nvfp4_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return Inductor options that share ConvRot NVFP4 preparation."""
    return preparation_sharing.add_post_grad_pass(options, compile_pass)


__all__ = ["convrot_nvfp4_compile_options"]
