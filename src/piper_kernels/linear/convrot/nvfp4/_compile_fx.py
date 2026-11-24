"""FX helpers for ConvRot NVFP4 compiler passes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation
from piper_kernels.weights.convrot._rotation import validate_group_size


@dataclass(frozen=True, slots=True)
class SemanticLinearNodes:
    """A canonical NVFP4 semantic linear plus its ConvRot group."""

    linear: nvfp4_compile_fx.SemanticLinearNodes
    group_size: int

    @classmethod
    def from_call(cls, node: torch.fx.Node) -> SemanticLinearNodes | None:
        """Parse one schema-bound ConvRot NVFP4 semantic-linear call."""
        bound = linear_fx.bind_call(node, torch.ops.piper_kernels.convrot_nvfp4_linear.default)
        if bound is None or bound.get("weight_only") is True:
            return None
        group_size = bound["group_size"]
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            return None
        linear = nvfp4_compile_fx.SemanticLinearNodes.from_values(
            tuple(
                bound[name]
                for name in (
                    "input",
                    "weight_qdata",
                    "weight_scale",
                    "weight_per_tensor_scale",
                    "activation_per_tensor_scale",
                    "bias",
                    "dynamic_activation_scale",
                    "high_first",
                )
            )
        )
        return None if linear is None else cls(linear, group_size)


def validated_semantic_linear(
    operands: SemanticLinearNodes,
    name: str,
) -> tuple[torch.Tensor, nvfp4_validation.LinearShape] | None:
    """Validate semantic NVFP4 metadata and ConvRot-specific invariants."""
    try:
        validate_group_size(operands.group_size)
    except ValueError:
        return None
    validated = nvfp4_compile_fx.validated_semantic_linear(operands.linear, name)
    if validated is None:
        return None
    input_value, shape = validated
    if (
        input_value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not isinstance(shape.input_features, int)
        or shape.input_features % operands.group_size
    ):
        return None
    return input_value, shape


__all__ = [
    "SemanticLinearNodes",
    "validated_semantic_linear",
]
