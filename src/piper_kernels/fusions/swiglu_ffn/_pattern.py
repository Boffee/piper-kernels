"""Projection-independent graph grammar for SwiGLU feed-forward networks."""

from __future__ import annotations

from torch._inductor.pattern_matcher import CallFunction, KeywordArg

from piper_kernels.fusions.ffn._pattern import ProjectionPattern
from piper_kernels.linear import _input_activation_compile as input_activation_compile


def semantic_ffn_pattern(
    gate_projection: ProjectionPattern,
    value_projection: ProjectionPattern,
    down_projection: ProjectionPattern,
    *,
    promote_gate: bool | None,
    reverse_multiply: bool,
) -> CallFunction:
    """Build a semantic gate/value SwiGLU FFN independent of projection storage."""
    input = KeywordArg("ffn_input")  # noqa: A001 - graph operand name
    gate_users = 2 if promote_gate is False else 1
    gate = gate_projection(input, "gate", gate_users)
    value = value_projection(input, "value", 1)
    activated = input_activation_compile.swiglu_product_pattern(
        value,
        gate,
        promote_gate=promote_gate,
        reverse_multiply=reverse_multiply,
    )
    return down_projection(activated, "down", None)


__all__ = ["semantic_ffn_pattern"]
