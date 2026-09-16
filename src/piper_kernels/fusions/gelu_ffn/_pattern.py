"""Projection-independent graph grammar for GELU feed-forward networks."""

from __future__ import annotations

from torch._inductor.pattern_matcher import CallFunction, KeywordArg

from piper_kernels.fusions.ffn._pattern import ProjectionPattern
from piper_kernels.linear import _input_activation_compile as input_activation_compile


def semantic_ffn_pattern(
    up_projection: ProjectionPattern,
    down_projection: ProjectionPattern,
    *,
    promote_input: bool,
) -> CallFunction:
    """Build ``down(gelu_tanh(up(input)))`` independent of projection storage."""
    input = KeywordArg("ffn_input")  # noqa: A001 - graph operand name
    up_users = 1 if promote_input else 4
    up = up_projection(input, "up", up_users)
    activated = input_activation_compile.gelu_tanh_activation_pattern(
        up,
        promote_input=promote_input,
    )
    return down_projection(activated, "down", None)


__all__ = ["semantic_ffn_pattern"]
