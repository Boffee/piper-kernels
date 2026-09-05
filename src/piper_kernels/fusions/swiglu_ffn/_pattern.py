"""Projection-independent graph grammar for SwiGLU FFN gated updates."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg

from piper_kernels.linear import _input_activation_compile as input_activation_compile

type ProjectionPattern = Callable[[object, str, int | None], CallFunction]


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


def _indexed_gate_pattern(
    *,
    gate_name: str,
    indices_name: str,
    use_aten_index: bool,
) -> CallFunction:
    gate = KeywordArg(gate_name)
    indices = KeywordArg(indices_name)
    if use_aten_index:
        return CallFunction(
            torch.ops.aten.index.Tensor,
            gate,
            [indices],
            _users=1,
        )
    return CallFunction(
        torch.ops.aten.index_select.default,
        gate,
        0,
        indices,
        _users=1,
    )


def gated_updates_pattern(
    ffn: CallFunction,
    *,
    use_aten_index: bool,
) -> CallFunction:
    """Wrap an exclusive FFN pattern in two indexed gated updates."""
    update_gate = _indexed_gate_pattern(
        gate_name="update_gate",
        indices_name="gate_indices",
        use_aten_index=use_aten_index,
    )
    gated_update = CallFunction(
        torch.ops.aten.mul.Tensor,
        update_gate,
        KeywordArg("reusable_update"),
        _users=1,
    )
    hidden = CallFunction(
        torch.ops.aten.add.Tensor,
        KeywordArg("base"),
        gated_update,
        _users=2,
    )
    ffn_gate = _indexed_gate_pattern(
        gate_name="ffn_gate",
        indices_name="gate_indices",
        use_aten_index=use_aten_index,
    )
    gated_ffn = CallFunction(torch.ops.aten.mul.Tensor, ffn_gate, ffn, _users=1)
    return CallFunction(
        torch.ops.aten.add.Tensor,
        hidden,
        gated_ffn,
    )


__all__ = ["ProjectionPattern", "gated_updates_pattern", "semantic_ffn_pattern"]
