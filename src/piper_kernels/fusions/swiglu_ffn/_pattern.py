"""Projection-independent graph grammar for SwiGLU FFN gated updates."""

from __future__ import annotations

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg


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
    """Wrap an exclusive FFN pattern in H3's two indexed gated updates."""
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


__all__ = ["gated_updates_pattern"]
