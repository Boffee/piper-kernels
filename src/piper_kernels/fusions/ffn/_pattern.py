"""Projection-independent graph grammar for feed-forward network updates."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg

from piper_kernels.fusions.residual_updates import _gates as residual_gates

type ProjectionPattern = Callable[[object, str, int | None], CallFunction]


def _indexed_gate_pattern(
    *,
    gate_name: str,
    indices_name: str,
    selection: torch._ops.OpOverload,
) -> CallFunction:
    arguments = residual_gates.selection_arguments(
        selection, KeywordArg(gate_name), KeywordArg(indices_name)
    )
    return CallFunction(selection, *arguments, _users=1)


def indexed_gated_updates_pattern(
    ffn_output: CallFunction,
    *,
    selection: torch._ops.OpOverload,
) -> CallFunction:
    """Wrap an exclusive FFN pattern in two gated updates whose gates ``selection`` picks.

    ``selection`` is one of ``residual_updates._gates.GATE_SELECTIONS``.
    """
    update_gate = _indexed_gate_pattern(
        gate_name="update_gate",
        indices_name="gate_indices",
        selection=selection,
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
        selection=selection,
    )
    gated_ffn = CallFunction(torch.ops.aten.mul.Tensor, ffn_gate, ffn_output, _users=1)
    return CallFunction(
        torch.ops.aten.add.Tensor,
        hidden,
        gated_ffn,
    )


__all__ = ["ProjectionPattern", "indexed_gated_updates_pattern"]
