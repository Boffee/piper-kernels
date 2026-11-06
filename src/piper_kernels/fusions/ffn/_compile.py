"""Projection-independent compiler rules for indexed FFN updates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import torch
from torch._inductor.pattern_matcher import Match
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates import _graph as residual_graph
from piper_kernels.linear import _preparation_sharing as preparation_sharing

type FfnPatternValidator = Callable[[Match], bool]

SOURCE_FILES = (__file__, residual_graph.__file__)


class UpDownChunkRows(Protocol):
    """An up/down FFN's chunk-size policy, from its input and weight metadata."""

    def __call__(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        up_weight_qdata: torch.Tensor,
        down_weight_qdata: torch.Tensor,
        /,
        *,
        adds_residual: bool,
    ) -> int: ...


def _metadata(match: Match, name: str) -> torch.Tensor | None:
    argument = match.kwargs[name]
    return (
        preparation_sharing.tensor_metadata(argument)
        if isinstance(argument, torch.fx.Node)
        else None
    )


def _shape_matches(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.ndim == right.ndim and all(
        preparation_sharing.dimension_key(left_dimension)
        == preparation_sharing.dimension_key(right_dimension)
        for left_dimension, right_dimension in zip(left.shape, right.shape, strict=True)
    )


def valid_indexed_gated_updates(
    match: Match,
    valid_ffn_pattern: FfnPatternValidator,
) -> bool:
    """Validate indexed gated updates and prove their intermediate safe to reuse."""
    if not valid_ffn_pattern(match):
        return False
    input_value = _metadata(match, "ffn_input")
    base = _metadata(match, "base")
    reusable_update = _metadata(match, "reusable_update")
    update_gate = _metadata(match, "update_gate")
    gate_indices = _metadata(match, "gate_indices")
    ffn_gate = _metadata(match, "ffn_gate")
    output = preparation_sharing.tensor_metadata(match.output_node())
    reusable_update_node = match.kwargs["reusable_update"]
    if (
        input_value is None
        or base is None
        or reusable_update is None
        or update_gate is None
        or gate_indices is None
        or ffn_gate is None
        or output is None
        or not isinstance(reusable_update_node, torch.fx.Node)
    ):
        return False
    # Both adds share ``base``, which takes the FFN input's dtype and the result's shape.
    return bool(
        _shape_matches(base, output)
        and base.dtype is input_value.dtype
        and base.device == input_value.device
        and residual_graph.supports_residual_add(base, reusable_update, update_gate, gate_indices)
        and residual_graph.supports_residual_add(base, None, ffn_gate, gate_indices)
        and preparation_sharing.operator_returns_fresh_tensor(reusable_update_node)
        and len(reusable_update_node.users) == 1
    )


def fold_residual_adds(
    graph: torch.fx.Graph,
    ffn: torch._ops.OpOverload,
    residual_add: torch._ops.OpOverload,
    chunk_rows: UpDownChunkRows | None = None,
) -> int:
    """Compute each ``ffn`` call that only a residual add reads inside that add.

    ``residual_add`` takes ``ffn``'s arguments by name, with the add's ``base``,
    ``gate`` as ``ffn_gate``, ``gate_indices``, and ``python_indexing``. An up/down FFN
    whose chunk size depends on whether it adds into a residual passes that policy as
    ``chunk_rows``; otherwise the FFN's chunk size carries over. Returns how many adds
    it replaced.
    """

    def arguments(operands: dict[str, Argument], add: dict[str, Argument]) -> dict[str, Argument]:
        values = {
            **operands,
            "base": add["base"],
            "ffn_gate": add["gate"],
            "gate_indices": add["gate_indices"],
            "python_indexing": add["python_indexing"],
        }
        if chunk_rows is not None:
            # Post-grad graphs carry metadata on every tensor operand.
            input_value, up_weight, down_weight = (
                preparation_sharing.tensor_metadata(cast(torch.fx.Node, values[name]))
                for name in ("input", "up_weight_qdata", "down_weight_qdata")
            )
            assert input_value is not None
            assert up_weight is not None
            assert down_weight is not None
            values["chunk_rows"] = chunk_rows(
                input_value, up_weight, down_weight, adds_residual=True
            )
        return values

    return residual_graph.compute_inside_adds(graph, ffn, residual_add, arguments)


def uses_python_indexing(match: Match) -> bool:
    """Return whether a matched update uses Python indexing semantics."""
    return any(
        node.op == "call_function" and node.target == torch.ops.aten.index.Tensor
        for node in match.nodes
    )


__all__ = [
    "SOURCE_FILES",
    "FfnPatternValidator",
    "UpDownChunkRows",
    "fold_residual_adds",
    "uses_python_indexing",
    "valid_indexed_gated_updates",
]
