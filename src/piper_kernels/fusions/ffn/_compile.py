"""Projection-independent compiler rules for indexed FFN updates."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch._inductor.pattern_matcher import Match
from torch.fx.node import Argument

from piper_kernels.fusions.residual_updates import _graph as residual_graph
from piper_kernels.linear import _preparation_sharing as preparation_sharing

type FfnPatternValidator = Callable[[Match], bool]
type UpDownChunkRows = Callable[..., int]

SOURCE_FILES = (__file__, residual_graph.__file__)


def _metadata(match: Match, name: str) -> torch.Tensor | None:
    argument = match.kwargs[name]
    return (
        preparation_sharing.tensor_metadata(argument)
        if isinstance(argument, torch.fx.Node)
        else None
    )


def _dimension_matches(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


def _shape_matches(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.ndim == right.ndim and all(
        _dimension_matches(left_dimension, right_dimension)
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
    if any(
        value is None
        for value in (
            input_value,
            base,
            reusable_update,
            update_gate,
            gate_indices,
            ffn_gate,
            output,
        )
    ):
        return False
    assert input_value is not None
    assert base is not None
    assert reusable_update is not None
    assert update_gate is not None
    assert gate_indices is not None
    assert ffn_gate is not None
    assert output is not None
    reusable_update_node = match.kwargs["reusable_update"]
    if not isinstance(reusable_update_node, torch.fx.Node):
        return False
    output_rows = math.prod(output.shape[:-1])
    outputs_valid = all(
        _shape_matches(value, output)
        and value.dtype is input_value.dtype
        and value.device == input_value.device
        and value.layout is torch.strided
        and value.is_contiguous()
        for value in (base, reusable_update)
    )
    gates_valid = all(
        gate.ndim == 2
        and gate.dtype is input_value.dtype
        and gate.device == input_value.device
        and gate.layout is torch.strided
        and gate.stride(-1) == 1
        and _dimension_matches(gate.shape[-1], output.shape[-1])
        for gate in (update_gate, ffn_gate)
    )
    indices_valid = (
        gate_indices.ndim == 1
        and gate_indices.dtype in (torch.int32, torch.int64)
        and gate_indices.device == input_value.device
        and gate_indices.layout is torch.strided
        and gate_indices.is_contiguous()
        and _dimension_matches(gate_indices.numel(), output_rows)
    )
    return bool(
        outputs_valid
        and gates_valid
        and indices_valid
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
    whose chunk size depends on the add's output workspace passes its policy as
    ``chunk_rows(input, up_weight_qdata, down_weight_qdata, gated_updates=True)``,
    called with their metadata; otherwise the FFN's chunk size carries over. Returns
    how many adds it replaced.
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
            metadata = [
                preparation_sharing.tensor_metadata(value)
                if isinstance(value := values[name], torch.fx.Node)
                else None
                for name in ("input", "up_weight_qdata", "down_weight_qdata")
            ]
            if all(value is not None for value in metadata):
                values["chunk_rows"] = chunk_rows(*metadata, gated_updates=True)
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
