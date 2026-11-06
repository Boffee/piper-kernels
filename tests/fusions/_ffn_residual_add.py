"""Compare an FFN residual-add operator with its two-update operator."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def assert_residual_add_matches_gated_updates(
    gated_updates: Callable[..., None],
    residual_add: Callable[..., None],
    arguments: Sequence[object],
    trailing: int,
    output_features: int,
    *,
    python_indexing: bool,
) -> None:
    """Assert ``residual_add`` stores what ``gated_updates`` stores for a zero reused update.

    ``arguments`` are the plain FFN operator's; both update operators take the update
    operands before its last ``trailing`` arguments. Adding a zero update leaves each
    FP32 sum unchanged, so the results match exactly.
    """
    input = arguments[0]  # noqa: A001 - match linear terminology
    assert isinstance(input, torch.Tensor)
    leading, tail = arguments[:-trailing], arguments[-trailing:]
    base = torch.randn((*input.shape[:-1], output_features), device=input.device, dtype=input.dtype)
    ffn_gate = torch.randn(7, output_features, device=input.device, dtype=input.dtype)
    gate_indices = torch.randint(
        -7 if python_indexing else 0, 7, (input.shape[:-1].numel(),), device=input.device
    )
    expected = torch.zeros_like(base)
    gated_updates(
        *leading, base, expected, ffn_gate, ffn_gate, gate_indices, python_indexing, *tail
    )
    actual = base.clone()

    result = residual_add(*leading, actual, ffn_gate, gate_indices, python_indexing, *tail)

    assert result is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
