"""Check FFN residual-add operators and the copy-back blocks that use them."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from _compile_capture import FunctionalTargetCapturePass
from torch._dynamo.utils import ReinplaceCounters

from piper_kernels.fusions.residual_updates import residual_updates_compile_options


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


def assert_copy_back_computes_ffn_inside_its_add(
    model: Callable[..., torch.Tensor],
    arguments: Sequence[torch.Tensor],
    ffn: str,
    ordinary_options: Mapping[str, object],
    fusion_options: Mapping[str, object],
    tolerance: float,
) -> None:
    """Assert a copy-back gated-update block adds into its base, computing ``ffn`` inside.

    ``model(base, ...)`` returns ``base`` plus a gated update and a gated ``ffn`` output,
    as the FFN fusions' gated-update test modules do; ``arguments[0]`` is ``base``.
    ``ffn`` names the FFN operator, such as ``"nvfp4_gelu_ffn"``. The block must match
    ``model`` compiled with ``ordinary_options`` within ``tolerance`` relative L2.
    """
    base, *operands = arguments
    capture = FunctionalTargetCapturePass()

    def copy_back(base: torch.Tensor, *operands: torch.Tensor) -> torch.Tensor:
        return base.copy_(model(base, *operands))

    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=dict(ordinary_options))(
            base.clone(), *operands
        )
        torch._dynamo.reset()
        ReinplaceCounters.clear()
        options = capture.wrap_options(residual_updates_compile_options(fusion_options))
        result = torch.compile(copy_back, fullgraph=True, options=options)(base, *operands)

    assert result.data_ptr() == base.data_ptr()
    relative_l2 = (base.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < tolerance
    # Inductor added both updates into ``base`` instead of copying it first.
    assert ReinplaceCounters.get_total_missed() == 0
    kernels = torch.ops.piper_kernels
    assert capture.targets.count(getattr(kernels, f"{ffn}_residual_add_").default) == 1
    assert capture.targets.count(kernels.residual_add_.default) == 1
    assert getattr(kernels, f"{ffn}_gated_updates_").default not in capture.targets
