"""Select an optional projection backend for shared-input, non-residual SwiGLU FFNs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._amd import policy as amd_policy

if TYPE_CHECKING:
    from ._amd.triton import PackedProjections
    from ._core import LinearOperands


def prepare_swiglu_projections(
    output: torch.Tensor, sources: tuple[LinearOperands, ...], down: LinearOperands, chunk_rows: int
) -> PackedProjections | None:
    """Pack into an owned output when a long, shared-input projection amortizes the copy."""
    rows, output_features = output.shape
    if (
        rows <= chunk_rows
        or output.dtype != torch.bfloat16
        or any(linear.bias is not None for linear in (*sources, down))
    ):
        return None
    intermediate_features, input_features = sources[0].weight_qdata.shape
    target = AcceleratorTarget.from_device(output.device)
    prefix_rows = amd_policy.packed_prefix_rows(
        target, rows, input_features, intermediate_features, output_features, chunk_rows
    )
    if not prefix_rows:
        return None
    from ._amd.triton import pack_projections  # noqa: PLC0415

    return pack_projections(output, sources, down, prefix_rows, chunk_rows, target)
