"""Generic in-place residual add."""

from __future__ import annotations

import torch

from . import triton as residual_updates


def _validate(
    base: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
) -> tuple[residual_updates.ResidualAdd, residual_updates.ResidualAddLayout]:
    if base.ndim < 1 or not base.dtype.is_floating_point:
        raise ValueError("residual add base must be a floating-point tensor with features")
    if (
        update.shape != base.shape
        or update.dtype is not base.dtype
        or update.device != base.device
        or update.layout is not torch.strided
        or not update.is_contiguous()
    ):
        raise ValueError("residual add update must be contiguous and match base")
    updates = residual_updates.ResidualAdd(
        base=base,
        gate=gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    return updates, residual_updates.validate_residual_add(base, updates, base.shape[-1])


@torch.library.custom_op("piper_kernels::residual_add_", mutates_args=("base",))
def _residual_add_op(
    base: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
) -> None:
    """Add ``gate[gate_indices] * update`` into ``base`` row by row, rounding once.

    Rows are ``base``'s leading positions; ``gate_indices`` selects one ``gate`` row
    for each, wrapping negative values under ``python_indexing``.
    """
    updates, layout = _validate(base, update, gate, gate_indices, python_indexing)
    if base.numel() == 0:
        return
    rows = base.view(-1, base.shape[-1])
    residual_updates.apply_residual_add(
        update.view(-1, base.shape[-1]), rows, rows, updates, layout, 0
    )


@_residual_add_op.register_fake
def _residual_add_op_fake(
    base: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
) -> None:
    _validate(base, update, gate, gate_indices, python_indexing)
