"""Generic in-place residual add, and the operand contract compiler rewrites check.

``residual_add_`` validates its operands when it runs. :func:`supports_residual_add`
checks the same contract, plus device support, from compile-time metadata, comparing
symbolic sizes without adding shape guards; change both together.
"""

from __future__ import annotations

import math

import torch

from piper_kernels._residual import ResidualAdd, validate_residual_add
from piper_kernels._residual.triton import apply_residual_add
from piper_kernels._triton import runtime
from piper_kernels.linear import _preparation_sharing as preparation_sharing


def _same_dimension(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


def supports_residual_add(
    base: torch.Tensor,
    update: torch.Tensor | None,
    gate: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Whether adding ``gate[indices] * update`` into ``base`` can run, from metadata.

    ``base``, and ``update`` when given, must be contiguous floating-point tensors with
    one shape and dtype, on a device Triton supports. ``gate`` is two-dimensional, with
    unit-stride rows of ``base``'s features and dtype; ``indices`` holds one int32 or
    int64 row index per ``base`` row. Symbolic sizes are compared without adding shape
    guards.
    """
    if base.ndim < 1 or not base.dtype.is_floating_point:
        return False
    rows = math.prod(base.shape[:-1])
    return bool(
        all(
            value.ndim == base.ndim
            and all(map(_same_dimension, value.shape, base.shape))
            and value.dtype is base.dtype
            and value.device == base.device
            and value.layout is torch.strided
            and value.is_contiguous()
            for value in (base, update)
            if value is not None
        )
        and gate.ndim == 2
        and gate.dtype is base.dtype
        and gate.device == base.device
        and gate.layout is torch.strided
        and gate.stride(-1) == 1
        and _same_dimension(gate.shape[-1], base.shape[-1])
        and indices.ndim == 1
        and indices.dtype in (torch.int32, torch.int64)
        and indices.device == base.device
        and indices.layout is torch.strided
        and indices.is_contiguous()
        and _same_dimension(indices.numel(), rows)
        and runtime.supports_device(base.device)
    )


def _validate(
    base: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
) -> ResidualAdd:
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
    residual_add = ResidualAdd(
        base=base,
        gate=gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    validate_residual_add(base, residual_add, base.shape[-1])
    return residual_add


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
    residual_add = _validate(base, update, gate, gate_indices, python_indexing)
    if base.numel() == 0:
        return
    apply_residual_add(update, residual_add)


@_residual_add_op.register_fake
def _residual_add_op_fake(
    base: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
) -> None:
    _validate(base, update, gate, gate_indices, python_indexing)
