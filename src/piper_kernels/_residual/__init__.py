"""Residual-update operands and metadata validation shared by operators and fusions.

This module imports without Triton. Entry points validate once; internal chunk
execution relies on that contract and never examines index or gate contents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ReusableUpdate:
    """A computed gated update whose ``update`` storage receives the residual sum."""

    update: torch.Tensor
    gate: torch.Tensor


@dataclass(frozen=True, slots=True)
class ResidualAdd:
    """Add ``gate[i] * projection`` to each row of ``base`` after each projected chunk.

    Each row becomes ``base + gate[i] * projection``, stored in ``base``. With a
    ``reusable`` update, it becomes ``base + reusable.gate[i] * reusable.update +
    gate[i] * projection``, stored over ``reusable.update`` so ``base`` is unchanged.
    Gate rows are selected by ``gate_indices``, which wrap negative values under
    ``python_indexing``. Indices must be in each gate's row range (or its negative
    row range with Python indexing); values are not checked.
    Arithmetic is FP32 with one rounding to the storage dtype.
    The projection is rounded to its logical dtype before residual arithmetic;
    native GEMM epilogues preserve that boundary without materializing a tensor.
    FP32 multiply-add contraction can differ between native and standalone paths.
    ``base`` and a reusable update have disjoint storage. Operands must not alias
    a GEMM's prepared input, weights, or scales, which other tiles may still read.
    """

    base: torch.Tensor
    gate: torch.Tensor
    gate_indices: torch.Tensor
    python_indexing: bool
    reusable: ReusableUpdate | None = None

    @property
    def output(self) -> torch.Tensor:
        """Storage that receives the updated rows."""
        return self.base if self.reusable is None else self.reusable.update


def validate_residual_add(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    residual_add: ResidualAdd,
    output_features: int | torch.SymInt,
) -> None:
    """Validate update metadata for one row per leading position of ``input``.

    ``input`` supplies the row shape, dtype, and device. Index values are caller
    preconditions under the library validation contract.
    """
    reusable = residual_add.reusable
    expected_shape = (*input.shape[:-1], output_features)
    for name, tensor in (
        ("base", residual_add.base),
        ("reusable update", None if reusable is None else reusable.update),
    ):
        if tensor is None:
            continue
        if (
            tensor.shape != expected_shape
            or tensor.dtype is not input.dtype
            or tensor.device != input.device
            or tensor.layout is not torch.strided
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"residual add {name} must be a contiguous strided tensor "
                f"with shape {expected_shape}, dtype {input.dtype}, and device {input.device}"
            )
    gates = (residual_add.gate, *(() if reusable is None else (reusable.gate,)))
    for gate in gates:
        if (
            gate.dtype is not input.dtype
            or gate.device != input.device
            or gate.layout is not torch.strided
            or gate.ndim != 2
            or gate.shape[0] < 1
            or gate.shape[-1] != output_features
            or gate.stride(-1) != 1
        ):
            raise ValueError(
                "residual add gates must be nonempty two-dimensional column-contiguous "
                f"strided tensors with last dimension {output_features}, dtype "
                f"{input.dtype}, and device {input.device}"
            )
    rows = math.prod(input.shape[:-1])
    gate_indices = residual_add.gate_indices
    if (
        gate_indices.numel() != rows
        or gate_indices.ndim != 1
        or gate_indices.dtype not in (torch.int32, torch.int64)
        or gate_indices.device != input.device
        or gate_indices.layout is not torch.strided
        or not gate_indices.is_contiguous()
    ):
        raise ValueError(
            "residual add indices must be a one-dimensional contiguous INT32 or INT64 "
            f"tensor with {rows} elements on {input.device}"
        )
    tensors = (residual_add.base, *gates, *(() if reusable is None else (reusable.update,)))
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("residual adds are inference-only and do not support autograd")
