"""Indexed gated residual-update epilogue shared by FFN and attention projections."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context

_EPILOGUE_BLOCK_SIZE = 256


@dataclass(frozen=True, slots=True)
class FirstGatedUpdate:
    """An earlier update applied with the projection's; ``storage`` receives the result."""

    storage: torch.Tensor
    gate: torch.Tensor


@dataclass(frozen=True, slots=True)
class IndexedGatedUpdates:
    """Gated updates of ``base`` applied after each projected chunk.

    Each row becomes ``base + first.gate[i] * first.storage + gate[i] * projection``,
    stored in ``first.storage``. Without a first update it becomes
    ``base + gate[i] * projection``, stored in ``base``. Gate rows are selected by
    ``gate_indices``, which wrap negative values under ``python_indexing``.
    Arithmetic is FP32 with one rounding to the storage dtype.
    """

    base: torch.Tensor
    gate: torch.Tensor
    gate_indices: torch.Tensor
    python_indexing: bool
    first: FirstGatedUpdate | None = None

    @property
    def output(self) -> torch.Tensor:
        """Storage that receives the updated rows."""
        return self.base if self.first is None else self.first.storage


@dataclass(frozen=True, slots=True)
class IndexedGatedUpdateLayout:
    """Validated gate layout values passed to the Triton epilogue."""

    gate_row_stride: int
    gate_rows: int
    first_gate_row_stride: int = 0
    first_gate_rows: int = 1


@triton.jit
def _indexed_gated_updates_kernel(
    projection_ptr,
    base_ptr,
    output_ptr,
    first_gate_ptr,
    gate_ptr,
    gate_indices_ptr,
    elements,
    row_offset,
    features: tl.constexpr,
    projection_row_stride: tl.constexpr,
    first_gate_row_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    first_gate_rows,
    gate_rows,
    has_first_update: tl.constexpr,
    python_indexing: tl.constexpr,
    block_size: tl.constexpr,
):
    """Apply an optional first indexed gated update and the projection's update.

    With a first update, ``output_ptr`` holds that update and receives the result;
    without one, it is ``base_ptr``. Each element is read before it is written.
    """
    offsets = (tl.program_id(0) * block_size + tl.arange(0, block_size)).to(tl.int64)
    valid = offsets < elements
    rows = offsets // features
    columns = offsets % features
    projection = tl.load(
        projection_ptr + rows * projection_row_stride + columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    hidden = tl.load(base_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    indices = tl.load(
        gate_indices_ptr + row_offset + rows,
        mask=valid,
        other=0,
    ).to(tl.int64)
    if python_indexing:
        gate_row = tl.where(indices < 0, indices + gate_rows, indices)
        first_gate_row = tl.where(indices < 0, indices + first_gate_rows, indices)
    else:
        gate_row = indices
        first_gate_row = indices
    if has_first_update:
        first_update = tl.load(
            output_ptr + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        first_gate = tl.load(
            first_gate_ptr + first_gate_row * first_gate_row_stride + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        hidden = hidden + first_gate * first_update
    gate = tl.load(
        gate_ptr + gate_row * gate_row_stride + columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    result = hidden + gate * projection
    tl.store(output_ptr + offsets, result, mask=valid)


def validate_indexed_gated_updates(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    updates: IndexedGatedUpdates,
    output_features: int | torch.SymInt,
) -> IndexedGatedUpdateLayout:
    """Validate update metadata for one row per leading position of ``input``.

    ``input`` supplies the row shape, dtype, and device. Index values are caller
    preconditions under the library validation contract.
    """
    first = updates.first
    expected_shape = (*input.shape[:-1], output_features)
    for name, tensor in (
        ("base", updates.base),
        ("reusable update", None if first is None else first.storage),
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
                f"gated update {name} must be a contiguous strided tensor "
                f"with shape {expected_shape}, dtype {input.dtype}, and device {input.device}"
            )
    for gate in (None if first is None else first.gate, updates.gate):
        if gate is None:
            continue
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
                "gated update gates must be nonempty two-dimensional column-contiguous "
                f"strided tensors with last dimension {output_features}, dtype "
                f"{input.dtype}, and device {input.device}"
            )
    rows = math.prod(input.shape[:-1])
    gate_indices = updates.gate_indices
    if (
        gate_indices.numel() != rows
        or gate_indices.ndim != 1
        or gate_indices.dtype not in (torch.int32, torch.int64)
        or gate_indices.device != input.device
        or gate_indices.layout is not torch.strided
        or not gate_indices.is_contiguous()
    ):
        raise ValueError(
            "gated update indices must be a one-dimensional contiguous INT32 or INT64 "
            f"tensor with {rows} elements on {input.device}"
        )
    tensors = (updates.base, updates.gate, *(() if first is None else (first.storage, first.gate)))
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("gated updates are inference-only and do not support autograd")
    if first is None:
        return IndexedGatedUpdateLayout(
            gate_row_stride=updates.gate.stride(0),
            gate_rows=updates.gate.shape[0],
        )
    return IndexedGatedUpdateLayout(
        gate_row_stride=updates.gate.stride(0),
        gate_rows=updates.gate.shape[0],
        first_gate_row_stride=first.gate.stride(0),
        first_gate_rows=first.gate.shape[0],
    )


def apply_indexed_gated_updates(
    projection: torch.Tensor,
    base: torch.Tensor,
    output: torch.Tensor,
    updates: IndexedGatedUpdates,
    layout: IndexedGatedUpdateLayout,
    row_offset: int,
) -> None:
    """Apply one validated chunk of gated updates into its output rows.

    ``base`` and ``output`` are the chunk's rows of ``updates.base`` and
    ``updates.output``; ``row_offset`` locates the chunk in ``gate_indices``.
    """
    elements = projection.numel()
    first = updates.first
    with device_context(projection.device):
        _indexed_gated_updates_kernel[(triton.cdiv(elements, _EPILOGUE_BLOCK_SIZE),)](
            projection,
            base,
            output,
            updates.gate if first is None else first.gate,
            updates.gate,
            updates.gate_indices,
            elements,
            row_offset,
            features=projection.shape[-1],
            projection_row_stride=projection.stride(0),
            first_gate_row_stride=layout.first_gate_row_stride,
            gate_row_stride=layout.gate_row_stride,
            first_gate_rows=layout.first_gate_rows,
            gate_rows=layout.gate_rows,
            has_first_update=first is not None,
            python_indexing=updates.python_indexing,
            block_size=_EPILOGUE_BLOCK_SIZE,
            num_warps=4,
        )


__all__ = [
    "FirstGatedUpdate",
    "IndexedGatedUpdateLayout",
    "IndexedGatedUpdates",
    "apply_indexed_gated_updates",
    "validate_indexed_gated_updates",
]
