"""Gated residual-add epilogue shared by FFN and attention projections."""

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
    ``python_indexing``. Arithmetic is FP32 with one rounding to the storage dtype.
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


@dataclass(frozen=True, slots=True)
class ResidualAddLayout:
    """Validated gate layout values passed to the Triton epilogue."""

    gate_row_stride: int
    gate_rows: int
    reusable_gate_row_stride: int = 0
    reusable_gate_rows: int = 1


@triton.jit
def _residual_add_kernel(
    projection_ptr,
    base_ptr,
    output_ptr,
    reusable_gate_ptr,
    gate_ptr,
    gate_indices_ptr,
    elements,
    row_offset,
    features: tl.constexpr,
    projection_row_stride: tl.constexpr,
    reusable_gate_row_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    reusable_gate_rows,
    gate_rows,
    has_reusable_update: tl.constexpr,
    python_indexing: tl.constexpr,
    block_size: tl.constexpr,
):
    """Add the projection's gated update, and an optional reusable update, to ``base``.

    With a reusable update, ``output_ptr`` holds that update and receives the result;
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
        reusable_gate_row = tl.where(indices < 0, indices + reusable_gate_rows, indices)
    else:
        gate_row = indices
        reusable_gate_row = indices
    if has_reusable_update:
        reusable_update = tl.load(
            output_ptr + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        reusable_gate = tl.load(
            reusable_gate_ptr + reusable_gate_row * reusable_gate_row_stride + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        hidden = hidden + reusable_gate * reusable_update
    gate = tl.load(
        gate_ptr + gate_row * gate_row_stride + columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    result = hidden + gate * projection
    tl.store(output_ptr + offsets, result, mask=valid)


def validate_residual_add(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    residual_add: ResidualAdd,
    output_features: int | torch.SymInt,
) -> ResidualAddLayout:
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
                f"gated update {name} must be a contiguous strided tensor "
                f"with shape {expected_shape}, dtype {input.dtype}, and device {input.device}"
            )
    for gate in (None if reusable is None else reusable.gate, residual_add.gate):
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
            "gated update indices must be a one-dimensional contiguous INT32 or INT64 "
            f"tensor with {rows} elements on {input.device}"
        )
    tensors = (
        residual_add.base,
        residual_add.gate,
        *(() if reusable is None else (reusable.update, reusable.gate)),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("gated updates are inference-only and do not support autograd")
    if reusable is None:
        return ResidualAddLayout(
            gate_row_stride=residual_add.gate.stride(0),
            gate_rows=residual_add.gate.shape[0],
        )
    return ResidualAddLayout(
        gate_row_stride=residual_add.gate.stride(0),
        gate_rows=residual_add.gate.shape[0],
        reusable_gate_row_stride=reusable.gate.stride(0),
        reusable_gate_rows=reusable.gate.shape[0],
    )


def apply_residual_add(
    projection: torch.Tensor,
    base: torch.Tensor,
    output: torch.Tensor,
    residual_add: ResidualAdd,
    layout: ResidualAddLayout,
    row_offset: int,
) -> None:
    """Apply one validated chunk of gated updates into its output rows.

    ``base`` and ``output`` are the chunk's rows of ``residual_add.base`` and
    ``residual_add.output``; ``row_offset`` locates the chunk in ``gate_indices``.
    """
    elements = projection.numel()
    reusable = residual_add.reusable
    with device_context(projection.device):
        _residual_add_kernel[(triton.cdiv(elements, _EPILOGUE_BLOCK_SIZE),)](
            projection,
            base,
            output,
            residual_add.gate if reusable is None else reusable.gate,
            residual_add.gate,
            residual_add.gate_indices,
            elements,
            row_offset,
            features=projection.shape[-1],
            projection_row_stride=projection.stride(0),
            reusable_gate_row_stride=layout.reusable_gate_row_stride,
            gate_row_stride=layout.gate_row_stride,
            reusable_gate_rows=layout.reusable_gate_rows,
            gate_rows=layout.gate_rows,
            has_reusable_update=reusable is not None,
            python_indexing=residual_add.python_indexing,
            block_size=_EPILOGUE_BLOCK_SIZE,
            num_warps=4,
        )


__all__ = [
    "ResidualAdd",
    "ResidualAddLayout",
    "ReusableUpdate",
    "apply_residual_add",
    "validate_residual_add",
]
