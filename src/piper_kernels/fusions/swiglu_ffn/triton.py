"""Shared indexed-gated-update Triton epilogue for chunked SwiGLU FFNs."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_EPILOGUE_BLOCK_SIZE = 256


@dataclass(frozen=True, slots=True)
class IndexedGatedUpdates:
    """Caller-owned tensors consumed by H3's indexed FFN update epilogue."""

    base: torch.Tensor
    reusable_update: torch.Tensor
    update_gate: torch.Tensor
    ffn_gate: torch.Tensor
    gate_indices: torch.Tensor
    python_indexing: bool


@dataclass(frozen=True, slots=True)
class IndexedGatedUpdateLayout:
    """Validated gate layout values passed to the Triton epilogue."""

    update_gate_row_stride: int
    ffn_gate_row_stride: int
    update_gate_rows: int
    ffn_gate_rows: int


@triton.jit
def _gated_updates_kernel(
    ffn_ptr,
    base_ptr,
    reusable_update_ptr,
    update_gate_ptr,
    ffn_gate_ptr,
    gate_indices_ptr,
    elements,
    row_offset,
    features: tl.constexpr,
    update_gate_row_stride: tl.constexpr,
    ffn_gate_row_stride: tl.constexpr,
    update_gate_rows,
    ffn_gate_rows,
    python_indexing: tl.constexpr,
    block_size: tl.constexpr,
):
    """Apply two indexed gated updates and reuse the first update as output."""
    offsets = (tl.program_id(0) * block_size + tl.arange(0, block_size)).to(tl.int64)
    valid = offsets < elements
    rows = offsets // features
    columns = offsets % features
    ffn = tl.load(ffn_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    reusable_update = tl.load(
        reusable_update_ptr + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    gate_rows = tl.load(
        gate_indices_ptr + row_offset + rows,
        mask=valid,
        other=0,
    ).to(tl.int64)
    if python_indexing:
        update_gate_row = tl.where(gate_rows < 0, gate_rows + update_gate_rows, gate_rows)
        ffn_gate_row = tl.where(gate_rows < 0, gate_rows + ffn_gate_rows, gate_rows)
    else:
        update_gate_row = gate_rows
        ffn_gate_row = gate_rows
    update_gate_row_valid = (update_gate_row >= 0) & (update_gate_row < update_gate_rows)
    ffn_gate_row_valid = (ffn_gate_row >= 0) & (ffn_gate_row < ffn_gate_rows)
    gate_rows_valid = update_gate_row_valid & ffn_gate_row_valid
    tl.device_assert(gate_rows_valid, "gate index out of bounds", mask=valid)
    update_gate = tl.load(
        update_gate_ptr + update_gate_row * update_gate_row_stride + columns,
        mask=valid & gate_rows_valid,
        other=float("nan"),
    ).to(tl.float32)
    ffn_gate = tl.load(
        ffn_gate_ptr + ffn_gate_row * ffn_gate_row_stride + columns,
        mask=valid & gate_rows_valid,
        other=float("nan"),
    ).to(tl.float32)
    hidden = base + update_gate * reusable_update
    result = hidden + ffn_gate * ffn
    tl.store(reusable_update_ptr + offsets, result, mask=valid)


def validate_indexed_gated_updates(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    updates: IndexedGatedUpdates,
    output_features: int,
) -> IndexedGatedUpdateLayout:
    """Validate indexed-update operands and return their strided gate layout."""
    expected_shape = (*input.shape[:-1], output_features)
    for name, tensor in (
        ("base", updates.base),
        ("reusable update", updates.reusable_update),
    ):
        if (
            tensor.shape != expected_shape
            or tensor.dtype is not input.dtype
            or tensor.device != input.device
            or tensor.layout is not torch.strided
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"chunked FFN {name} must be a contiguous strided tensor "
                f"with shape {expected_shape}, dtype {input.dtype}, and device {input.device}"
            )
    for gate in (updates.update_gate, updates.ffn_gate):
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
                "chunked FFN gates must be nonempty two-dimensional column-contiguous "
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
            "chunked FFN gate indices must be a one-dimensional contiguous INT32 or INT64 "
            f"tensor with {rows} elements on {input.device}"
        )
    if torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (
            updates.base,
            updates.reusable_update,
            updates.update_gate,
            updates.ffn_gate,
        )
    ):
        raise RuntimeError("chunked FFN is inference-only and does not support autograd")
    return IndexedGatedUpdateLayout(
        update_gate_row_stride=updates.update_gate.stride(0),
        ffn_gate_row_stride=updates.ffn_gate.stride(0),
        update_gate_rows=updates.update_gate.shape[0],
        ffn_gate_rows=updates.ffn_gate.shape[0],
    )


def apply_indexed_gated_updates(
    ffn: torch.Tensor,
    base: torch.Tensor,
    output: torch.Tensor,
    updates: IndexedGatedUpdates,
    layout: IndexedGatedUpdateLayout,
    row_offset: int,
) -> None:
    """Apply one validated chunk of indexed gated updates into reusable output storage."""
    elements = ffn.numel()
    _gated_updates_kernel[(triton.cdiv(elements, _EPILOGUE_BLOCK_SIZE),)](
        ffn,
        base,
        output,
        updates.update_gate,
        updates.ffn_gate,
        updates.gate_indices,
        elements,
        row_offset,
        features=ffn.shape[-1],
        update_gate_row_stride=layout.update_gate_row_stride,
        ffn_gate_row_stride=layout.ffn_gate_row_stride,
        update_gate_rows=layout.update_gate_rows,
        ffn_gate_rows=layout.ffn_gate_rows,
        python_indexing=updates.python_indexing,
        block_size=_EPILOGUE_BLOCK_SIZE,
        num_warps=4,
        debug=True,
    )


__all__ = [
    "IndexedGatedUpdateLayout",
    "IndexedGatedUpdates",
    "apply_indexed_gated_updates",
    "validate_indexed_gated_updates",
]
