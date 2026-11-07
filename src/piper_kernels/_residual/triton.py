"""Residual arithmetic shared by GEMM epilogues and standalone residual updates."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context

from . import ResidualAdd

_EPILOGUE_BLOCK_SIZE = 256


@triton.jit
def _load_gate(
    gate_ptr,
    indices,
    columns,
    valid,
    gate_row_stride: tl.constexpr,
    gate_rows,
    python_indexing: tl.constexpr,
):
    if python_indexing:
        indices = tl.where(indices < 0, indices + gate_rows, indices)
    return tl.load(
        gate_ptr + indices * gate_row_stride + columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)


@triton.jit
def add_residual(
    projection,
    base_ptr,
    output_ptr,
    gate_ptr,
    reusable_gate_ptr,
    gate_indices_ptr,
    rows,
    columns,
    valid,
    row_valid,
    features,
    row_offset,
    gate_row_stride: tl.constexpr,
    gate_rows,
    reusable_gate_row_stride: tl.constexpr,
    reusable_gate_rows,
    has_reusable_update: tl.constexpr,
    python_indexing: tl.constexpr,
):
    """Add a gated projection tile to residual rows, with an optional reusable update.

    ``rows``/``columns`` may describe a vector or a matrix tile. The output pointer
    addresses this chunk; with a reusable update it initially holds that update.
    """
    offsets = rows.to(tl.int64) * features + columns.to(tl.int64)
    indices = tl.load(gate_indices_ptr + row_offset + rows, mask=row_valid, other=0).to(tl.int64)
    hidden = tl.load(base_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    if has_reusable_update:
        reusable_update = tl.load(output_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        reusable_gate = _load_gate(
            reusable_gate_ptr,
            indices,
            columns,
            valid,
            reusable_gate_row_stride,
            reusable_gate_rows,
            python_indexing,
        )
        hidden = hidden + reusable_gate * reusable_update
    gate = _load_gate(
        gate_ptr,
        indices,
        columns,
        valid,
        gate_row_stride,
        gate_rows,
        python_indexing,
    )
    return hidden + gate * projection.to(tl.float32)


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
    result = add_residual(
        projection,
        base_ptr,
        output_ptr,
        gate_ptr,
        reusable_gate_ptr,
        gate_indices_ptr,
        rows,
        columns,
        valid,
        valid,
        features,
        row_offset,
        gate_row_stride,
        gate_rows,
        reusable_gate_row_stride,
        reusable_gate_rows,
        has_reusable_update,
        python_indexing,
    )
    tl.store(output_ptr + offsets, result, mask=valid)


def apply_residual_add(
    projection: torch.Tensor,
    residual_add: ResidualAdd,
    row_offset: int = 0,
) -> torch.Tensor:
    """Add a materialized projection to validated residual rows and return their view.

    ``projection`` is one chunk in the residual's dtype and device: rows in its
    leading dimensions, which flatten without a copy, and column-contiguous
    features. It may alias its output at matching element positions.
    ``row_offset`` locates its rows in the flattened residual and gate indices;
    the chunk must fit there. The result has ``projection``'s shape.
    The owning entry point validates the full residual before chunk execution.
    """
    features = projection.shape[-1]
    projection_rows = projection.view(-1, features)
    rows = projection_rows.shape[0]
    base = residual_add.base.view(-1, features)[row_offset : row_offset + rows]
    output = residual_add.output.view(-1, features)[row_offset : row_offset + rows]
    elements = projection.numel()
    if elements:
        gate = residual_add.gate
        reusable = residual_add.reusable
        reusable_gate = None if reusable is None else reusable.gate
        with device_context(projection.device):
            _residual_add_kernel[(triton.cdiv(elements, _EPILOGUE_BLOCK_SIZE),)](
                projection_rows,
                base,
                output,
                reusable_gate,
                gate,
                residual_add.gate_indices,
                elements,
                row_offset,
                features=features,
                projection_row_stride=projection_rows.stride(0),
                reusable_gate_row_stride=0 if reusable_gate is None else reusable_gate.stride(0),
                gate_row_stride=gate.stride(0),
                reusable_gate_rows=0 if reusable_gate is None else reusable_gate.shape[0],
                gate_rows=gate.shape[0],
                has_reusable_update=reusable is not None,
                python_indexing=residual_add.python_indexing,
                block_size=_EPILOGUE_BLOCK_SIZE,
                num_warps=4,
            )
    return output.view(projection.shape)
