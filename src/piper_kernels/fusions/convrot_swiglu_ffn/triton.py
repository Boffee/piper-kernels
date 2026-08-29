"""Bounded-workspace composition of a ConvRot SwiGLU feed-forward network."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import reference
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

_DEFAULT_CHUNK_ROWS = 4_096
_EPILOGUE_BLOCK_SIZE = 256


@dataclass(frozen=True, slots=True)
class _IndexedGatedUpdates:
    base: torch.Tensor
    reusable_update: torch.Tensor
    update_gate: torch.Tensor
    ffn_gate: torch.Tensor
    gate_indices: torch.Tensor
    python_indexing: bool


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
    base = tl.load(
        base_ptr + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
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


def _validate_bias(
    bias: torch.Tensor | None,
    *,
    features: int,
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    name: str,
) -> None:
    if bias is None:
        return
    if (
        bias.shape != (features,)
        or bias.dtype is not input.dtype
        or bias.device != input.device
        or bias.layout is not torch.strided
        or not bias.is_contiguous()
    ):
        raise ValueError(
            f"chunked ConvRot {name} bias must be a strided {input.dtype} "
            f"contiguous tensor with shape ({features},) on {input.device}"
        )


def _validate_inputs(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    chunk_rows: int,
) -> tuple[int, int, int]:
    if input.ndim == 0 or input.layout is not torch.strided or not input.is_contiguous():
        raise ValueError("chunked ConvRot FFN input must be a non-scalar contiguous strided tensor")
    if input.device.type != "cuda":
        raise ValueError("chunked ConvRot FFN currently requires CUDA")
    target = AcceleratorTarget.from_device(input.device)
    if not target.cuda_capability_at_least(7, 5):
        raise ValueError("chunked ConvRot FFN requires NVIDIA CUDA capability 7.5 or newer")
    if math.prod(input.shape[:-1]) < 1:
        raise ValueError("chunked ConvRot FFN requires at least one input row")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("chunked ConvRot FFN chunk_rows must be a positive integer")

    reference.validate_storage(
        up_weight_qdata,
        up_weight_scale,
        up_group_size,
        input.dtype,
    )
    reference.validate_storage(
        down_weight_qdata,
        down_weight_scale,
        down_group_size,
        input.dtype,
    )
    tensors = up_weight_qdata, up_weight_scale, down_weight_qdata, down_weight_scale
    if any(tensor.device != input.device for tensor in tensors):
        raise ValueError("chunked ConvRot FFN operands must share a CUDA device")

    input_features = up_weight_qdata.shape[1]
    intermediate_features = down_weight_qdata.shape[1]
    output_features = down_weight_qdata.shape[0]
    if input.shape[-1] != input_features:
        raise ValueError(
            f"chunked ConvRot FFN input has {input.shape[-1]} features, expected {input_features}"
        )
    if up_weight_qdata.shape[0] != 2 * intermediate_features:
        raise ValueError("chunked ConvRot FFN up projection must produce packed up/gate features")
    _validate_bias(
        up_bias,
        features=2 * intermediate_features,
        input=input,
        name="up",
    )
    _validate_bias(
        down_bias,
        features=output_features,
        input=input,
        name="down",
    )
    if torch.is_grad_enabled() and (
        input.requires_grad
        or up_weight_scale.requires_grad
        or down_weight_scale.requires_grad
        or (up_bias is not None and up_bias.requires_grad)
        or (down_bias is not None and down_bias.requires_grad)
    ):
        raise RuntimeError("chunked ConvRot FFN is inference-only and does not support autograd")
    return input_features, intermediate_features, output_features


def _validate_indexed_gated_updates(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    updates: _IndexedGatedUpdates,
    output_features: int,
) -> tuple[int, int, int, int]:
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
                f"chunked ConvRot FFN {name} must be a contiguous strided tensor "
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
                "chunked ConvRot FFN gates must be nonempty two-dimensional column-contiguous "
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
            "chunked ConvRot FFN gate indices must be a one-dimensional contiguous INT32 or INT64 "
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
        raise RuntimeError("chunked ConvRot FFN is inference-only and does not support autograd")
    return (
        updates.update_gate.stride(0),
        updates.ffn_gate.stride(0),
        updates.update_gate.shape[0],
        updates.ffn_gate.shape[0],
    )


def _run_chunked_swiglu_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    chunk_rows: int,
    *,
    gated_updates: _IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Run one materialization-equivalent FFN with bounded row workspaces."""
    input_features, intermediate_features, output_features = _validate_inputs(
        input,
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        chunk_rows,
    )
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)
    capacity = min(rows, chunk_rows)
    input_2d = input.reshape(rows, input_features)
    gate_layout = (
        None
        if gated_updates is None
        else _validate_indexed_gated_updates(
            input,
            gated_updates,
            output_features,
        )
    )
    output = (
        torch.empty(
            (*leading_shape, output_features),
            device=input.device,
            dtype=input.dtype,
        )
        if gated_updates is None
        else gated_updates.reusable_update
    )
    output_2d = output.reshape(rows, output_features)
    base_2d = None if gated_updates is None else gated_updates.base.reshape(rows, output_features)
    packed = torch.empty(
        (capacity, 2 * intermediate_features),
        device=input.device,
        dtype=input.dtype,
    )
    projected = None
    if gated_updates is not None:
        projected = (
            packed.reshape(-1)[: capacity * output_features].view(
                capacity,
                output_features,
            )
            if output_features <= 2 * intermediate_features
            else torch.empty(
                (capacity, output_features),
                device=input.device,
                dtype=input.dtype,
            )
        )
    prepared_storage = torch.empty(
        capacity * max(input_features, intermediate_features),
        device=input.device,
        dtype=torch.int8,
    )
    scale_storage = torch.empty(capacity, device=input.device, dtype=torch.float32)
    target = AcceleratorTarget.from_device(input.device)
    up_plan = convrot_backend.default_execution_plan(up_weight_qdata)
    down_plan = convrot_backend.default_execution_plan(down_weight_qdata)

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        chunk_row_count = stop - start
        prepared_input = prepared_storage[: chunk_row_count * input_features].view(
            chunk_row_count,
            input_features,
        )
        prepared_scale = scale_storage[:chunk_row_count]
        convrot_backend._prepare_input(
            input_2d[start:stop],
            input_features,
            up_group_size,
            activation_fn=None,
            execution_plan=up_plan,
            target=target,
            out=(prepared_input, prepared_scale),
        )
        packed_chunk = packed[:chunk_row_count]
        convrot_backend._execute_prepared_linear(
            prepared_input,
            prepared_scale,
            up_weight_qdata,
            up_weight_scale,
            up_bias,
            input.dtype,
            up_plan,
            out=packed_chunk,
        )

        prepared_swiglu = prepared_storage[: chunk_row_count * intermediate_features].view(
            chunk_row_count,
            intermediate_features,
        )
        convrot_backend._prepare_input(
            packed_chunk,
            intermediate_features,
            down_group_size,
            activation_fn="swiglu",
            execution_plan=down_plan,
            target=target,
            out=(prepared_swiglu, prepared_scale),
        )
        output_chunk = output_2d[start:stop] if projected is None else projected[:chunk_row_count]
        convrot_backend._execute_prepared_linear(
            prepared_swiglu,
            prepared_scale,
            down_weight_qdata,
            down_weight_scale,
            down_bias,
            input.dtype,
            down_plan,
            out=output_chunk,
        )
        if gated_updates is not None:
            assert base_2d is not None
            assert gate_layout is not None
            (
                update_gate_row_stride,
                ffn_gate_row_stride,
                update_gate_rows,
                ffn_gate_rows,
            ) = gate_layout
            elements = output_chunk.numel()
            _gated_updates_kernel[(triton.cdiv(elements, _EPILOGUE_BLOCK_SIZE),)](
                output_chunk,
                base_2d[start:stop],
                output_2d[start:stop],
                gated_updates.update_gate,
                gated_updates.ffn_gate,
                gated_updates.gate_indices,
                elements,
                start,
                features=output_features,
                update_gate_row_stride=update_gate_row_stride,
                ffn_gate_row_stride=ffn_gate_row_stride,
                update_gate_rows=update_gate_rows,
                ffn_gate_rows=ffn_gate_rows,
                python_indexing=gated_updates.python_indexing,
                block_size=_EPILOGUE_BLOCK_SIZE,
                num_warps=4,
                debug=True,
            )
    return output


@torch.library.custom_op("piper_kernels::convrot_swiglu_ffn", mutates_args=())
def _chunked_swiglu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    chunk_rows: int,
) -> torch.Tensor:
    return _run_chunked_swiglu_ffn(
        input,
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        chunk_rows,
    )


@_chunked_swiglu_ffn_op.register_fake
def _chunked_swiglu_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002
    _up_weight_qdata: torch.Tensor,
    _up_weight_scale: torch.Tensor,
    _up_bias: torch.Tensor | None,
    _up_group_size: int,
    down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_bias: torch.Tensor | None,
    _down_group_size: int,
    _chunk_rows: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], down_weight_qdata.shape[0]))


@torch.library.custom_op(
    "piper_kernels::convrot_swiglu_ffn_gated_updates_",
    mutates_args=("reusable_update",),
)
def _chunked_swiglu_ffn_gated_updates_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
) -> None:
    """Run a chunked FFN and apply indexed updates in reusable caller-owned storage."""
    _run_chunked_swiglu_ffn(
        input,
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        chunk_rows,
        gated_updates=_IndexedGatedUpdates(
            base=base,
            reusable_update=reusable_update,
            update_gate=update_gate,
            ffn_gate=ffn_gate,
            gate_indices=gate_indices,
            python_indexing=python_indexing,
        ),
    )


@_chunked_swiglu_ffn_gated_updates_op.register_fake
def _chunked_swiglu_ffn_gated_updates_op_fake(
    _input: torch.Tensor,
    _up_weight_qdata: torch.Tensor,
    _up_weight_scale: torch.Tensor,
    _up_bias: torch.Tensor | None,
    _up_group_size: int,
    _down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_bias: torch.Tensor | None,
    _down_group_size: int,
    _base: torch.Tensor,
    _reusable_update: torch.Tensor,
    _update_gate: torch.Tensor,
    _ffn_gate: torch.Tensor,
    _gate_indices: torch.Tensor,
    _python_indexing: bool,
    _chunk_rows: int,
) -> None:
    return None
