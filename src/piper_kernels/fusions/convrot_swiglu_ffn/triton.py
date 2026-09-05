"""Bounded-workspace composition of a ConvRot SwiGLU feed-forward network."""

from __future__ import annotations

import math

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.swiglu_ffn import triton as gated_updates_backend
from piper_kernels.linear import _bias
from piper_kernels.linear.convrot.int8 import reference
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

_DEFAULT_CHUNK_ROWS = 4_096


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
        or bias.device != input.device
        or bias.layout is not torch.strided
        or not bias.is_contiguous()
    ):
        raise ValueError(
            f"chunked ConvRot {name} bias must be a contiguous strided tensor "
            f"with shape ({features},) on {input.device}"
        )
    _bias.validate_dtype(bias, f"chunked ConvRot {name}")


def _validate_inputs(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    gate_weight_qdata: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
    gate_group_size: int,
    value_weight_qdata: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_bias: torch.Tensor | None,
    value_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    chunk_rows: int,
) -> tuple[int, int, int]:
    if input.ndim == 0 or input.layout is not torch.strided or not input.is_contiguous():
        raise ValueError("ConvRot FFN input must be a non-scalar contiguous strided tensor")
    if input.device.type != "cuda":
        raise ValueError("ConvRot FFN currently requires CUDA")
    target = AcceleratorTarget.from_device(input.device)
    if not target.cuda_capability_at_least(7, 5):
        raise ValueError("ConvRot FFN requires NVIDIA CUDA capability 7.5 or newer")
    if math.prod(input.shape[:-1]) < 1:
        raise ValueError("ConvRot FFN requires at least one input row")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("ConvRot FFN chunk_rows must be a positive integer")
    if gate_group_size != value_group_size:
        raise ValueError("ConvRot gate and value projections must share one group size")

    for weight_qdata, weight_scale, group_size in (
        (gate_weight_qdata, gate_weight_scale, gate_group_size),
        (value_weight_qdata, value_weight_scale, value_group_size),
        (down_weight_qdata, down_weight_scale, down_group_size),
    ):
        reference.validate_storage(weight_qdata, weight_scale, group_size, input.dtype)
        if weight_qdata.device != input.device or weight_scale.device != input.device:
            raise ValueError("ConvRot FFN operands must share a CUDA device")

    input_features = gate_weight_qdata.shape[1]
    intermediate_features = gate_weight_qdata.shape[0]
    output_features = down_weight_qdata.shape[0]
    if input.shape[-1] != input_features:
        raise ValueError(
            f"ConvRot FFN input has {input.shape[-1]} features, expected {input_features}"
        )
    if value_weight_qdata.shape != (intermediate_features, input_features):
        raise ValueError("ConvRot gate and value projections must have matching shapes")
    if down_weight_qdata.shape[1] != intermediate_features:
        raise ValueError("ConvRot down projection must consume the gate/value width")
    _validate_bias(gate_bias, features=intermediate_features, input=input, name="gate")
    _validate_bias(value_bias, features=intermediate_features, input=input, name="value")
    _validate_bias(down_bias, features=output_features, input=input, name="down")
    differentiable = (
        input,
        gate_weight_scale,
        value_weight_scale,
        down_weight_scale,
        *(bias for bias in (gate_bias, value_bias, down_bias) if bias is not None),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in differentiable):
        raise RuntimeError("ConvRot FFN is inference-only and does not support autograd")
    return input_features, intermediate_features, output_features


def _run_chunked_swiglu_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    gate_weight_qdata: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
    gate_group_size: int,
    value_weight_qdata: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_bias: torch.Tensor | None,
    value_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    chunk_rows: int,
    *,
    gated_updates: gated_updates_backend.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Run a semantic gate/value SwiGLU FFN with bounded row workspaces."""
    input_features, intermediate_features, output_features = _validate_inputs(
        input,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        value_weight_qdata,
        value_weight_scale,
        value_bias,
        value_group_size,
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
        else gated_updates_backend.validate_indexed_gated_updates(
            input,
            gated_updates,
            output_features,
        )
    )
    output = (
        torch.empty((*leading_shape, output_features), device=input.device, dtype=input.dtype)
        if gated_updates is None
        else gated_updates.reusable_update
    )
    output_2d = output.reshape(rows, output_features)
    base_2d = None if gated_updates is None else gated_updates.base.reshape(rows, output_features)
    projection_workspace = torch.empty(
        (capacity, 2 * intermediate_features),
        device=input.device,
        dtype=input.dtype,
    )
    projected = None
    if gated_updates is not None:
        projected = (
            projection_workspace.reshape(-1)[: capacity * output_features].view(
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
    input_plan = convrot_backend.default_execution_plan(gate_weight_qdata)
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
            gate_group_size,
            activation_fn=None,
            execution_plan=input_plan,
            target=target,
            out=(prepared_input, prepared_scale),
        )
        projections = projection_workspace[:chunk_row_count]
        convrot_backend._execute_prepared_linear(
            prepared_input,
            prepared_scale,
            value_weight_qdata,
            value_weight_scale,
            value_bias,
            input.dtype,
            input_plan,
            out=projections,
            second_projection=(gate_weight_qdata, gate_weight_scale, gate_bias),
        )

        prepared_swiglu = prepared_storage[: chunk_row_count * intermediate_features].view(
            chunk_row_count,
            intermediate_features,
        )
        convrot_backend._prepare_input(
            projections,
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
            gated_updates_backend.apply_indexed_gated_updates(
                output_chunk,
                base_2d[start:stop],
                output_2d[start:stop],
                gated_updates,
                gate_layout,
                start,
            )
    return output


@torch.library.custom_op("piper_kernels::convrot_swiglu_ffn", mutates_args=())
def _chunked_swiglu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    gate_weight_qdata: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
    gate_group_size: int,
    value_weight_qdata: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_bias: torch.Tensor | None,
    value_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    chunk_rows: int,
) -> torch.Tensor:
    return _run_chunked_swiglu_ffn(
        input,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        value_weight_qdata,
        value_weight_scale,
        value_bias,
        value_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        chunk_rows,
    )


@_chunked_swiglu_ffn_op.register_fake
def _chunked_swiglu_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002
    _gate_weight_qdata: torch.Tensor,
    _gate_weight_scale: torch.Tensor,
    _gate_bias: torch.Tensor | None,
    _gate_group_size: int,
    _value_weight_qdata: torch.Tensor,
    _value_weight_scale: torch.Tensor,
    _value_bias: torch.Tensor | None,
    _value_group_size: int,
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
    gate_weight_qdata: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
    gate_group_size: int,
    value_weight_qdata: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_bias: torch.Tensor | None,
    value_group_size: int,
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
    _run_chunked_swiglu_ffn(
        input,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        value_weight_qdata,
        value_weight_scale,
        value_bias,
        value_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        chunk_rows,
        gated_updates=gated_updates_backend.IndexedGatedUpdates(
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
    _gate_weight_qdata: torch.Tensor,
    _gate_weight_scale: torch.Tensor,
    _gate_bias: torch.Tensor | None,
    _gate_group_size: int,
    _value_weight_qdata: torch.Tensor,
    _value_weight_scale: torch.Tensor,
    _value_bias: torch.Tensor | None,
    _value_group_size: int,
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
