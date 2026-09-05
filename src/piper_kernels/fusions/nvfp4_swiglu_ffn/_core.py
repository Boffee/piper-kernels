"""Shared bounded-workspace execution for NVFP4 SwiGLU FFNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.fusions.swiglu_ffn import triton as gated_updates_backend
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import _projection as nvfp4_projection
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend

DEFAULT_CHUNK_ROWS = 1_536
_SCALE_ROW_BLOCK = 128


@dataclass(frozen=True, slots=True)
class LinearOperands:
    """Canonical storage and scaling operands for one NVFP4 projection."""

    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    weight_per_tensor_scale: torch.Tensor | None
    activation_per_tensor_scale: torch.Tensor | None
    bias: torch.Tensor | None
    dynamic_activation_scale: bool
    high_first: bool


class PreparationBackend(Protocol):
    """Format-specific activation preparation used by the shared FFN runner."""

    def dynamic_source_scale(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
    ) -> torch.Tensor:
        """Calculate one global scale for the complete FFN input."""
        ...

    def prepare_source(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare one FFN input chunk into reusable storage."""
        ...

    def prepare_down(
        self,
        projections: torch.Tensor,
        activation_per_tensor_scale: torch.Tensor | None,
        dynamic_activation_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply SwiGLU and prepare one down-projection input chunk."""
        ...


def _dynamic_swiglu_scale(projections: torch.Tensor) -> torch.Tensor:
    """Calculate the dynamic scale in FP32 without materializing SwiGLU."""
    value, gate = projections.chunk(2, dim=-1)
    return per_tensor_amax_to_scale((value.float() * F.silu(gate.float())).abs().amax())


dynamic_swiglu_scale = torch.compile(_dynamic_swiglu_scale, fullgraph=True)


def linear_operands(  # noqa: PLR0913, PLR0917 - explicit custom-op projection operands
    gate_weight_qdata: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_weight_per_tensor_scale: torch.Tensor | None,
    gate_activation_per_tensor_scale: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    gate_dynamic_activation_scale: bool,
    gate_high_first: bool,
    value_weight_qdata: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_weight_per_tensor_scale: torch.Tensor | None,
    value_activation_per_tensor_scale: torch.Tensor | None,
    value_bias: torch.Tensor | None,
    value_dynamic_activation_scale: bool,
    value_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_high_first: bool,
) -> tuple[LinearOperands, LinearOperands, LinearOperands]:
    return (
        LinearOperands(
            gate_weight_qdata,
            gate_weight_scale,
            gate_weight_per_tensor_scale,
            gate_activation_per_tensor_scale,
            gate_bias,
            gate_dynamic_activation_scale,
            gate_high_first,
        ),
        LinearOperands(
            value_weight_qdata,
            value_weight_scale,
            value_weight_per_tensor_scale,
            value_activation_per_tensor_scale,
            value_bias,
            value_dynamic_activation_scale,
            value_high_first,
        ),
        LinearOperands(
            down_weight_qdata,
            down_weight_scale,
            down_weight_per_tensor_scale,
            down_activation_per_tensor_scale,
            down_bias,
            down_dynamic_activation_scale,
            down_high_first,
        ),
    )


def _project_chunk_out(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    linear: LinearOperands,
    start: int,
    stop: int,
    output: torch.Tensor,
) -> torch.Tensor:
    return nvfp4_projection.matmul_prepared_chunk_out(
        input_qdata,
        input_scale,
        linear.weight_qdata,
        linear.weight_scale,
        start,
        stop,
        output,
    )


def _projection_global_scale(
    input_per_tensor_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
) -> torch.Tensor:
    if weight_per_tensor_scale is None:
        return input_per_tensor_scale
    return input_per_tensor_scale * weight_per_tensor_scale


def _same_tensor_storage(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(
        left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype is right.dtype
        and left.device == right.device
        and left.storage_offset() == right.storage_offset()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _validate_inputs(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    gate: LinearOperands,
    value: LinearOperands,
    down: LinearOperands,
    chunk_rows: int,
) -> tuple[int, int, int]:
    gate_shape = nvfp4_validation.validate_semantic_linear(
        input,
        gate.weight_qdata,
        gate.weight_scale,
        gate.weight_per_tensor_scale,
        gate.activation_per_tensor_scale,
        gate.bias,
        gate.dynamic_activation_scale,
        "NVFP4 FFN gate projection",
    )
    value_shape = nvfp4_validation.validate_semantic_linear(
        input,
        value.weight_qdata,
        value.weight_scale,
        value.weight_per_tensor_scale,
        value.activation_per_tensor_scale,
        value.bias,
        value.dynamic_activation_scale,
        "NVFP4 FFN value projection",
    )
    if input.dtype is not torch.bfloat16:
        raise ValueError("NVFP4 FFN currently requires BF16 activations")
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < _SCALE_ROW_BLOCK
        or chunk_rows % _SCALE_ROW_BLOCK
    ):
        raise ValueError("NVFP4 FFN chunk_rows must be a positive multiple of 128")
    if gate.high_first != value.high_first:
        raise ValueError("NVFP4 gate and value projections must share nibble ordering")
    if (
        gate_shape.rows != value_shape.rows
        or gate_shape.input_features != value_shape.input_features
        or gate_shape.output_features != value_shape.output_features
        or not isinstance(gate_shape.rows, int)
        or not isinstance(gate_shape.output_features, int)
    ):
        raise ValueError("NVFP4 gate and value projections must have matching shapes")
    intermediate_features = gate_shape.output_features
    nvfp4_validation.validate_activation_scale(
        down.activation_per_tensor_scale,
        down.dynamic_activation_scale,
        input.device,
        "NVFP4 FFN down projection",
    )
    output_features = nvfp4_validation.validate_weight(
        down.weight_qdata,
        down.weight_scale,
        down.weight_per_tensor_scale,
        down.bias,
        input_features=intermediate_features,
        device=input.device,
        name="NVFP4 FFN down projection",
    )
    if not isinstance(output_features, int):
        raise ValueError("NVFP4 FFN requires concrete projection dimensions")
    differentiable_tensors = (
        input,
        gate.weight_scale,
        value.weight_scale,
        down.weight_scale,
        *(
            tensor
            for tensor in (
                gate.weight_per_tensor_scale,
                gate.activation_per_tensor_scale,
                gate.bias,
                value.weight_per_tensor_scale,
                value.activation_per_tensor_scale,
                value.bias,
                down.weight_per_tensor_scale,
                down.activation_per_tensor_scale,
                down.bias,
            )
            if tensor is not None
        ),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in differentiable_tensors):
        raise RuntimeError("NVFP4 FFN is inference-only and does not support autograd")
    return gate_shape.rows, intermediate_features, output_features


def _project_affine_source_chunk(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    linear: LinearOperands,
    rows: int,
    output: torch.Tensor,
) -> None:
    # Mixed-dtype bias needs the original combined FP32 scale/bias epilogue;
    # the affine GEMM helper would round to BF16 before adding that bias.
    if linear.bias is None or linear.bias.dtype is output.dtype:
        _project_affine_chunk(
            input_qdata, input_scale, input_per_tensor_scale, linear, rows, output
        )
        return
    projected = _project_chunk_out(input_qdata, input_scale, linear, 0, rows, output)
    nvfp4_backend.apply_projection_epilogue(
        projected,
        _projection_global_scale(input_per_tensor_scale, linear.weight_per_tensor_scale),
        linear.bias,
        projected,
    )


def _project_affine_chunk(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    linear: LinearOperands,
    rows: int,
    output: torch.Tensor,
) -> None:
    weight_per_tensor_scale = linear.weight_per_tensor_scale
    if weight_per_tensor_scale is not None:
        nvfp4_projection.matmul_prepared_chunk_affine_out(
            input_qdata,
            input_scale,
            input_per_tensor_scale,
            linear.weight_qdata,
            linear.weight_scale,
            weight_per_tensor_scale,
            linear.bias,
            0,
            rows,
            output,
        )
        return
    projected = _project_chunk_out(
        input_qdata,
        input_scale,
        linear,
        0,
        rows,
        output,
    )
    nvfp4_backend.apply_projection_epilogue(
        projected,
        input_per_tensor_scale,
        linear.bias,
        projected,
    )


def run_chunked_swiglu_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    gate: LinearOperands,
    value: LinearOperands,
    down: LinearOperands,
    chunk_rows: int,
    preparation: PreparationBackend,
    *,
    gated_updates: gated_updates_backend.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Run a semantic gate/value NVFP4 FFN with bounded row workspaces."""
    rows, intermediate_features, output_features = _validate_inputs(
        input,
        gate,
        value,
        down,
        chunk_rows,
    )
    leading_shape = input.shape[:-1]
    input_features = input.shape[-1]
    input_2d = input.reshape(rows, input_features)
    gate_per_tensor_scale = (
        preparation.dynamic_source_scale(input)
        if gate.dynamic_activation_scale
        else gate.activation_per_tensor_scale
    )
    assert gate_per_tensor_scale is not None
    value_per_tensor_scale = (
        gate_per_tensor_scale
        if gate.dynamic_activation_scale and value.dynamic_activation_scale
        else (
            preparation.dynamic_source_scale(input)
            if value.dynamic_activation_scale
            else value.activation_per_tensor_scale
        )
    )
    assert value_per_tensor_scale is not None
    shared_input_preparation = (
        gate.dynamic_activation_scale and value.dynamic_activation_scale
    ) or (
        not gate.dynamic_activation_scale
        and not value.dynamic_activation_scale
        and _same_tensor_storage(
            gate.activation_per_tensor_scale,
            value.activation_per_tensor_scale,
        )
    )
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
    workspace_rows = min(rows, chunk_rows)
    source_storage = nvfp4_layout.prepare_activation_storage(
        input,
        workspace_rows,
        input_features,
    )
    projection_workspace = torch.empty(
        (workspace_rows, 2 * intermediate_features),
        device=input.device,
        dtype=input.dtype,
    )
    projected_workspace = (
        None
        if gated_updates is None
        else torch.empty(
            (workspace_rows, output_features),
            device=input.device,
            dtype=input.dtype,
        )
    )

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        chunk_row_count = stop - start
        value_input_qdata, value_input_scale = preparation.prepare_source(
            input_2d[start:stop],
            value_per_tensor_scale,
            source_storage,
        )
        projections = projection_workspace[:chunk_row_count]
        value_output = projections[:, :intermediate_features]
        gate_output = projections[:, intermediate_features:]
        _project_affine_source_chunk(
            value_input_qdata,
            value_input_scale,
            value_per_tensor_scale,
            value,
            chunk_row_count,
            value_output,
        )
        gate_input_qdata, gate_input_scale = (
            (value_input_qdata, value_input_scale)
            if shared_input_preparation
            else preparation.prepare_source(
                input_2d[start:stop],
                gate_per_tensor_scale,
                source_storage,
            )
        )
        _project_affine_source_chunk(
            gate_input_qdata,
            gate_input_scale,
            gate_per_tensor_scale,
            gate,
            chunk_row_count,
            gate_output,
        )
        down_qdata, down_scale, down_per_tensor_scale = preparation.prepare_down(
            projections,
            down.activation_per_tensor_scale,
            down.dynamic_activation_scale,
        )
        if gated_updates is None:
            _project_affine_chunk(
                down_qdata,
                down_scale,
                down_per_tensor_scale,
                down,
                chunk_row_count,
                output_2d[start:stop],
            )
            continue
        assert projected_workspace is not None
        assert base_2d is not None
        assert gate_layout is not None
        projected = _project_chunk_out(
            down_qdata,
            down_scale,
            down,
            0,
            chunk_row_count,
            projected_workspace,
        )
        down_global_scale = _projection_global_scale(
            down_per_tensor_scale,
            down.weight_per_tensor_scale,
        )
        gated_updates_backend.apply_indexed_gated_updates(
            projected,
            base_2d[start:stop],
            output_2d[start:stop],
            gated_updates,
            gate_layout,
            start,
            ffn_scale=down_global_scale,
            ffn_bias=down.bias,
        )
    return output


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "LinearOperands",
    "PreparationBackend",
    "dynamic_swiglu_scale",
    "run_chunked_swiglu_ffn",
]
