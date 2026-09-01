"""Shared bounded-workspace execution for NVFP4 SwiGLU FFNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch

from piper_kernels.fusions.swiglu_ffn import triton as gated_updates_backend
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


class PreparationBackend(Protocol):
    """Format-specific activation preparation used by the shared FFN runner."""

    def prepare_up(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        up: LinearOperands,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the FFN input for the up projection."""
        ...

    def dynamic_down_scale(
        self,
        packed: torch.Tensor,
        up_global_scale: torch.Tensor,
        up_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Calculate one chunk's post-SwiGLU activation scale."""
        ...

    def prepare_down(
        self,
        packed: torch.Tensor,
        down_per_tensor_scale: torch.Tensor,
        up_global_scale: torch.Tensor,
        up_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare one projected SwiGLU chunk for the down projection."""
        ...


def linear_operands(
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_activation_scale: bool,
) -> LinearOperands:
    """Group one projection's public custom-op schema into canonical operands."""
    return LinearOperands(
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        dynamic_activation_scale,
    )


def _validate_inputs(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up: LinearOperands,
    down: LinearOperands,
    chunk_rows: int,
) -> tuple[int, int]:
    up_shape = nvfp4_validation.validate_semantic_linear(
        input,
        up.weight_qdata,
        up.weight_scale,
        up.weight_per_tensor_scale,
        up.activation_per_tensor_scale,
        up.bias,
        up.dynamic_activation_scale,
        "chunked NVFP4 FFN up projection",
    )
    if input.dtype is not torch.bfloat16:
        raise ValueError("chunked NVFP4 FFN currently requires BF16 activations")
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < _SCALE_ROW_BLOCK
        or chunk_rows % _SCALE_ROW_BLOCK
    ):
        raise ValueError("chunked NVFP4 FFN chunk_rows must be a positive multiple of 128")
    up_features = up_shape.output_features
    if not isinstance(up_features, int) or up_features % 2:
        raise ValueError("chunked NVFP4 FFN up projection must contain equal up/gate halves")
    intermediate_features = up_features // 2
    nvfp4_validation.validate_activation_scale(
        down.activation_per_tensor_scale,
        down.dynamic_activation_scale,
        input.device,
        "chunked NVFP4 FFN down projection",
    )
    output_features = nvfp4_validation.validate_weight(
        down.weight_qdata,
        down.weight_scale,
        down.weight_per_tensor_scale,
        down.bias,
        input_features=intermediate_features,
        logical_dtype=input.dtype,
        device=input.device,
        name="chunked NVFP4 FFN down projection",
    )
    if not isinstance(output_features, int):
        raise ValueError("chunked NVFP4 FFN requires concrete projection dimensions")
    differentiable_tensors = (
        input,
        up.weight_scale,
        down.weight_scale,
        *(
            tensor
            for tensor in (
                up.weight_per_tensor_scale,
                up.activation_per_tensor_scale,
                up.bias,
                down.weight_per_tensor_scale,
                down.activation_per_tensor_scale,
                down.bias,
            )
            if tensor is not None
        ),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in differentiable_tensors):
        raise RuntimeError("chunked NVFP4 FFN is inference-only and does not support autograd")
    return cast(int, up_shape.rows), output_features


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


def _project_affine_down_chunk(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    linear: LinearOperands,
    rows: int,
    output: torch.Tensor,
    global_scale: torch.Tensor | None,
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
    assert global_scale is not None
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
        global_scale,
        linear.bias,
        projected,
    )


def run_chunked_swiglu_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up: LinearOperands,
    down: LinearOperands,
    chunk_rows: int,
    preparation: PreparationBackend,
    *,
    gated_updates: gated_updates_backend.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Run an NVFP4 FFN while bounding its materialized BF16 intermediate.

    Down-projection affine terms run in the GEMM accumulator before its output conversion.
    Dynamic down scales intentionally cover one chunk rather than the otherwise-unavailable full
    intermediate.
    """
    rows, output_features = _validate_inputs(
        input,
        up,
        down,
        chunk_rows,
    )
    leading_shape = input.shape[:-1]
    up_qdata, up_scale, up_per_tensor_scale = preparation.prepare_up(input, up)
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
    workspace_rows = min(rows, chunk_rows)
    packed_workspace = torch.empty(
        (workspace_rows, up.weight_qdata.shape[0]),
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
    up_global_scale = _projection_global_scale(
        up_per_tensor_scale,
        up.weight_per_tensor_scale,
    )
    static_down_scale = down.activation_per_tensor_scale
    static_down_global_scale = None
    if not down.dynamic_activation_scale:
        assert static_down_scale is not None
        if gated_updates is not None or down.weight_per_tensor_scale is None:
            static_down_global_scale = _projection_global_scale(
                static_down_scale,
                down.weight_per_tensor_scale,
            )

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        packed = _project_chunk_out(
            up_qdata,
            up_scale,
            up,
            start,
            stop,
            packed_workspace,
        )
        down_per_tensor_scale = static_down_scale
        down_global_scale = static_down_global_scale
        if down.dynamic_activation_scale:
            down_per_tensor_scale = preparation.dynamic_down_scale(
                packed,
                up_global_scale,
                up.bias,
            )
            if gated_updates is not None or down.weight_per_tensor_scale is None:
                down_global_scale = _projection_global_scale(
                    down_per_tensor_scale,
                    down.weight_per_tensor_scale,
                )
        assert down_per_tensor_scale is not None
        down_qdata, down_scale = preparation.prepare_down(
            packed,
            down_per_tensor_scale,
            up_global_scale,
            up.bias,
        )
        if gated_updates is None:
            _project_affine_down_chunk(
                down_qdata,
                down_scale,
                down_per_tensor_scale,
                down,
                stop - start,
                output_2d[start:stop],
                down_global_scale,
            )
            continue
        assert projected_workspace is not None
        assert base_2d is not None
        assert gate_layout is not None
        assert down_global_scale is not None
        projected = _project_chunk_out(
            down_qdata,
            down_scale,
            down,
            0,
            stop - start,
            projected_workspace,
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
    "linear_operands",
    "run_chunked_swiglu_ffn",
]
