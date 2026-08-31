"""Bounded-workspace composition of an NVFP4 SwiGLU feed-forward network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.fusions.swiglu_ffn import triton as gated_updates_backend
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import _projection as nvfp4_projection
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend

_DEFAULT_CHUNK_ROWS = 1_536
_SCALE_ROW_BLOCK = 128


@dataclass(frozen=True, slots=True)
class _LinearOperands:
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    weight_per_tensor_scale: torch.Tensor | None
    activation_per_tensor_scale: torch.Tensor | None
    bias: torch.Tensor | None
    dynamic_activation_scale: bool


def _ffn_operands(
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
) -> tuple[_LinearOperands, _LinearOperands]:
    """Group the public custom-op schema into canonical projection operands."""
    return (
        _LinearOperands(
            up_weight_qdata,
            up_weight_scale,
            up_weight_per_tensor_scale,
            up_activation_per_tensor_scale,
            up_bias,
            up_dynamic_activation_scale,
        ),
        _LinearOperands(
            down_weight_qdata,
            down_weight_scale,
            down_weight_per_tensor_scale,
            down_activation_per_tensor_scale,
            down_bias,
            down_dynamic_activation_scale,
        ),
    )


def _validate_inputs(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up: _LinearOperands,
    down: _LinearOperands,
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
    linear: _LinearOperands,
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


def _dynamic_swiglu_scale(input: torch.Tensor) -> torch.Tensor:  # noqa: A002
    up, gate = input.chunk(2, dim=-1)
    activated = up * F.silu(gate)
    return per_tensor_amax_to_scale(activated.abs().amax())


def _projected_swiglu_scale(
    input: torch.Tensor,  # noqa: A002
    global_scale: torch.Tensor,
) -> torch.Tensor:
    projected = (input.float() * global_scale.float()).to(input.dtype)
    return _dynamic_swiglu_scale(projected)


def _projected_swiglu_scale_and_add_bias(
    input: torch.Tensor,  # noqa: A002
    global_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    projected = (input.float() * global_scale.float() + bias.float()).to(input.dtype)
    return _dynamic_swiglu_scale(projected)


_compiled_projected_swiglu_scale = torch.compile(_projected_swiglu_scale, fullgraph=True)
_compiled_projected_swiglu_scale_and_add_bias = torch.compile(
    _projected_swiglu_scale_and_add_bias,
    fullgraph=True,
)


def _projected_swiglu_dynamic_scale(
    input: torch.Tensor,  # noqa: A002
    global_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Calculate a dynamic scale without materializing the projected SwiGLU."""
    if bias is None:
        return _compiled_projected_swiglu_scale(input, global_scale)
    return _compiled_projected_swiglu_scale_and_add_bias(input, global_scale, bias)


def _project_affine_down_chunk(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    linear: _LinearOperands,
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


def _run_chunked_swiglu_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up: _LinearOperands,
    down: _LinearOperands,
    chunk_rows: int,
    *,
    gated_updates: gated_updates_backend.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Run one NVFP4 FFN while bounding its materialized BF16 intermediate.

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
    assert isinstance(rows, int)
    leading_shape = input.shape[:-1]
    up_qdata, up_scale, up_per_tensor_scale = nvfp4_ops._prepare_compiled(
        input,
        up.activation_per_tensor_scale,
        up.dynamic_activation_scale,
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
            down_per_tensor_scale = _projected_swiglu_dynamic_scale(
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
        down_qdata, down_scale = nvfp4_backend.prepare_static_projected_swiglu(
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


@torch.library.custom_op("piper_kernels::nvfp4_swiglu_ffn", mutates_args=())
def _chunked_swiglu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    chunk_rows: int,
) -> torch.Tensor:
    up, down = _ffn_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
    )
    return _run_chunked_swiglu_ffn(
        input,
        up,
        down,
        chunk_rows,
    )


@_chunked_swiglu_ffn_op.register_fake
def _chunked_swiglu_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002
    _up_weight_qdata: torch.Tensor,
    _up_weight_scale: torch.Tensor,
    _up_weight_per_tensor_scale: torch.Tensor | None,
    _up_activation_per_tensor_scale: torch.Tensor | None,
    _up_bias: torch.Tensor | None,
    _up_dynamic_activation_scale: bool,
    down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_weight_per_tensor_scale: torch.Tensor | None,
    _down_activation_per_tensor_scale: torch.Tensor | None,
    _down_bias: torch.Tensor | None,
    _down_dynamic_activation_scale: bool,
    _chunk_rows: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], down_weight_qdata.shape[0]))


@torch.library.custom_op(
    "piper_kernels::nvfp4_swiglu_ffn_gated_updates_",
    mutates_args=("reusable_update",),
)
def _chunked_swiglu_ffn_gated_updates_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
) -> None:
    up, down = _ffn_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
    )
    _run_chunked_swiglu_ffn(
        input,
        up,
        down,
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
    _up_weight_qdata: torch.Tensor,
    _up_weight_scale: torch.Tensor,
    _up_weight_per_tensor_scale: torch.Tensor | None,
    _up_activation_per_tensor_scale: torch.Tensor | None,
    _up_bias: torch.Tensor | None,
    _up_dynamic_activation_scale: bool,
    _down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_weight_per_tensor_scale: torch.Tensor | None,
    _down_activation_per_tensor_scale: torch.Tensor | None,
    _down_bias: torch.Tensor | None,
    _down_dynamic_activation_scale: bool,
    _base: torch.Tensor,
    _reusable_update: torch.Tensor,
    _update_gate: torch.Tensor,
    _ffn_gate: torch.Tensor,
    _gate_indices: torch.Tensor,
    _python_indexing: bool,
    _chunk_rows: int,
) -> None:
    return None


__all__ = ["_chunked_swiglu_ffn_gated_updates_op", "_chunked_swiglu_ffn_op"]
