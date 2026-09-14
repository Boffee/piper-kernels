"""Bounded-workspace ConvRot INT8 SwiGLU custom operations."""

from __future__ import annotations

import torch

from piper_kernels.fusions.convrot_int8_ffn import _core
from piper_kernels.fusions.ffn import triton as indexed_updates

_DEFAULT_CHUNK_ROWS = _core.DEFAULT_CHUNK_ROWS


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
    gate_input_scale: torch.Tensor | None = None,
    value_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
    *,
    gated_updates: indexed_updates.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Run the public gate/value topology through the shared bounded runner."""
    gate = _core.LinearOperands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        gate_input_scale,
    )
    value = _core.LinearOperands(
        value_weight_qdata,
        value_weight_scale,
        value_bias,
        value_group_size,
        value_input_scale,
    )
    down = _core.LinearOperands(
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        down_input_scale,
    )
    return _core.run_chunked_ffn(
        input,
        (value, gate),
        down,
        "swiglu",
        chunk_rows,
        gated_updates=gated_updates,
    )


@torch.library.custom_op("piper_kernels::convrot_int8_swiglu_ffn", mutates_args=())
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
    gate_input_scale: torch.Tensor | None = None,
    value_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
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
        gate_input_scale,
        value_input_scale,
        down_input_scale,
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
    _gate_input_scale: torch.Tensor | None = None,
    _value_input_scale: torch.Tensor | None = None,
    _down_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], down_weight_qdata.shape[0]))


@torch.library.custom_op(
    "piper_kernels::convrot_int8_swiglu_ffn_gated_updates_",
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
    gate_input_scale: torch.Tensor | None = None,
    value_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
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
        gate_input_scale,
        value_input_scale,
        down_input_scale,
        gated_updates=indexed_updates.IndexedGatedUpdates(
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
    _gate_input_scale: torch.Tensor | None = None,
    _value_input_scale: torch.Tensor | None = None,
    _down_input_scale: torch.Tensor | None = None,
) -> None:
    return None
