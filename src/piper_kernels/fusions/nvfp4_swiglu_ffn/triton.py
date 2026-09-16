"""Bounded-workspace composition of an NVFP4 SwiGLU feed-forward network."""

from __future__ import annotations

import torch

from piper_kernels.fusions.ffn import triton as indexed_updates
from piper_kernels.fusions.nvfp4_ffn import _core
from piper_kernels.fusions.nvfp4_ffn._preparation import StandardSourcePreparation

from . import _operands
from ._preparation import StandardSwiGLUPreparation

_DEFAULT_CHUNK_ROWS = _core.DEFAULT_CHUNK_ROWS


@torch.library.custom_op("piper_kernels::nvfp4_swiglu_ffn", mutates_args=())
def _chunked_swiglu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
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
    chunk_rows: int,
) -> torch.Tensor:
    gate, value, down = _operands.linear_operands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_high_first,
        value_weight_qdata,
        value_weight_scale,
        value_weight_per_tensor_scale,
        value_activation_per_tensor_scale,
        value_bias,
        value_dynamic_activation_scale,
        value_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
    )
    return _core.run_chunked_ffn(
        input,
        (value, gate),
        down,
        chunk_rows,
        StandardSourcePreparation(gate_high_first),
        StandardSwiGLUPreparation(down_high_first),
    )


@_chunked_swiglu_ffn_op.register_fake
def _chunked_swiglu_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002
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
    chunk_rows: int,
) -> torch.Tensor:
    gate, value, down = _operands.linear_operands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_high_first,
        value_weight_qdata,
        value_weight_scale,
        value_weight_per_tensor_scale,
        value_activation_per_tensor_scale,
        value_bias,
        value_dynamic_activation_scale,
        value_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
    )
    output_features = _core.validate_ffn(
        input,
        (value, gate),
        down,
        chunk_rows,
        StandardSourcePreparation(gate_high_first),
        StandardSwiGLUPreparation(down_high_first),
    )[2]
    return input.new_empty((*input.shape[:-1], output_features))


@torch.library.custom_op(
    "piper_kernels::nvfp4_swiglu_ffn_gated_updates_",
    mutates_args=("reusable_update",),
)
def _chunked_swiglu_ffn_gated_updates_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
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
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
) -> None:
    gate, value, down = _operands.linear_operands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_high_first,
        value_weight_qdata,
        value_weight_scale,
        value_weight_per_tensor_scale,
        value_activation_per_tensor_scale,
        value_bias,
        value_dynamic_activation_scale,
        value_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
    )
    _core.run_chunked_ffn(
        input,
        (value, gate),
        down,
        chunk_rows,
        StandardSourcePreparation(gate_high_first),
        StandardSwiGLUPreparation(down_high_first),
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
    input: torch.Tensor,  # noqa: A002 - match linear terminology
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
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
) -> None:
    gate, value, down = _operands.linear_operands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_high_first,
        value_weight_qdata,
        value_weight_scale,
        value_weight_per_tensor_scale,
        value_activation_per_tensor_scale,
        value_bias,
        value_dynamic_activation_scale,
        value_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
    )
    output_features = _core.validate_ffn(
        input,
        (value, gate),
        down,
        chunk_rows,
        StandardSourcePreparation(gate_high_first),
        StandardSwiGLUPreparation(down_high_first),
    )[2]
    indexed_updates.validate_indexed_gated_updates(
        input,
        indexed_updates.IndexedGatedUpdates(
            base,
            reusable_update,
            update_gate,
            ffn_gate,
            gate_indices,
            python_indexing,
        ),
        output_features,
    )


__all__ = [
    "_chunked_swiglu_ffn_gated_updates_op",
    "_chunked_swiglu_ffn_op",
]
