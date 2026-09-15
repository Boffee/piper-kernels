"""Bounded-workspace standard/ConvRot NVFP4 GELU custom operations."""

from __future__ import annotations

import torch

from piper_kernels.fusions.convrot_nvfp4_ffn._preparation import ConvRotSourcePreparation
from piper_kernels.fusions.ffn import triton as indexed_updates
from piper_kernels.fusions.nvfp4_ffn import _core
from piper_kernels.fusions.nvfp4_ffn._preparation import StandardSourcePreparation
from piper_kernels.fusions.nvfp4_gelu_ffn import _operands
from piper_kernels.fusions.nvfp4_gelu_ffn._preparation import StandardGELUPreparation

from ._preparation import ConvRotGELUPreparation


def _preparation_backends(
    up_group_size: int | None,
    down_group_size: int | None,
    up_high_first: bool,
    down_high_first: bool,
) -> tuple[_core.SourcePreparationBackend, _core.ActivationPreparationBackend]:
    source = (
        StandardSourcePreparation(up_high_first)
        if up_group_size is None
        else ConvRotSourcePreparation(up_group_size, up_high_first)
    )
    activation = (
        StandardGELUPreparation(down_high_first)
        if down_group_size is None
        else ConvRotGELUPreparation(down_group_size, down_high_first)
    )
    return source, activation


def _run_chunked_gelu_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    chunk_rows: int,
    *,
    gated_updates: indexed_updates.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    up, down = _operands.linear_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
    )
    source_preparation, activation_preparation = _preparation_backends(
        up_group_size,
        down_group_size,
        up_high_first,
        down_high_first,
    )
    return _core.run_chunked_ffn(
        input,
        (up,),
        down,
        chunk_rows,
        source_preparation,
        activation_preparation,
        gated_updates=gated_updates,
    )


def _validate_gelu_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    chunk_rows: int,
) -> int | torch.SymInt:
    """Validate fake/runtime metadata through the shared bounded-runner contract."""
    up, down = _operands.linear_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
    )
    _source_preparation, activation_preparation = _preparation_backends(
        up_group_size,
        down_group_size,
        up_high_first,
        down_high_first,
    )
    return _core.validate_ffn(
        input,
        (up,),
        down,
        chunk_rows,
        activation_preparation,
    )[2]


@torch.library.custom_op("piper_kernels::convrot_nvfp4_gelu_ffn", mutates_args=())
def _chunked_gelu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    chunk_rows: int,
) -> torch.Tensor:
    return _run_chunked_gelu_ffn(
        input,
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
        chunk_rows,
    )


@_chunked_gelu_ffn_op.register_fake
def _chunked_gelu_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    chunk_rows: int,
) -> torch.Tensor:
    output_features = _validate_gelu_ffn(
        input,
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
        chunk_rows,
    )
    return input.new_empty((*input.shape[:-1], output_features))


@torch.library.custom_op(
    "piper_kernels::convrot_nvfp4_gelu_ffn_gated_updates_",
    mutates_args=("reusable_update",),
)
def _chunked_gelu_ffn_gated_updates_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
) -> None:
    _run_chunked_gelu_ffn(
        input,
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
        chunk_rows,
        gated_updates=indexed_updates.IndexedGatedUpdates(
            base=base,
            reusable_update=reusable_update,
            update_gate=update_gate,
            ffn_gate=ffn_gate,
            gate_indices=gate_indices,
            python_indexing=python_indexing,
        ),
    )


@_chunked_gelu_ffn_gated_updates_op.register_fake
def _chunked_gelu_ffn_gated_updates_op_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
) -> None:
    output_features = _validate_gelu_ffn(
        input,
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
        chunk_rows,
    )
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
    "_chunked_gelu_ffn_gated_updates_op",
    "_chunked_gelu_ffn_op",
]
