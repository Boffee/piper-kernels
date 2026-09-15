"""Bounded-workspace composition of a standard NVFP4 GELU feed-forward network."""

from __future__ import annotations

import math
from typing import cast

import torch

from piper_kernels.fusions.ffn import triton as indexed_updates
from piper_kernels.fusions.nvfp4_ffn import _core
from piper_kernels.fusions.nvfp4_ffn._preparation import StandardSourcePreparation
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

from . import _operands
from ._preparation import StandardGELUPreparation

_WORKSPACE_LIMIT_BYTES = 512 * 1_024 * 1_024
_CHUNK_ROW_GRANULARITY = 128


def _prepared_bytes_per_row(features: int) -> int:
    """Return exact packed-data and padded-scale storage for an aligned row tile."""
    tile_rows = nvfp4_layout.SCALE_ROW_TILE
    qdata_shape = cast(tuple[int, int], nvfp4_layout.qdata_shape(tile_rows, features))
    scale_shape = cast(tuple[int, int], nvfp4_layout.scale_shape(tile_rows, features))
    return (math.prod(qdata_shape) + math.prod(scale_shape)) // tile_rows


def _default_chunk_rows(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    down_weight_qdata: torch.Tensor,
    *,
    gated_updates: bool,
) -> int:
    """Bound GELU scratch while amortizing long-sequence chunk launches."""
    input_features = 2 * up_weight_qdata.shape[1]
    intermediate_features = up_weight_qdata.shape[0]
    output_features = down_weight_qdata.shape[0]
    if not all(
        isinstance(features, int)
        for features in (input_features, intermediate_features, output_features)
    ):
        return _core.DEFAULT_CHUNK_ROWS
    source_preparation_bytes = _prepared_bytes_per_row(input_features)
    activation_preparation_bytes = _prepared_bytes_per_row(intermediate_features)
    projection_bytes = intermediate_features * input.element_size()
    projected_output_bytes = (
        output_features * input.element_size()
        if gated_updates and output_features > intermediate_features
        else 0
    )
    bytes_per_row = (
        source_preparation_bytes
        + projection_bytes
        + activation_preparation_bytes
        + projected_output_bytes
    )
    rows = _WORKSPACE_LIMIT_BYTES // bytes_per_row
    return max(
        _CHUNK_ROW_GRANULARITY,
        rows // _CHUNK_ROW_GRANULARITY * _CHUNK_ROW_GRANULARITY,
    )


def _run_chunked_gelu_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
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
    return _core.run_chunked_ffn(
        input,
        (up,),
        down,
        chunk_rows,
        StandardSourcePreparation(up_high_first),
        StandardGELUPreparation(down_high_first),
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
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
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
    return _core.validate_ffn(
        input,
        (up,),
        down,
        chunk_rows,
        StandardGELUPreparation(down_high_first),
    )[2]


@torch.library.custom_op("piper_kernels::nvfp4_gelu_ffn", mutates_args=())
def _chunked_gelu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
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
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
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
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
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
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
        chunk_rows,
    )
    return input.new_empty((*input.shape[:-1], output_features))


@torch.library.custom_op(
    "piper_kernels::nvfp4_gelu_ffn_gated_updates_",
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
    up_high_first: bool,
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
    _run_chunked_gelu_ffn(
        input,
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
    up_high_first: bool,
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
    output_features = _validate_gelu_ffn(
        input,
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
