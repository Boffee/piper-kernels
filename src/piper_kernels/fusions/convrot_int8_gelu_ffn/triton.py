"""Bounded-workspace ConvRot INT8 GELU custom operations."""

from __future__ import annotations

import torch

from piper_kernels.fusions.convrot_int8_ffn import _core
from piper_kernels.fusions.ffn import triton as indexed_updates

_WORKSPACE_LIMIT_BYTES = 1_024 * 1_024 * 1_024
_CHUNK_ROW_GRANULARITY = 128


def _default_chunk_rows(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    down_weight_qdata: torch.Tensor,
    *,
    gated_updates: bool,
) -> int:
    """Bound reusable row scratch while amortizing per-chunk launches.

    ConvRot backends independently choose their power-of-two feature chunks.
    """
    input_features = up_weight_qdata.shape[1]
    intermediate_features = up_weight_qdata.shape[0]
    output_features = down_weight_qdata.shape[0]
    if not all(
        isinstance(features, int)
        for features in (input_features, intermediate_features, output_features)
    ):
        return _core.DEFAULT_CHUNK_ROWS
    projection_bytes = intermediate_features * input.element_size()
    preparation_bytes = max(input_features, intermediate_features) * torch.int8.itemsize
    scale_bytes = torch.float32.itemsize
    projected_output_bytes = (
        output_features * input.element_size()
        if gated_updates and output_features > intermediate_features
        else 0
    )
    bytes_per_row = projection_bytes + preparation_bytes + scale_bytes + projected_output_bytes
    rows = _WORKSPACE_LIMIT_BYTES // bytes_per_row
    return max(_CHUNK_ROW_GRANULARITY, rows // _CHUNK_ROW_GRANULARITY * _CHUNK_ROW_GRANULARITY)


def _run_chunked_gelu_ffn(
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
    up_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
    *,
    gated_updates: indexed_updates.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Run the public up/down topology through the shared bounded runner."""
    up = _core.LinearOperands(
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        up_input_scale,
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
        (up,),
        down,
        "gelu_tanh",
        chunk_rows,
        gated_updates=gated_updates,
    )


def _validate_gelu_ffn(
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
    up_input_scale: torch.Tensor | None,
    down_input_scale: torch.Tensor | None,
) -> int:
    """Validate fake/runtime metadata through the shared bounded-runner contract."""
    up = _core.LinearOperands(
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        up_input_scale,
    )
    down = _core.LinearOperands(
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        down_input_scale,
    )
    return _core.validate_ffn(input, (up,), down, "gelu_tanh", chunk_rows)[2]


@torch.library.custom_op("piper_kernels::convrot_int8_gelu_ffn", mutates_args=())
def _chunked_gelu_ffn_op(
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
    up_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    return _run_chunked_gelu_ffn(
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
        up_input_scale,
        down_input_scale,
    )


@_chunked_gelu_ffn_op.register_fake
def _chunked_gelu_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    chunk_rows: int,
    up_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    output_features = _validate_gelu_ffn(
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
        up_input_scale,
        down_input_scale,
    )
    return input.new_empty((*input.shape[:-1], output_features))


@torch.library.custom_op(
    "piper_kernels::convrot_int8_gelu_ffn_gated_updates_",
    mutates_args=("reusable_update",),
)
def _chunked_gelu_ffn_gated_updates_op(
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
    up_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> None:
    _run_chunked_gelu_ffn(
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
        up_input_scale,
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


@_chunked_gelu_ffn_gated_updates_op.register_fake
def _chunked_gelu_ffn_gated_updates_op_fake(
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
    up_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> None:
    output_features = _validate_gelu_ffn(
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
        up_input_scale,
        down_input_scale,
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
