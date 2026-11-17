"""ConvRot INT8 projection directly into sparse Piper's requested V storage.

``value_format`` is a static ValueFormat ID. The compiler selects it from the
attention consumer; fake implementations use it without querying a device.
Quantization groups and per-tile means do not depend on storage.
"""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.fusions.convrot_int8_projection._validation import validate_projection_inputs

from . import _backend
from ._layout import SUPPORTED_HEAD_DIMS, TILE_ROWS, padded_sequence_length, validate_block_lengths


def _validate_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    head_dim: int = 128,
    *,
    bias: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError("V projection requires head_dim=64 or 128")
    batch, sequence_length, heads = validate_projection_inputs(
        input_qdata, input_scale, weight_qdata, weight_scale, bias, head_dim=head_dim, name="V"
    )
    if sequence_length < TILE_ROWS:
        raise ValueError(f"V projection requires at least {TILE_ROWS} sequence rows")
    return batch, sequence_length, heads


def _launch_value_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    head_dim: int = 128,
    bias: torch.Tensor | None = None,
    value_format: int = int(ValueFormat.TRANSPOSED),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, heads = _validate_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        head_dim,
        bias=bias,
    )
    validate_block_lengths(block_lengths, sequence_length, input_qdata.device)
    storage_sequence_length = padded_sequence_length(sequence_length)
    storage_tiles = storage_sequence_length // TILE_ROWS
    storage_format = ValueFormat(value_format)
    backend = _backend.require_projection_backend(input_qdata, head_dim=head_dim)
    value = torch.empty(
        storage_format.shape(batch, heads, storage_sequence_length, head_dim),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    value_scale_multiplier = torch.empty(
        (batch, heads, storage_tiles, 1),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    value_mean = torch.empty(
        (batch, heads, storage_tiles, head_dim),
        device=input_qdata.device,
        dtype=torch.bfloat16,
    )
    backend.project_value(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        block_lengths,
        value_format=storage_format,
        out=(value, value_scale_multiplier, value_mean),
        bias=bias,
    )
    return value, value_scale_multiplier, value_mean


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_project_value",
    mutates_args=(),
)
def _project_value_op(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
    head_dim: int = 128,
    bias: torch.Tensor | None = None,
    value_format: int = int(ValueFormat.TRANSPOSED),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_value_projection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        block_lengths,
        value_format=value_format,
        head_dim=head_dim,
        bias=bias,
    )


@_project_value_op.register_fake
def _project_value_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _block_lengths: torch.Tensor | None = None,
    head_dim: int = 128,
    _bias: torch.Tensor | None = None,
    value_format: int = int(ValueFormat.TRANSPOSED),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    storage_sequence_length = padded_sequence_length(sequence_length)
    storage_tiles = storage_sequence_length // TILE_ROWS
    heads = weight_qdata.shape[0] // head_dim
    return (
        input_qdata.new_empty(
            ValueFormat(value_format).shape(batch, heads, storage_sequence_length, head_dim)
        ),
        input_qdata.new_empty(
            (batch, heads, storage_tiles, 1),
            dtype=torch.float32,
        ),
        input_qdata.new_empty((batch, heads, storage_tiles, head_dim), dtype=torch.bfloat16),
    )
