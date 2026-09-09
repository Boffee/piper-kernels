"""ConvRot INT8 projection and tile-scaled INT8 V preparation for sparse Piper."""

from __future__ import annotations

import torch

from piper_kernels.linear import _bias

from . import _backend
from ._layout import SUPPORTED_HEAD_DIMS, TILE_ROWS, padded_sequence_length, validate_block_lengths


def _validate_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    head_dim: int = 128,
    *,
    bias: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError("V projection requires head_dim=64 or 128")
    if input_qdata.ndim != 3 or input_qdata.dtype is not torch.int8:
        raise ValueError("V projection input must be [batch,sequence,features] INT8")
    batch, sequence_length, input_features = input_qdata.shape
    if input_scale.shape != (batch, sequence_length) or input_scale.dtype is not torch.float32:
        raise ValueError("V projection input scale must be a batch/sequence FP32 matrix")
    if input_mean.shape != (batch, input_features) or input_mean.dtype is not torch.float32:
        raise ValueError("V projection represented-input mean must be a batch/feature FP32 matrix")
    if weight_qdata.ndim != 2 or weight_qdata.dtype is not torch.int8:
        raise ValueError("V projection weight must be a two-dimensional INT8 tensor")
    if weight_qdata.shape[1] != input_features or weight_qdata.shape[0] % head_dim:
        raise ValueError("V projection weight must map the input to complete D64/D128 heads")
    if weight_scale.shape != (weight_qdata.shape[0], 1) or weight_scale.dtype is not torch.float32:
        raise ValueError("V projection weight scale must be one FP32 value per output feature")
    if bias is not None:
        _bias.validate_dtype(bias, "V projection")
        if bias.shape != (weight_qdata.shape[0],):
            raise ValueError("V projection bias must have one value per output feature")
    operands = tuple(
        operand
        for operand in (input_qdata, input_scale, input_mean, weight_qdata, weight_scale, bias)
        if operand is not None
    )
    if any(operand.device != input_qdata.device for operand in operands):
        raise ValueError("V projection operands must share a device")
    if any(
        operand.layout is not torch.strided or not operand.is_contiguous() for operand in operands
    ):
        raise ValueError("V projection operands must be contiguous")
    if sequence_length < TILE_ROWS:
        raise ValueError(f"V projection requires at least {TILE_ROWS} sequence rows")
    return batch, sequence_length, weight_qdata.shape[0] // head_dim


def _launch_value_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    emit_block_mean: bool,
    head_dim: int = 128,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, heads = _validate_inputs(
        input_qdata,
        input_scale,
        input_mean,
        weight_qdata,
        weight_scale,
        head_dim,
        bias=bias,
    )
    validate_block_lengths(block_lengths, sequence_length, input_qdata.device)
    storage_sequence_length = padded_sequence_length(sequence_length)
    backend = _backend.require_projection_backend(input_qdata, head_dim=head_dim)
    value = torch.empty(
        (batch, heads, head_dim, storage_sequence_length),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    value_scale_multiplier = torch.empty(
        (batch, heads, storage_sequence_length // TILE_ROWS, 1),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    value_mean = torch.empty(
        (batch, heads, head_dim),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    block_mean = (
        torch.empty(
            (batch, heads, storage_sequence_length // TILE_ROWS, head_dim),
            device=input_qdata.device,
            dtype=torch.float32,
        )
        if emit_block_mean
        else value_mean
    )
    backend.project_value(
        input_qdata,
        input_scale,
        input_mean,
        weight_qdata,
        weight_scale,
        block_lengths,
        emit_block_mean=emit_block_mean,
        out=(value, value_scale_multiplier, value_mean, block_mean),
        bias=bias,
    )
    return value, value_scale_multiplier, value_mean, block_mean


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_project_value",
    mutates_args=(),
)
def _project_value_op(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
    head_dim: int = 128,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, value_scale_multiplier, value_mean, _block_mean = _launch_value_projection(
        input_qdata,
        input_scale,
        input_mean,
        weight_qdata,
        weight_scale,
        block_lengths,
        emit_block_mean=False,
        head_dim=head_dim,
        bias=bias,
    )
    return value, value_scale_multiplier, value_mean


def _fake_value_projection(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // head_dim
    return (
        input_qdata.new_empty((batch, heads, head_dim, storage_sequence_length)),
        input_qdata.new_empty(
            (batch, heads, storage_sequence_length // TILE_ROWS, 1),
            dtype=torch.float32,
        ),
        input_qdata.new_empty((batch, heads, head_dim), dtype=torch.float32),
        input_qdata.new_empty(
            (batch, heads, storage_sequence_length // TILE_ROWS, head_dim),
            dtype=torch.float32,
        ),
    )


@_project_value_op.register_fake
def _project_value_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _block_lengths: torch.Tensor | None = None,
    head_dim: int = 128,
    _bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, value_scale_multiplier, value_mean, _block_mean = _fake_value_projection(
        input_qdata,
        weight_qdata,
        head_dim,
    )
    return value, value_scale_multiplier, value_mean


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_project_value_with_block_means",
    mutates_args=(),
)
def _project_value_with_block_means_op(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
    head_dim: int = 128,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_value_projection(
        input_qdata,
        input_scale,
        input_mean,
        weight_qdata,
        weight_scale,
        block_lengths,
        emit_block_mean=True,
        head_dim=head_dim,
        bias=bias,
    )


@_project_value_with_block_means_op.register_fake
def _project_value_with_block_means_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _block_lengths: torch.Tensor | None = None,
    head_dim: int = 128,
    _bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fake_value_projection(input_qdata, weight_qdata, head_dim)
