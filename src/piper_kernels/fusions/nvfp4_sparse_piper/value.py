"""Chunked NVFP4 value projection and sparse-Piper preparation."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import (
    HEAD_DIM,
    TILE_ROWS,
    padded_sequence_length,
)
from piper_kernels.linear.nvfp4._chunking import (
    DEFAULT_CHUNK_ROWS,
    PreparedProjection,
    run_chunked_projection,
)

from . import _epilogue
from ._validation import validate_block_lengths, validate_projection


def _launch_value(  # noqa: PLR0913
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    value_mean: torch.Tensor,
    chunk_rows: int,
    block_lengths: torch.Tensor | None,
    *,
    emit_block_mean: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length, heads = validate_projection(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        chunk_rows,
        "V projection",
    )
    if (
        value_mean.shape != (1, heads, HEAD_DIM)
        or value_mean.dtype is not torch.float32
        or value_mean.device != input_qdata.device
        or not value_mean.is_contiguous()
    ):
        raise ValueError("V projection mean must be a contiguous [1,heads,D128] FP32 tensor")
    validate_block_lengths(
        block_lengths,
        sequence_length,
        input_qdata.device,
        "V projection",
    )
    storage_sequence_length = padded_sequence_length(sequence_length)
    value = torch.empty(
        (1, heads, HEAD_DIM, storage_sequence_length),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    value_scale = torch.empty(
        (1, heads, storage_sequence_length // TILE_ROWS, 1),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    block_mean = (
        torch.empty(
            (1, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM),
            device=input_qdata.device,
            dtype=torch.float32,
        )
        if emit_block_mean
        else value_mean
    )
    projection = PreparedProjection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
    )

    def consume(chunk: torch.Tensor, start: int) -> None:
        _epilogue.launch_value(
            chunk,
            input_per_tensor_scale,
            weight_per_tensor_scale,
            bias,
            value_mean,
            value,
            value_scale,
            block_mean,
            block_lengths,
            start,
            sequence_length,
            emit_block_mean=emit_block_mean,
        )

    consumer_tensors = [input_per_tensor_scale, value_mean, value, value_scale]
    consumer_tensors.extend(
        operand for operand in (weight_per_tensor_scale, bias) if operand is not None
    )
    if block_lengths is not None:
        consumer_tensors.append(block_lengths)
    if emit_block_mean:
        consumer_tensors.append(block_mean)
    run_chunked_projection(
        projection,
        chunk_rows,
        consume,
        consumer_tensors,
    )
    return value, value_scale, block_mean


@torch.library.custom_op("piper_kernels::nvfp4_sparse_piper_project_value", mutates_args=())
def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    value_mean: torch.Tensor,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    value, value_scale, _block_mean = _launch_value(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        value_mean,
        chunk_rows,
        block_lengths,
        emit_block_mean=False,
    )
    return value, value_scale


def _fake_value_projection(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = input_qdata.shape[0]
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // HEAD_DIM
    return (
        input_qdata.new_empty((1, heads, HEAD_DIM, storage_sequence_length), dtype=torch.int8),
        input_qdata.new_empty(
            (1, heads, storage_sequence_length // TILE_ROWS, 1),
            dtype=torch.float32,
        ),
        input_qdata.new_empty(
            (1, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM),
            dtype=torch.float32,
        ),
    )


@project_value.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _project_value_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _value_mean: torch.Tensor,
    _chunk_rows: int = DEFAULT_CHUNK_ROWS,
    _block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    value, value_scale, _block_mean = _fake_value_projection(input_qdata, weight_qdata)
    return value, value_scale


@torch.library.custom_op(
    "piper_kernels::nvfp4_sparse_piper_project_value_with_block_means",
    mutates_args=(),
)
def project_value_with_block_means(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    value_mean: torch.Tensor,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_value(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        value_mean,
        chunk_rows,
        block_lengths,
        emit_block_mean=True,
    )


@project_value_with_block_means.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _project_value_with_block_means_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _value_mean: torch.Tensor,
    _chunk_rows: int = DEFAULT_CHUNK_ROWS,
    _block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fake_value_projection(input_qdata, weight_qdata)


__all__ = ["project_value", "project_value_with_block_means"]
