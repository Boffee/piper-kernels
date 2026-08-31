"""Chunked NVFP4 value projection and sparse-Piper preparation."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import (
    HEAD_DIM,
    TILE_ROWS,
    padded_sequence_length,
)

from . import _epilogue
from ._chunking import PreparedProjection, run_chunked_projection
from ._validation import validate_projection
from .query import DEFAULT_CHUNK_ROWS


def _launch_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    value_mean: torch.Tensor,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
            start,
            sequence_length,
        )

    consumer_tensors = [input_per_tensor_scale, value_mean]
    consumer_tensors.extend(
        operand for operand in (weight_per_tensor_scale, bias) if operand is not None
    )
    run_chunked_projection(
        projection,
        chunk_rows,
        consume,
        (*consumer_tensors, value, value_scale),
    )
    return value, value_scale


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
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = input_qdata.shape[0]
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // HEAD_DIM
    return (
        input_qdata.new_empty((1, heads, HEAD_DIM, storage_sequence_length), dtype=torch.int8),
        input_qdata.new_empty(
            (1, heads, storage_sequence_length // TILE_ROWS, 1),
            dtype=torch.float32,
        ),
    )


__all__ = ["project_value"]
