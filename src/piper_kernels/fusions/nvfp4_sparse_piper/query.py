"""Chunked NVFP4 query projection and sparse-Piper preparation."""

from __future__ import annotations

import math

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import (
    HEAD_DIM,
    QUERY_SCALE_ROWS,
    TILE_ROWS,
    padded_sequence_length,
)

from . import _epilogue
from ._chunking import PreparedProjection, run_chunked_projection
from ._validation import validate_projection, validate_qk_epilogue

DEFAULT_CHUNK_ROWS = 4096


def _launch_query(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    chunk_rows: int,
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
        "Q projection",
    )
    validate_qk_epilogue(
        input_qdata,
        sequence_length,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        "Q projection",
    )
    operands = (norm_weight, cos, sin)
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Q projection softmax scale must be finite and positive")
    storage_sequence_length = padded_sequence_length(sequence_length)
    query = torch.empty(
        (1, heads, storage_sequence_length, HEAD_DIM),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    query_scale = torch.empty(
        (1, heads, storage_sequence_length // QUERY_SCALE_ROWS),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    query_summary = torch.empty(
        (1, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM),
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
        _epilogue.launch_query(
            chunk,
            input_per_tensor_scale,
            weight_per_tensor_scale,
            bias,
            norm_weight,
            cos,
            sin,
            query,
            query_scale,
            query_summary,
            start,
            sequence_length,
            norm_epsilon,
            softmax_scale,
        )

    consumer_tensors = [input_per_tensor_scale, *operands]
    consumer_tensors.extend(
        operand for operand in (weight_per_tensor_scale, bias) if operand is not None
    )
    run_chunked_projection(
        projection,
        chunk_rows,
        consume,
        (*consumer_tensors, query, query_scale, query_summary),
    )
    return query, query_scale, query_summary


@torch.library.custom_op("piper_kernels::nvfp4_sparse_piper_project_query", mutates_args=())
def project_query(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_query(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        chunk_rows,
    )


@project_query.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _project_query_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _norm_weight: torch.Tensor,
    cos: torch.Tensor,
    _sin: torch.Tensor,
    _norm_epsilon: float,
    _softmax_scale: float,
    _chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = input_qdata.shape[0]
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // HEAD_DIM
    return (
        input_qdata.new_empty((1, heads, storage_sequence_length, HEAD_DIM), dtype=torch.int8),
        input_qdata.new_empty(
            (1, heads, storage_sequence_length // QUERY_SCALE_ROWS),
            dtype=torch.float32,
        ),
        cos.new_empty(
            (1, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM),
            dtype=torch.float32,
        ),
    )


__all__ = ["DEFAULT_CHUNK_ROWS", "project_query"]
