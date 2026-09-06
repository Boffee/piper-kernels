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
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
    validate_routing_mode,
)
from piper_kernels.linear.nvfp4._chunking import (
    DEFAULT_CHUNK_ROWS,
    PreparedProjection,
    run_chunked_projection,
)
from piper_kernels.linear.nvfp4._projection import matmul_prepared_chunk_out

from . import _epilogue
from ._validation import validate_block_lengths, validate_projection, validate_qk_epilogue


def _launch_query_range(  # noqa: PLR0913, PLR0917
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
    projection_chunk_rows: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    chunk_start: int = 0,
    chunk_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_routing_mode(routing_mode)
    sequence_length, heads = validate_projection(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        projection_chunk_rows,
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
    validate_block_lengths(
        block_lengths,
        sequence_length,
        input_qdata.device,
        "Q projection",
    )
    operands = (norm_weight, cos, sin)
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Q projection softmax scale must be finite and positive")
    if chunk_rows is None:
        chunk_rows = sequence_length
    if (
        isinstance(chunk_start, bool)
        or not isinstance(chunk_start, int)
        or isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_start < 0
        or chunk_rows < 1
        or chunk_start % TILE_ROWS
        or chunk_start + chunk_rows > sequence_length
    ):
        raise ValueError("Q projection range must be a nonempty aligned sequence window")
    storage_sequence_length = padded_sequence_length(chunk_rows)
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
            chunk_start + start,
            norm_epsilon,
            softmax_scale,
            routing_mode == _MEAN_ROUTING,
            block_lengths,
            storage_chunk_start=start,
        )

    consumer_tensors = [input_per_tensor_scale, *operands]
    consumer_tensors.extend(
        operand for operand in (weight_per_tensor_scale, bias) if operand is not None
    )
    if block_lengths is not None:
        consumer_tensors.append(block_lengths)
    # ``run_chunked_projection`` assumes row-zero scale addressing. A local
    # query range therefore projects from the original prepared storage with
    # explicit global row bounds while retaining its bounded BF16 temporary.
    if chunk_start == 0 and chunk_rows == sequence_length:
        run_chunked_projection(
            PreparedProjection(
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
            ),
            projection_chunk_rows,
            consume,
            (*consumer_tensors, query, query_scale, query_summary),
        )
    else:
        projection_buffer = torch.empty(
            (min(chunk_rows, projection_chunk_rows), weight_qdata.shape[0]),
            device=input_qdata.device,
            dtype=torch.bfloat16,
        )
        for local_start in range(0, chunk_rows, projection_chunk_rows):
            local_rows = min(projection_chunk_rows, chunk_rows - local_start)
            projected = matmul_prepared_chunk_out(
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                chunk_start + local_start,
                chunk_start + local_start + local_rows,
                projection_buffer,
            )
            consume(projected, local_start)
    return query, query_scale, query_summary


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
    routing_mode: int,
    block_lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project complete query storage for the public standalone boundary."""
    return _launch_query_range(
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
        routing_mode,
        block_lengths,
    )


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
    routing_mode: int = _MINMAX_ROUTING,
    block_lengths: torch.Tensor | None = None,
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
        routing_mode,
        block_lengths,
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
    _routing_mode: int = _MINMAX_ROUTING,
    _block_lengths: torch.Tensor | None = None,
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
