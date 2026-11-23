"""Chunked NVFP4 query projection and sparse-Piper preparation."""

from __future__ import annotations

import torch

from piper_kernels._validation import validate_positive_finite_float
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
    validate_routing_mode,
)
from piper_kernels.fusions.attention import _sparse_output as output_common
from piper_kernels.fusions.attention._validation import resolve_head_dim
from piper_kernels.linear.nvfp4._chunking import (
    DEFAULT_CHUNK_ROWS,
    PreparedProjection,
    run_chunked_projection,
)
from piper_kernels.linear.nvfp4._projection import matmul_prepared_chunk_out

from . import _epilogue
from ._validation import validate_block_lengths, validate_projection, validate_qk_epilogue


def _validate_query_inputs(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    projection_chunk_rows: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    head_dim: int | None = None,
) -> tuple[int, int, int]:
    """Validate Q projection and RoPE/summary metadata once at the owning boundary."""
    validate_routing_mode(routing_mode)
    resolved_head_dim = resolve_head_dim(norm_weight, head_dim)
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
        resolved_head_dim,
    )
    validate_qk_epilogue(
        input_qdata,
        sequence_length,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        "Q projection",
        head_dim=resolved_head_dim,
    )
    validate_block_lengths(
        block_lengths,
        sequence_length,
        input_qdata.device,
        "Q projection",
    )
    validate_positive_finite_float(
        softmax_scale,
        "Q projection softmax scale must be finite and positive",
    )
    return sequence_length, heads, resolved_head_dim


def _launch_query_range(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
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
    head_dim: int | None = None,
    validated: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length, heads, head_dim = (
        _validate_query_inputs(
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
            projection_chunk_rows,
            routing_mode,
            block_lengths,
            head_dim=head_dim,
        )
        if validated is None
        else validated
    )
    if chunk_rows is None:
        chunk_rows = sequence_length
    output_common.validate_query_range(chunk_start, chunk_rows, sequence_length)
    query, query_scale, query_summary = output_common.allocate_query_storage(
        input_qdata, 1, heads, chunk_rows, head_dim
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

    consumer_tensors = [input_per_tensor_scale, cos, sin]
    consumer_tensors.extend(
        operand
        for operand in (norm_weight, weight_per_tensor_scale, bias, block_lengths)
        if operand is not None
    )
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


@torch.library.custom_op("piper_kernels::nvfp4_sparse_piper_project_query", mutates_args=())
def project_query(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    routing_mode: int = _MINMAX_ROUTING,
    block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        head_dim=head_dim,
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
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    _sin: torch.Tensor,
    _norm_epsilon: float,
    _softmax_scale: float,
    _chunk_rows: int = DEFAULT_CHUNK_ROWS,
    _routing_mode: int = _MINMAX_ROUTING,
    _block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = input_qdata.shape[0]
    head_dim = resolve_head_dim(norm_weight, head_dim)
    heads = weight_qdata.shape[0] // head_dim
    return output_common.allocate_query_storage(
        input_qdata,
        1,
        heads,
        sequence_length,
        head_dim,
        summary_template=cos,
    )


__all__ = ["DEFAULT_CHUNK_ROWS", "project_query"]
