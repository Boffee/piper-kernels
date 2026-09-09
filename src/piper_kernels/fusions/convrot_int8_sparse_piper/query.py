"""One-pass ConvRot INT8 projection and sparse-Piper INT8 query preparation."""

from __future__ import annotations

import math

import torch

from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    validate_routing_mode,
)
from piper_kernels.fusions.convrot_int8_sage_qk._validation import (
    validate_qk_projection_inputs,
)
from piper_kernels.fusions.projected_qk._validation import resolve_head_dim

from . import _backend
from ._interfaces import ProjectionBackend
from ._layout import (
    QUERY_SCALE_ROWS,
    TILE_ROWS,
    padded_sequence_length,
    validate_block_lengths,
)


def _validate_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
    softmax_scale: float,
    head_dim: int | None = None,
) -> tuple[int, int, int]:
    result = validate_qk_projection_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        name="Q",
        head_dim=head_dim,
    )
    if result[1] < TILE_ROWS:
        raise ValueError(f"Q projection requires at least {TILE_ROWS} sequence rows")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Q projection softmax scale must be finite and positive")
    return result


def _launch_query_projection_range(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    *,
    chunk_start: int = 0,
    chunk_rows: int | None = None,
    backend: ProjectionBackend | None = None,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a Q window, reusing the fusion's selected backend when supplied."""
    validate_routing_mode(routing_mode)
    batch, sequence_length, heads = _validate_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        softmax_scale=softmax_scale,
        head_dim=head_dim,
    )
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
    head_dim = resolve_head_dim(norm_weight, head_dim)
    storage_sequence_length = padded_sequence_length(chunk_rows)
    validate_block_lengths(block_lengths, sequence_length, input_qdata.device)
    if backend is None:
        backend = _backend.require_projection_backend(input_qdata, head_dim=head_dim)
    query = torch.empty(
        (batch, heads, storage_sequence_length, head_dim),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    query_scale = torch.empty(
        (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    query_summary = torch.empty(
        (batch, heads, storage_sequence_length // TILE_ROWS, head_dim),
        device=input_qdata.device,
        dtype=torch.float32,
    )

    backend.project_query(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        routing_mode,
        block_lengths,
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
        out=(query, query_scale, query_summary),
    )
    return query, query_scale, query_summary


def _launch_query_projection(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the complete query storage for the public standalone boundary."""
    return _launch_query_projection_range(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        routing_mode,
        block_lengths,
        head_dim=head_dim,
    )


@torch.library.custom_op("piper_kernels::convrot_int8_sparse_piper_project_query", mutates_args=())
def _project_query_op(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_query_projection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        routing_mode,
        block_lengths,
        head_dim=head_dim,
    )


@_project_query_op.register_fake
def _project_query_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    _sin: torch.Tensor,
    _norm_epsilon: float,
    _softmax_scale: float,
    _routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    head_dim = resolve_head_dim(norm_weight, head_dim)
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // head_dim
    return (
        input_qdata.new_empty((batch, heads, storage_sequence_length, head_dim)),
        input_qdata.new_empty(
            (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
            dtype=torch.float32,
        ),
        cos.new_empty((batch, heads, storage_sequence_length // TILE_ROWS, head_dim)),
    )
