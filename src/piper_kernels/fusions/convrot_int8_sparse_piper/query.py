"""One-pass ConvRot INT8 projection and sparse-Piper INT8 query preparation."""

from __future__ import annotations

import torch

from piper_kernels._validation import validate_positive_finite_float
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS
from piper_kernels.attention.sparse_piper_attention._contracts import validate_routing_mode
from piper_kernels.fusions.attention import _sparse_output as output_common
from piper_kernels.fusions.attention._validation import resolve_head_dim
from piper_kernels.fusions.convrot_int8_projection._validation import (
    validate_qk_projection_inputs,
)

from . import _backend
from ._interfaces import ProjectionBackend


def _validate_inputs(  # noqa: PLR0913
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
    bias: torch.Tensor | None = None,
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
        bias=bias,
    )
    validate_positive_finite_float(
        softmax_scale,
        "Q projection softmax scale must be finite and positive",
    )
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
    bias: torch.Tensor | None = None,
    validated_shape: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a Q window from standalone operands or a validated attention context."""
    if validated_shape is None:
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
            bias=bias,
        )
        head_dim = resolve_head_dim(norm_weight, head_dim)
        output_common.validate_optional_block_lengths(
            block_lengths, sequence_length, input_qdata.device, "Q projection"
        )
    else:
        batch, sequence_length, heads, head_dim = validated_shape
    if chunk_rows is None:
        if sequence_length < TILE_ROWS:
            raise ValueError(f"Q projection requires at least {TILE_ROWS} sequence rows")
        chunk_rows = sequence_length
    output_common.validate_query_range(chunk_start, chunk_rows, sequence_length)
    if backend is None:
        backend = _backend.require_projection_backend(input_qdata, head_dim=head_dim)
    out = output_common.allocate_query_storage(input_qdata, batch, heads, chunk_rows, head_dim)

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
        out=out,
        bias=bias,
    )
    return out


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
    bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        bias=bias,
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
    _bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    head_dim = resolve_head_dim(norm_weight, head_dim)
    heads = weight_qdata.shape[0] // head_dim
    return output_common.allocate_query_storage(
        input_qdata,
        batch,
        heads,
        sequence_length,
        head_dim,
        summary_template=cos,
    )
