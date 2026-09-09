"""Chunked NVFP4 key projection and sparse-Piper preparation."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import (
    TILE_ROWS,
    padded_sequence_length,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
    validate_routing_mode,
)
from piper_kernels.fusions.projected_qk._validation import resolve_head_dim
from piper_kernels.linear.nvfp4._chunking import (
    DEFAULT_CHUNK_ROWS,
    PreparedProjection,
    run_chunked_projection,
)

from . import _epilogue
from ._validation import validate_block_lengths, validate_projection, validate_qk_epilogue


def _launch_key(  # noqa: PLR0913, PLR0917
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
    chunk_rows: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_routing_mode(routing_mode)
    head_dim = resolve_head_dim(norm_weight, head_dim)
    sequence_length, heads = validate_projection(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        chunk_rows,
        "K projection",
        head_dim,
    )
    validate_qk_epilogue(
        input_qdata,
        sequence_length,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        "K projection",
        head_dim=head_dim,
    )
    validate_block_lengths(
        block_lengths,
        sequence_length,
        input_qdata.device,
        "K projection",
    )
    storage_sequence_length = padded_sequence_length(sequence_length)
    key = torch.empty(
        (1, heads, storage_sequence_length, head_dim),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    key_scale = torch.empty(
        (1, heads, storage_sequence_length // TILE_ROWS),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    summary_shape = (1, heads, storage_sequence_length // TILE_ROWS, head_dim)
    key_summary = torch.empty(summary_shape, device=input_qdata.device, dtype=torch.float32)
    mean_pool_summary = routing_mode == _MEAN_ROUTING
    key_aux = (
        torch.empty(
            (1, heads, 0, head_dim),
            device=input_qdata.device,
            dtype=torch.float32,
        )
        if mean_pool_summary
        else torch.empty_like(key_summary)
    )
    projection = PreparedProjection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
    )

    def consume(chunk: torch.Tensor, start: int) -> None:
        _epilogue.launch_key(
            chunk,
            input_per_tensor_scale,
            weight_per_tensor_scale,
            bias,
            norm_weight,
            cos,
            sin,
            key,
            key_scale,
            key_summary,
            key_aux,
            start,
            sequence_length,
            norm_epsilon,
            mean_pool_summary,
            block_lengths,
        )

    outputs = (key, key_scale, key_summary, key_aux)
    consumer_tensors = [input_per_tensor_scale, cos, sin]
    consumer_tensors.extend(
        operand
        for operand in (norm_weight, weight_per_tensor_scale, bias, block_lengths)
        if operand is not None
    )
    run_chunked_projection(projection, chunk_rows, consume, (*consumer_tensors, *outputs))
    return outputs


@torch.library.custom_op("piper_kernels::nvfp4_sparse_piper_project_key", mutates_args=())
def project_key(  # noqa: PLR0913, PLR0917
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
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    routing_mode: int = _MINMAX_ROUTING,
    block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_key(
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
        chunk_rows,
        routing_mode,
        block_lengths,
        head_dim=head_dim,
    )


@project_key.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _project_key_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _norm_epsilon: float,
    _chunk_rows: int = DEFAULT_CHUNK_ROWS,
    routing_mode: int = _MINMAX_ROUTING,
    _block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = input_qdata.shape[0]
    storage_sequence_length = padded_sequence_length(sequence_length)
    head_dim = resolve_head_dim(norm_weight, head_dim)
    heads = weight_qdata.shape[0] // head_dim
    key = input_qdata.new_empty((1, heads, storage_sequence_length, head_dim), dtype=torch.int8)
    key_scale = input_qdata.new_empty(
        (1, heads, storage_sequence_length // TILE_ROWS),
        dtype=torch.float32,
    )
    summary = input_qdata.new_empty(
        (1, heads, storage_sequence_length // TILE_ROWS, head_dim),
        dtype=torch.float32,
    )
    key_aux = (
        summary.new_empty((1, heads, 0, head_dim))
        if routing_mode == _MEAN_ROUTING
        else summary.new_empty(summary.shape)
    )
    return key, key_scale, summary, key_aux


__all__ = ["project_key"]
