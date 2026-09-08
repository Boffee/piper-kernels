"""One-pass ConvRot INT8 projection and sparse-Piper INT8 key preparation."""

from __future__ import annotations

import torch

from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    validate_routing_mode,
)
from piper_kernels.fusions.convrot_int8_sage_qk._validation import (
    validate_qk_projection_inputs,
)

from . import _backend
from ._layout import HEAD_DIM, TILE_ROWS, padded_sequence_length, validate_block_lengths


def _validate_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
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
        name="K",
    )
    if result[1] < TILE_ROWS:
        raise ValueError(f"K projection requires at least {TILE_ROWS} sequence rows")
    return result


def _launch_key_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    )
    storage_sequence_length = padded_sequence_length(sequence_length)
    validate_block_lengths(block_lengths, sequence_length, input_qdata.device)
    backend = _backend.require_projection_backend(input_qdata)
    key = torch.empty(
        (batch, heads, storage_sequence_length, HEAD_DIM),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    key_scale = torch.empty(
        (batch, heads, storage_sequence_length // TILE_ROWS),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    summary_shape = (batch, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM)
    key_summary = torch.empty(summary_shape, device=input_qdata.device, dtype=torch.float32)
    mean_pool_summary = routing_mode == _MEAN_ROUTING
    key_aux = (
        torch.empty(
            (batch, heads, 0, HEAD_DIM),
            device=input_qdata.device,
            dtype=torch.float32,
        )
        if mean_pool_summary
        else torch.empty_like(key_summary)
    )
    backend.project_key(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        routing_mode,
        block_lengths,
        out=(key, key_scale, key_summary, key_aux),
    )
    return key, key_scale, key_summary, key_aux


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_project_key",
    mutates_args=(),
)
def _project_key_op(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_key_projection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        routing_mode,
        block_lengths,
    )


@_project_key_op.register_fake
def _project_key_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _norm_weight: torch.Tensor,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _norm_epsilon: float,
    routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // HEAD_DIM
    key = input_qdata.new_empty((batch, heads, storage_sequence_length, HEAD_DIM))
    key_scale = input_qdata.new_empty(
        (batch, heads, storage_sequence_length // TILE_ROWS),
        dtype=torch.float32,
    )
    summary = input_qdata.new_empty(
        (batch, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM),
        dtype=torch.float32,
    )
    key_aux = (
        summary.new_empty((batch, heads, 0, HEAD_DIM))
        if routing_mode == _MEAN_ROUTING
        else summary.new_empty(summary.shape)
    )
    return key, key_scale, summary, key_aux
