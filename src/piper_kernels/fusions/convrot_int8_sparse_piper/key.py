"""ConvRot INT8 projection and tile-centered sparse-Piper K64 preparation."""

from __future__ import annotations

import torch

from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    validate_routing_mode,
)
from piper_kernels.fusions.attention._validation import resolve_head_dim
from piper_kernels.fusions.convrot_int8_projection._validation import (
    validate_qk_projection_inputs,
)

from . import _backend
from ._layout import TILE_ROWS, padded_sequence_length, validate_block_lengths


def validate_key_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
    head_dim: int | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    """Validate fused ConvRot INT8 K projection tensors and return ``(batch, seq, heads)``."""
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
        head_dim=head_dim,
        bias=bias,
    )
    if result[1] < TILE_ROWS:
        raise ValueError(f"K projection requires at least {TILE_ROWS} sequence rows")
    return result


_validate_inputs = validate_key_inputs


def allocate_key_output(
    template: torch.Tensor,
    batch: int,
    heads: int,
    storage_sequence_length: int,
    head_dim: int,
    routing_mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate contiguous sparse-Piper key storage and per-tile metadata."""
    storage_tiles = storage_sequence_length // TILE_ROWS
    key = template.new_empty(
        (batch, heads, storage_sequence_length, head_dim),
        dtype=torch.int8,
    )
    key_scale = template.new_empty((batch, heads, storage_tiles), dtype=torch.float32)
    key_center = template.new_empty(
        (batch, heads, storage_tiles, 2, head_dim),
        dtype=torch.int8,
    )
    key_center_scale = template.new_empty((batch, heads, storage_tiles, 2), dtype=torch.float32)
    summary_shape = (batch, heads, storage_tiles, head_dim)
    key_summary = template.new_empty(summary_shape, dtype=torch.float32)
    key_aux = (
        template.new_empty((batch, heads, 0, head_dim), dtype=torch.float32)
        if routing_mode == _MEAN_ROUTING
        else template.new_empty(summary_shape, dtype=torch.float32)
    )
    return key, key_scale, key_center, key_center_scale, key_summary, key_aux


def _launch_key_projection(  # noqa: PLR0913
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        head_dim=head_dim,
        bias=bias,
    )
    head_dim = resolve_head_dim(norm_weight, head_dim)
    storage_sequence_length = padded_sequence_length(sequence_length)
    validate_block_lengths(block_lengths, sequence_length, input_qdata.device)
    backend = _backend.require_projection_backend(input_qdata, head_dim=head_dim)
    key, key_scale, key_center, key_center_scale, key_summary, key_aux = allocate_key_output(
        input_qdata,
        batch,
        heads,
        storage_sequence_length,
        head_dim,
        routing_mode,
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
        out=(key, key_scale, key_center, key_center_scale, key_summary, key_aux),
        bias=bias,
    )
    return key, key_scale, key_center, key_center_scale, key_summary, key_aux


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_project_key",
    mutates_args=(),
)
def _project_key_op(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        head_dim=head_dim,
        bias=bias,
    )


@_project_key_op.register_fake
def _project_key_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _norm_epsilon: float,
    routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
    _bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    head_dim = resolve_head_dim(norm_weight, head_dim)
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // head_dim
    return allocate_key_output(
        input_qdata,
        batch,
        heads,
        storage_sequence_length,
        head_dim,
        routing_mode,
    )
