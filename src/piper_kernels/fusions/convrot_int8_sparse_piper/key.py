"""ConvRot INT8 projection and tile-centered sparse-Piper K64 preparation."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS
from piper_kernels.attention.sparse_piper_attention._contracts import validate_routing_mode
from piper_kernels.fusions.attention import _sparse_output as output_common
from piper_kernels.fusions.attention._validation import resolve_head_dim
from piper_kernels.fusions.convrot_int8_projection._validation import (
    validate_qk_projection_inputs,
)

from . import _backend


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
allocate_key_output = output_common.allocate_key_storage


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
    output_common.validate_optional_block_lengths(
        block_lengths, sequence_length, input_qdata.device, "K projection"
    )
    backend = _backend.require_projection_backend(input_qdata, head_dim=head_dim)
    key, key_scale, key_center, key_center_scale, key_summary, key_aux = allocate_key_output(
        input_qdata,
        batch,
        heads,
        sequence_length,
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
    heads = weight_qdata.shape[0] // head_dim
    return allocate_key_output(
        input_qdata,
        batch,
        heads,
        sequence_length,
        head_dim,
        routing_mode,
    )
