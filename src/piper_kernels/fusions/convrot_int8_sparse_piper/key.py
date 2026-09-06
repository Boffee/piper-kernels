"""One-pass ConvRot INT8 projection and sparse-Piper INT8 key preparation."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

# Triton device functions cannot carry ordinary Python type annotations.
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_kernels,
)
from piper_kernels.attention.sparse_piper_attention._routes import (
    _MEAN_ROUTING,
    validate_routing_mode,
)
from piper_kernels.fusions.convrot_int8_sage_qk.triton import (
    project_rmsnorm_rope_tile,
    validate_qk_projection_inputs,
)

from ._layout import HEAD_DIM, TILE_ROWS, padded_sequence_length, validate_block_lengths

_BLOCK_M = 128
_BLOCK_K = 128
_HEADS_PER_PROGRAM = 2
_BLOCK_N = HEAD_DIM * _HEADS_PER_PROGRAM
_JIT_KEY_TILE_ROWS = tl.constexpr(TILE_ROWS)


@triton.jit
def _convrot_project_quantize_key_kernel(  # noqa: PLR0913, PLR0917
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    key_ptr,
    key_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    rows,
    logical_sequence_length,
    storage_sequence_length,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    mean_pool_summary: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    aligned_projection: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Project K once and emit INT8 operands plus route summaries."""
    row_block = row_block_offset + tl.program_id(0)
    head_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    row_offsets = batch * logical_sequence_length + sequence_offsets
    projection_feature_offsets = tl.arange(0, block_n)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + projection_feature_offsets
    key = project_rmsnorm_rope_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        row_offsets,
        weight_offsets,
        sequence_offsets,
        rows,
        logical_sequence_length,
        input_features,
        heads * head_dim,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        aligned_projection,
        mask_ragged_tail,
        block_m,
        block_n,
        block_k,
    )
    sparse_piper_kernels.store_key_tile(
        key,
        key_ptr,
        key_scale_ptr,
        key_summary_ptr,
        key_aux_ptr,
        block_lengths_ptr,
        batch,
        heads,
        head_offsets,
        sequence_offsets,
        logical_sequence_length,
        storage_sequence_length,
        row_block,
        mean_pool_summary,
        mask_block_lengths,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_KEY_TILE_ROWS,
    )


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
) -> tuple[int, int, int, int]:
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


def _launch_projection(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    *,
    batch: int,
    logical_sequence_length: int,
    storage_sequence_length: int,
    heads: int,
    rotary_dim: int,
    norm_epsilon: float,
    mean_pool_summary: bool,
    block_lengths: torch.Tensor | None,
) -> None:
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else key_scale

    with device_context(input_qdata.device):

        def launch(row_block_count: int, row_block_offset: int, *, aligned_rows: bool) -> None:
            _convrot_project_quantize_key_kernel[
                (
                    row_block_count,
                    triton.cdiv(heads, _HEADS_PER_PROGRAM),
                    batch,
                )
            ](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                norm_weight,
                cos,
                sin,
                key,
                key_scale,
                key_summary,
                key_aux,
                block_lengths_ptr,
                batch * logical_sequence_length,
                logical_sequence_length,
                storage_sequence_length,
                row_block_offset,
                input_features=input_qdata.shape[2],
                heads=heads,
                heads_per_program=_HEADS_PER_PROGRAM,
                head_dim=HEAD_DIM,
                rotary_dim=rotary_dim,
                norm_epsilon=norm_epsilon,
                mean_pool_summary=mean_pool_summary,
                mask_block_lengths=has_block_lengths,
                aligned_projection=(
                    aligned_rows
                    and input_qdata.shape[2] % _BLOCK_K == 0
                    and heads % _HEADS_PER_PROGRAM == 0
                ),
                mask_ragged_tail=not aligned_rows,
                block_m=_BLOCK_M,
                block_n=_BLOCK_N,
                block_k=_BLOCK_K,
                num_warps=8,
                num_stages=3,
            )

        full_row_blocks = logical_sequence_length // _BLOCK_M
        if full_row_blocks:
            launch(full_row_blocks, 0, aligned_rows=True)
        if logical_sequence_length % _BLOCK_M:
            launch(1, full_row_blocks, aligned_rows=False)


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
    batch, sequence_length, heads, rotary_dim = _validate_inputs(
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
    _launch_projection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        key,
        key_scale,
        key_summary,
        key_aux,
        batch=batch,
        logical_sequence_length=sequence_length,
        storage_sequence_length=storage_sequence_length,
        heads=heads,
        rotary_dim=rotary_dim,
        norm_epsilon=norm_epsilon,
        mean_pool_summary=mean_pool_summary,
        block_lengths=block_lengths,
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
