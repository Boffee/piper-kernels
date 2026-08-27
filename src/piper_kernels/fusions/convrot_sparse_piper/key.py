"""One-pass ConvRot projection and sparse-Piper INT8 key preparation."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

# Triton device functions cannot carry ordinary Python type annotations.
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels.fusions.convrot_sage_qk.triton import (
    project_rmsnorm_rope_tile,
    quantize_key_tile,
    validate_qk_projection_inputs,
)

_BLOCK_M = 128
_KEY_TILE_ROWS = 64
_BLOCK_K = 128
_HEAD_DIM = 128
_HEADS_PER_PROGRAM = 2
_BLOCK_N = _HEAD_DIM * _HEADS_PER_PROGRAM
_JIT_KEY_TILE_ROWS = tl.constexpr(64)


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
    key_max_ptr,
    key_min_ptr,
    rows,
    sequence_length,
    valid_sequence_length,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    aligned_projection: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Project K once and emit INT8 operands plus exact DSA extrema."""
    row_block = row_block_offset + tl.program_id(0)
    head_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    row_offsets = batch * sequence_length + sequence_offsets
    projection_feature_offsets = tl.arange(0, block_n)
    feature_offsets = tl.arange(0, head_dim)
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
        sequence_length,
        input_features,
        heads * head_dim,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        aligned_projection,
        block_m,
        block_n,
        block_k,
    ).to(tl.float32)
    valid_rows = sequence_offsets < valid_sequence_length
    key = tl.where(valid_rows[:, None, None], key, 0.0)
    key_head_major = tl.permute(key, (1, 0, 2))
    grouped = tl.reshape(
        key_head_major,
        (
            heads_per_program,
            block_m // _JIT_KEY_TILE_ROWS,
            _JIT_KEY_TILE_ROWS,
            head_dim,
        ),
    )
    local_tile_offsets = tl.arange(0, block_m // _JIT_KEY_TILE_ROWS)
    tile_offsets = row_block * (block_m // _JIT_KEY_TILE_ROWS) + local_tile_offsets
    row_in_tile = tl.arange(0, _JIT_KEY_TILE_ROWS)
    valid = (
        tile_offsets[:, None] * _JIT_KEY_TILE_ROWS + row_in_tile[None, :] < valid_sequence_length
    )
    key_max = tl.max(
        tl.where(valid[None, :, :, None], grouped, -float("inf")),
        axis=2,
    )
    key_min = tl.min(
        tl.where(valid[None, :, :, None], grouped, float("inf")),
        axis=2,
    )

    quantized, key_scale = quantize_key_tile(
        key,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_KEY_TILE_ROWS,
    )
    key_offsets = (
        (batch * heads + head_offsets[:, None, None]) * sequence_length * head_dim
        + sequence_offsets[None, :, None] * head_dim
        + feature_offsets[None, None, :]
    )
    tl.store(
        key_ptr + key_offsets,
        quantized,
        mask=(head_offsets[:, None, None] < heads)
        & (sequence_offsets[None, :, None] < sequence_length),
    )
    tile_count = sequence_length // _JIT_KEY_TILE_ROWS
    scale_offsets = (batch * heads + head_offsets[:, None]) * tile_count + tile_offsets[None, :]
    tile_mask = (head_offsets[:, None] < heads) & (tile_offsets[None, :] < tile_count)
    tl.store(key_scale_ptr + scale_offsets, key_scale, mask=tile_mask)
    summary_offsets = scale_offsets[:, :, None] * head_dim + feature_offsets[None, None, :]
    tl.store(key_max_ptr + summary_offsets, key_max, mask=tile_mask[:, :, None])
    tl.store(key_min_ptr + summary_offsets, key_min, mask=tile_mask[:, :, None])


def _validate_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    valid_sequence_length: int,
    norm_epsilon: float,
) -> tuple[int, int, int, int]:
    return validate_qk_projection_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        valid_sequence_length=valid_sequence_length,
        norm_epsilon=norm_epsilon,
        block_rows=_KEY_TILE_ROWS,
        name="K",
    )


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
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    *,
    batch: int,
    sequence_length: int,
    valid_sequence_length: int,
    heads: int,
    rotary_dim: int,
    norm_epsilon: float,
) -> None:
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
            key_max,
            key_min,
            batch * sequence_length,
            sequence_length,
            valid_sequence_length,
            row_block_offset,
            input_features=input_qdata.shape[2],
            heads=heads,
            heads_per_program=_HEADS_PER_PROGRAM,
            head_dim=_HEAD_DIM,
            rotary_dim=rotary_dim,
            norm_epsilon=norm_epsilon,
            aligned_projection=(
                aligned_rows
                and input_qdata.shape[2] % _BLOCK_K == 0
                and heads % _HEADS_PER_PROGRAM == 0
            ),
            block_m=_BLOCK_M,
            block_n=_BLOCK_N,
            block_k=_BLOCK_K,
            num_warps=8,
            num_stages=3,
        )

    full_row_blocks = sequence_length // _BLOCK_M
    if full_row_blocks:
        launch(full_row_blocks, 0, aligned_rows=True)
    if sequence_length % _BLOCK_M:
        launch(1, full_row_blocks, aligned_rows=False)


def _launch_key_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    valid_sequence_length: int,
    norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, heads, rotary_dim = _validate_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        valid_sequence_length=valid_sequence_length,
        norm_epsilon=norm_epsilon,
    )
    key = torch.empty(
        (batch, heads, sequence_length, _HEAD_DIM),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    key_scale = torch.empty(
        (batch, heads, sequence_length // _KEY_TILE_ROWS),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    summary_shape = (batch, heads, sequence_length // _KEY_TILE_ROWS, _HEAD_DIM)
    key_max = torch.empty(summary_shape, device=input_qdata.device, dtype=torch.float32)
    key_min = torch.empty_like(key_max)
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
        key_max,
        key_min,
        batch=batch,
        sequence_length=sequence_length,
        valid_sequence_length=valid_sequence_length,
        heads=heads,
        rotary_dim=rotary_dim,
        norm_epsilon=norm_epsilon,
    )
    return key, key_scale, key_max, key_min


@torch.library.custom_op(
    "piper_kernels::convrot_sparse_piper_project_key",
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
    valid_sequence_length: int,
    norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_key_projection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        valid_sequence_length,
        norm_epsilon,
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
    _valid_sequence_length: int,
    _norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    heads = weight_qdata.shape[0] // _HEAD_DIM
    key = input_qdata.new_empty((batch, heads, sequence_length, _HEAD_DIM))
    key_scale = input_qdata.new_empty(
        (batch, heads, sequence_length // _KEY_TILE_ROWS),
        dtype=torch.float32,
    )
    summary = input_qdata.new_empty(
        (batch, heads, sequence_length // _KEY_TILE_ROWS, _HEAD_DIM),
        dtype=torch.float32,
    )
    return key, key_scale, summary, summary.new_empty(summary.shape)
