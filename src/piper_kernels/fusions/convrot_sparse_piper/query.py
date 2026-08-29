"""One-pass ConvRot projection and sparse-Piper INT8 query preparation."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

# Triton device functions cannot carry ordinary Python type annotations.
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from piper_kernels.fusions.convrot_sage_qk.triton import (
    project_rmsnorm_rope_tile,
    quantize_query_tile,
    validate_qk_projection_inputs,
)

from ._layout import HEAD_DIM, QUERY_SCALE_ROWS, TILE_ROWS, padded_sequence_length

_BLOCK_M = TILE_ROWS
_BLOCK_K = 128
_HEADS_PER_PROGRAM = 2
_BLOCK_N = HEAD_DIM * _HEADS_PER_PROGRAM
_JIT_QUERY_SCALE_ROWS = tl.constexpr(QUERY_SCALE_ROWS)


@triton.jit
def _convrot_project_rmsnorm_rope_quantize_query_kernel(  # noqa: PLR0913, PLR0917
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_ptr,
    query_scale_ptr,
    query_summary_ptr,
    rows,
    logical_sequence_length,
    storage_sequence_length,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    softmax_scale: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    aligned_projection: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Project one Q64/two-head tile and emit Q32 INT8 plus exact Q64 summaries."""
    tl.static_assert(block_m == 64)
    tl.static_assert(heads_per_program == 2)
    tl.static_assert(block_n == heads_per_program * head_dim)
    tl.static_assert(head_dim == 128)
    tl.static_assert(rotary_dim <= head_dim)
    tl.static_assert(rotary_dim % 2 == 0)

    query_block = tl.program_id(0)
    if mask_ragged_tail:
        query_block = logical_sequence_length // block_m
    head_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = query_block * block_m + tl.arange(0, block_m)
    row_offsets = batch * logical_sequence_length + sequence_offsets
    feature_offsets = tl.arange(0, head_dim)
    projection_feature_offsets = tl.arange(0, block_n)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + projection_feature_offsets
    rope = project_rmsnorm_rope_tile(
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
        block_m,
        block_n,
        block_k,
    )

    rope_fp32 = rope.to(tl.float32)
    if mask_ragged_tail:
        valid_rows = sequence_offsets < logical_sequence_length
        rope_fp32 = tl.where(valid_rows[:, None, None], rope_fp32, 0.0)
        query_summary = tl.max(
            tl.where(valid_rows[:, None, None], rope_fp32, -float("inf")),
            axis=0,
        ) + tl.min(
            tl.where(valid_rows[:, None, None], rope_fp32, float("inf")),
            axis=0,
        )
    else:
        query_summary = tl.max(rope_fp32, axis=0) + tl.min(rope_fp32, axis=0)
    summary_offsets = (
        (batch * heads + head_offsets[:, None]) * (storage_sequence_length // block_m) + query_block
    ) * head_dim + feature_offsets[None, :]
    tl.store(
        query_summary_ptr + summary_offsets,
        query_summary,
        mask=head_offsets[:, None] < heads,
    )

    group_offsets = tl.arange(0, block_m // _JIT_QUERY_SCALE_ROWS)
    group_valid = head_offsets[:, None] < heads
    if mask_ragged_tail:
        group_starts = query_block * block_m + group_offsets * _JIT_QUERY_SCALE_ROWS
        group_valid = group_valid & (group_starts[None, :] < logical_sequence_length)
    quantized, stored_scale = quantize_query_tile(
        rope_fp32,
        group_valid,
        softmax_scale,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_QUERY_SCALE_ROWS,
    )
    query_offsets = (
        batch * heads * storage_sequence_length * head_dim
        + head_offsets[:, None, None] * storage_sequence_length * head_dim
        + sequence_offsets[None, :, None] * head_dim
        + feature_offsets[None, None, :]
    )
    tl.store(
        query_ptr + query_offsets,
        quantized,
        mask=head_offsets[:, None, None] < heads,
    )
    scale_offsets = (
        (batch * heads + head_offsets[:, None]) * (storage_sequence_length // _JIT_QUERY_SCALE_ROWS)
        + query_block * (block_m // _JIT_QUERY_SCALE_ROWS)
        + group_offsets[None, :]
    )
    tl.store(
        query_scale_ptr + scale_offsets,
        stored_scale,
        mask=head_offsets[:, None] < heads,
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
    softmax_scale: float,
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
        name="Q",
    )
    if result[1] < TILE_ROWS:
        raise ValueError(f"Q projection requires at least {TILE_ROWS} sequence rows")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Q projection softmax scale must be finite and positive")
    return result


def _launch_query_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, heads, rotary_dim = _validate_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        softmax_scale=softmax_scale,
    )
    storage_sequence_length = padded_sequence_length(sequence_length)
    query = torch.empty(
        (batch, heads, storage_sequence_length, HEAD_DIM),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    query_scale = torch.empty(
        (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    query_summary = torch.empty(
        (batch, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM),
        device=input_qdata.device,
        dtype=torch.float32,
    )

    def launch(row_block_count: int, *, mask_ragged_tail: bool) -> None:
        _convrot_project_rmsnorm_rope_quantize_query_kernel[
            (row_block_count, triton.cdiv(heads, _HEADS_PER_PROGRAM), batch)
        ](
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            norm_weight,
            cos,
            sin,
            query,
            query_scale,
            query_summary,
            batch * sequence_length,
            sequence_length,
            storage_sequence_length,
            input_features=input_qdata.shape[2],
            heads=heads,
            heads_per_program=_HEADS_PER_PROGRAM,
            head_dim=HEAD_DIM,
            rotary_dim=rotary_dim,
            norm_epsilon=norm_epsilon,
            softmax_scale=softmax_scale,
            mask_ragged_tail=mask_ragged_tail,
            aligned_projection=(
                not mask_ragged_tail
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
        launch(full_row_blocks, mask_ragged_tail=False)
    if sequence_length % _BLOCK_M:
        launch(1, mask_ragged_tail=True)
    return query, query_scale, query_summary


@torch.library.custom_op("piper_kernels::convrot_sparse_piper_project_query", mutates_args=())
def _project_query_op(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
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
    )


@_project_query_op.register_fake
def _project_query_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _norm_weight: torch.Tensor,
    cos: torch.Tensor,
    _sin: torch.Tensor,
    _norm_epsilon: float,
    _softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // HEAD_DIM
    return (
        input_qdata.new_empty((batch, heads, storage_sequence_length, HEAD_DIM)),
        input_qdata.new_empty(
            (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
            dtype=torch.float32,
        ),
        cos.new_empty((batch, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM)),
    )
