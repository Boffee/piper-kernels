"""Projection-independent sparse-Piper epilogues for BF16 row chunks."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false, reportIndexIssue=false

# Triton device functions cannot carry ordinary Python type annotations.
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import triton
import triton.language as tl

from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_kernels,
)
from piper_kernels.attention.kernels.sparse_piper.layout import (
    HEAD_DIM,
    QUERY_SCALE_ROWS,
    TILE_ROWS,
)
from piper_kernels.fusions.projected_qk import triton as projected_qk

_QUERY_BLOCK_M = TILE_ROWS
_KEY_VALUE_BLOCK_M = 2 * TILE_ROWS
_HEADS_PER_PROGRAM = 2
_BLOCK_N = HEAD_DIM * _HEADS_PER_PROGRAM
_JIT_QUERY_SCALE_ROWS = tl.constexpr(QUERY_SCALE_ROWS)
_JIT_TILE_ROWS = tl.constexpr(TILE_ROWS)


@triton.jit
def _load_projection_tile(  # noqa: PLR0913, PLR0917
    projection_ptr,
    input_per_tensor_scale_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    local_sequence_offsets,
    head_offsets,
    chunk_rows,
    output_features: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
):
    projection_features = head_offsets[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
    values = tl.load(
        projection_ptr
        + local_sequence_offsets[:, None, None] * output_features
        + projection_features[None, :, :],
        mask=(local_sequence_offsets[:, None, None] < chunk_rows)
        & (head_offsets[None, :, None] < heads),
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(input_per_tensor_scale_ptr).to(tl.float32)
    if has_weight_per_tensor_scale:
        global_scale *= tl.load(weight_per_tensor_scale_ptr).to(tl.float32)
    values *= global_scale
    if has_bias:
        feature_offsets = head_offsets[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
        bias = tl.load(
            bias_ptr + feature_offsets,
            mask=head_offsets[:, None] < heads,
            other=0.0,
        ).to(tl.float32)
        values += bias[None, :, :]
    return tl.reshape(values, (block_m, heads_per_program, head_dim))


@triton.jit
def _query_epilogue_kernel(  # noqa: PLR0913, PLR0917
    projection_ptr,
    input_per_tensor_scale_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_ptr,
    query_scale_ptr,
    query_summary_ptr,
    chunk_rows,
    chunk_start,
    logical_sequence_length,
    storage_sequence_length,
    row_block_offset,
    batch,
    output_features: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    softmax_scale: tl.constexpr,
    mean_pool_summary: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    block_m: tl.constexpr,
):
    local_block = row_block_offset + tl.program_id(0)
    query_block = chunk_start // block_m + local_block
    head_block = tl.program_id(1)
    local_sequence_offsets = local_block * block_m + tl.arange(0, block_m)
    sequence_offsets = chunk_start + local_sequence_offsets
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    projection = _load_projection_tile(
        projection_ptr,
        input_per_tensor_scale_ptr,
        weight_per_tensor_scale_ptr,
        bias_ptr,
        local_sequence_offsets,
        head_offsets,
        chunk_rows,
        output_features,
        has_weight_per_tensor_scale,
        has_bias,
        heads,
        heads_per_program,
        head_dim,
        block_m,
    )
    transformed = projected_qk.rmsnorm_rope_tile(
        projection,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        sequence_offsets,
        logical_sequence_length,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        mask_ragged_tail,
        block_m,
    )
    sparse_piper_kernels.store_query_tile(
        transformed,
        query_ptr,
        query_scale_ptr,
        query_summary_ptr,
        batch,
        heads,
        head_offsets,
        sequence_offsets,
        logical_sequence_length,
        storage_sequence_length,
        query_block,
        softmax_scale,
        mean_pool_summary,
        mask_ragged_tail,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_QUERY_SCALE_ROWS,
    )


@triton.jit
def _key_epilogue_kernel(  # noqa: PLR0913, PLR0917
    projection_ptr,
    input_per_tensor_scale_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    key_ptr,
    key_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    chunk_rows,
    chunk_start,
    logical_sequence_length,
    storage_sequence_length,
    row_block_offset,
    batch,
    output_features: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    mean_pool_summary: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    block_m: tl.constexpr,
):
    local_block = row_block_offset + tl.program_id(0)
    row_block = chunk_start // block_m + local_block
    head_block = tl.program_id(1)
    local_sequence_offsets = local_block * block_m + tl.arange(0, block_m)
    sequence_offsets = chunk_start + local_sequence_offsets
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    projection = _load_projection_tile(
        projection_ptr,
        input_per_tensor_scale_ptr,
        weight_per_tensor_scale_ptr,
        bias_ptr,
        local_sequence_offsets,
        head_offsets,
        chunk_rows,
        output_features,
        has_weight_per_tensor_scale,
        has_bias,
        heads,
        heads_per_program,
        head_dim,
        block_m,
    )
    transformed = projected_qk.rmsnorm_rope_tile(
        projection,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        sequence_offsets,
        logical_sequence_length,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        mask_ragged_tail,
        block_m,
    )
    sparse_piper_kernels.store_key_tile(
        transformed,
        key_ptr,
        key_scale_ptr,
        key_summary_ptr,
        key_aux_ptr,
        batch,
        heads,
        head_offsets,
        sequence_offsets,
        logical_sequence_length,
        storage_sequence_length,
        row_block,
        mean_pool_summary,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_TILE_ROWS,
    )


@triton.jit
def _value_epilogue_kernel(  # noqa: PLR0913, PLR0917
    projection_ptr,
    input_per_tensor_scale_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    value_mean_ptr,
    value_ptr,
    value_scale_ptr,
    chunk_rows,
    chunk_start,
    logical_sequence_length,
    storage_sequence_length,
    row_block_offset,
    batch,
    output_features: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
):
    local_block = row_block_offset + tl.program_id(0)
    row_block = chunk_start // block_m + local_block
    head_block = tl.program_id(1)
    local_sequence_offsets = local_block * block_m + tl.arange(0, block_m)
    sequence_offsets = chunk_start + local_sequence_offsets
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    projection = _load_projection_tile(
        projection_ptr,
        input_per_tensor_scale_ptr,
        weight_per_tensor_scale_ptr,
        bias_ptr,
        local_sequence_offsets,
        head_offsets,
        chunk_rows,
        output_features,
        has_weight_per_tensor_scale,
        has_bias,
        heads,
        heads_per_program,
        head_dim,
        block_m,
    )
    sparse_piper_kernels.store_value_tile(
        projection,
        value_mean_ptr,
        value_ptr,
        value_scale_ptr,
        batch,
        heads,
        head_offsets,
        sequence_offsets,
        logical_sequence_length,
        storage_sequence_length,
        row_block,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_TILE_ROWS,
    )


def launch_query(  # noqa: PLR0913, PLR0917
    projection,
    input_per_tensor_scale,
    weight_per_tensor_scale,
    bias,
    norm_weight,
    cos,
    sin,
    query,
    query_scale,
    query_summary,
    chunk_start: int,
    logical_sequence_length: int,
    norm_epsilon: float,
    softmax_scale: float,
    mean_pool_summary: bool,
) -> None:
    chunk_rows, output_features = projection.shape
    heads = output_features // HEAD_DIM
    storage_sequence_length = query.shape[2]

    def launch(row_blocks: int, row_block_offset: int, *, ragged: bool) -> None:
        _query_epilogue_kernel[(row_blocks, triton.cdiv(heads, _HEADS_PER_PROGRAM))](
            projection,
            input_per_tensor_scale,
            weight_per_tensor_scale,
            bias,
            norm_weight,
            cos,
            sin,
            query,
            query_scale,
            query_summary,
            chunk_rows,
            chunk_start,
            logical_sequence_length,
            storage_sequence_length,
            row_block_offset,
            0,
            output_features=output_features,
            has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
            has_bias=bias is not None,
            heads=heads,
            heads_per_program=_HEADS_PER_PROGRAM,
            head_dim=HEAD_DIM,
            rotary_dim=cos.shape[1],
            norm_epsilon=norm_epsilon,
            softmax_scale=softmax_scale,
            mean_pool_summary=mean_pool_summary,
            mask_ragged_tail=ragged,
            block_m=_QUERY_BLOCK_M,
            num_warps=8,
        )

    full_blocks = chunk_rows // _QUERY_BLOCK_M
    if full_blocks:
        launch(full_blocks, 0, ragged=False)
    if chunk_rows % _QUERY_BLOCK_M:
        launch(1, full_blocks, ragged=True)


def launch_key(  # noqa: PLR0913, PLR0917
    projection,
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
    chunk_start: int,
    logical_sequence_length: int,
    norm_epsilon: float,
    mean_pool_summary: bool,
) -> None:
    chunk_rows, output_features = projection.shape
    heads = output_features // HEAD_DIM
    storage_sequence_length = key.shape[2]

    def launch(row_blocks: int, row_block_offset: int, *, ragged: bool) -> None:
        _key_epilogue_kernel[(row_blocks, triton.cdiv(heads, _HEADS_PER_PROGRAM))](
            projection,
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
            chunk_rows,
            chunk_start,
            logical_sequence_length,
            storage_sequence_length,
            row_block_offset,
            0,
            output_features=output_features,
            has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
            has_bias=bias is not None,
            heads=heads,
            heads_per_program=_HEADS_PER_PROGRAM,
            head_dim=HEAD_DIM,
            rotary_dim=cos.shape[1],
            norm_epsilon=norm_epsilon,
            mean_pool_summary=mean_pool_summary,
            mask_ragged_tail=ragged,
            block_m=_KEY_VALUE_BLOCK_M,
            num_warps=8,
        )

    full_blocks = chunk_rows // _KEY_VALUE_BLOCK_M
    if full_blocks:
        launch(full_blocks, 0, ragged=False)
    if chunk_rows % _KEY_VALUE_BLOCK_M:
        launch(1, full_blocks, ragged=True)


def launch_value(
    projection,
    input_per_tensor_scale,
    weight_per_tensor_scale,
    bias,
    value_mean,
    value,
    value_scale,
    chunk_start: int,
    logical_sequence_length: int,
) -> None:
    chunk_rows, output_features = projection.shape
    heads = output_features // HEAD_DIM
    storage_sequence_length = value.shape[3]

    def launch(row_blocks: int, row_block_offset: int) -> None:
        _value_epilogue_kernel[(row_blocks, triton.cdiv(heads, _HEADS_PER_PROGRAM))](
            projection,
            input_per_tensor_scale,
            weight_per_tensor_scale,
            bias,
            value_mean,
            value,
            value_scale,
            chunk_rows,
            chunk_start,
            logical_sequence_length,
            storage_sequence_length,
            row_block_offset,
            0,
            output_features=output_features,
            has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
            has_bias=bias is not None,
            heads=heads,
            heads_per_program=_HEADS_PER_PROGRAM,
            head_dim=HEAD_DIM,
            block_m=_KEY_VALUE_BLOCK_M,
            num_warps=8,
        )

    full_blocks = chunk_rows // _KEY_VALUE_BLOCK_M
    if full_blocks:
        launch(full_blocks, 0)
    if chunk_rows % _KEY_VALUE_BLOCK_M:
        launch(1, full_blocks)


__all__ = ["launch_key", "launch_query", "launch_value"]
