"""Shared fused FP32 projection, transformation, and sparse operand stores."""

# Triton device parameters are not Python runtime values.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false

import triton
import triton.language as tl

from piper_kernels.attention.kernels.sparse_piper import triton as sparse_piper_kernels
from piper_kernels.fusions.convrot_int8_projection.triton import (
    project_prepared_input_mean_kernel as _project_prepared_input_mean_kernel,  # noqa: F401 - shared launcher export
)
from piper_kernels.fusions.convrot_int8_projection.triton import project_tile
from piper_kernels.fusions.convrot_int8_projection.triton import (
    projection_tile_ids as _projection_tile_ids,
)
from piper_kernels.fusions.convrot_int8_sage_qk.triton import project_rmsnorm_rope_tile

from ._layout import QUERY_SCALE_ROWS, TILE_ROWS

_JIT_QUERY_SCALE_ROWS = tl.constexpr(QUERY_SCALE_ROWS)
_JIT_VALUE_TILE_ROWS = tl.constexpr(TILE_ROWS)


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
    block_lengths_ptr,
    rows,
    chunk_start,
    chunk_rows,
    logical_sequence_length,
    query_sequence_end,
    storage_sequence_length,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    softmax_scale: tl.constexpr,
    mean_pool_summary: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    aligned_projection: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    round_rsqrt_to_nearest: tl.constexpr = False,
    group_m: tl.constexpr = 0,
    bias_ptr=None,
):
    """Project a Q64 tile and emit Q32 INT8 plus route summaries."""
    tl.static_assert(block_m == 64)
    tl.static_assert(heads_per_program == 1 or heads_per_program == 2)  # noqa: PLR1714
    tl.static_assert(block_n == heads_per_program * head_dim)
    tl.static_assert(head_dim == 64 or head_dim == 128)  # noqa: PLR1714
    tl.static_assert(rotary_dim <= head_dim)
    tl.static_assert(rotary_dim % 2 == 0)

    storage_query_block, head_block = _projection_tile_ids(group_m)
    if mask_ragged_tail:
        storage_query_block = chunk_rows // block_m
    global_query_block = chunk_start // block_m + storage_query_block
    batch = tl.program_id(2)
    storage_sequence_offsets = storage_query_block * block_m + tl.arange(0, block_m)
    global_sequence_offsets = chunk_start + storage_sequence_offsets
    row_offsets = batch * logical_sequence_length + global_sequence_offsets
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
        global_sequence_offsets,
        rows,
        query_sequence_end,
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
        round_rsqrt_to_nearest,
        bias_ptr=bias_ptr,
    )

    sparse_piper_kernels.store_query_tile(
        rope,
        query_ptr,
        query_scale_ptr,
        query_summary_ptr,
        block_lengths_ptr,
        batch,
        heads,
        head_offsets,
        global_sequence_offsets,
        storage_sequence_offsets,
        query_sequence_end,
        storage_sequence_length,
        global_query_block,
        storage_query_block,
        softmax_scale,
        mean_pool_summary,
        mask_block_lengths,
        mask_ragged_tail,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_QUERY_SCALE_ROWS,
    )


@triton.jit
def _convrot_project_quantize_sparse_value_kernel(  # noqa: PLR0913, PLR0917
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    value_mean_ptr,
    value_ptr,
    value_scale_ptr,
    block_mean_ptr,
    block_lengths_ptr,
    rows,
    logical_sequence_length,
    storage_sequence_length,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    aligned_projection: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    emit_block_mean: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr = 0,
    bias_ptr=None,
):
    """Project two heads over two K64 tiles and emit sparse Piper's V format."""
    tl.static_assert(block_m == 2 * _JIT_VALUE_TILE_ROWS)
    tl.static_assert(heads_per_program == 2)
    tl.static_assert(block_n == heads_per_program * head_dim)
    tl.static_assert(head_dim == 64 or head_dim == 128)  # noqa: PLR1714

    row_block, head_block = _projection_tile_ids(group_m)
    row_block += row_block_offset
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    row_offsets = batch * logical_sequence_length + sequence_offsets
    projection_feature_offsets = tl.arange(0, block_n)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + projection_feature_offsets
    projection = project_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        row_offsets,
        weight_offsets,
        rows,
        input_features,
        heads * head_dim,
        aligned_projection,
        block_m,
        block_n,
        block_k,
        bias_ptr=bias_ptr,
    )
    projection = tl.reshape(projection, (block_m, heads_per_program, head_dim))
    sparse_piper_kernels.store_value_tile(
        projection,
        value_mean_ptr,
        value_ptr,
        value_scale_ptr,
        block_mean_ptr,
        block_lengths_ptr,
        batch,
        heads,
        head_offsets,
        sequence_offsets,
        logical_sequence_length,
        storage_sequence_length,
        row_block,
        mask_block_lengths,
        emit_block_mean,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_VALUE_TILE_ROWS,
    )
