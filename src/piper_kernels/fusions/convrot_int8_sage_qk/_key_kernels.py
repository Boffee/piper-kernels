"""Project BF16 K storage, FP32 tile sums, and optional sparse routing summaries."""

# Triton device parameters are not Python runtime values.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
# pyright: reportAssignmentType=false

import triton
import triton.language as tl

from piper_kernels.attention.kernels.sparse_piper.triton import summarize_block_tiles
from piper_kernels.fusions.convrot_int8_centered_projection._kernels import store_projection_tile
from piper_kernels.fusions.convrot_int8_projection.triton import projection_tile_ids
from piper_kernels.fusions.convrot_int8_sage_qk.triton import project_rmsnorm_rope_tile


@triton.jit
def _project_key_kernel(  # noqa: PLR0913, PLR0917
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    stored_ptr,
    partial_ptr,
    summary_ptr,
    auxiliary_ptr,
    block_lengths_ptr,
    rows,
    sequence_length,
    storage_length,
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
    round_rsqrt_to_nearest: tl.constexpr,
    group_m: tl.constexpr,
    bias_ptr=None,
):
    """Keep transforms and routing in FP32; sum the represented BF16 K values."""
    tile_rows: tl.constexpr = 64
    tl.static_assert(block_m % tile_rows == 0)
    row_block, head_block = projection_tile_ids(group_m)
    row_block += row_block_offset
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    feature_offsets = tl.arange(0, head_dim)
    weight_offsets = head_block * block_n + tl.arange(0, block_n)
    transformed = project_rmsnorm_rope_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        batch * sequence_length + sequence_offsets,
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
        mask_ragged_tail,
        block_m,
        block_n,
        block_k,
        round_rsqrt_to_nearest,
        bias_ptr=bias_ptr,
    )
    valid = sequence_offsets < sequence_length
    if mask_block_lengths:
        lengths = tl.load(
            block_lengths_ptr + sequence_offsets // tile_rows,
            sequence_offsets < storage_length,
            0,
        )
        valid = valid & (sequence_offsets % tile_rows < lengths)
    transformed = tl.where(valid[:, None, None], transformed, 0.0)
    grouped = tl.reshape(
        tl.permute(transformed, (1, 0, 2)),
        (heads_per_program, block_m // tile_rows, tile_rows, head_dim),
    )
    tiles = row_block * (block_m // tile_rows) + tl.arange(0, block_m // tile_rows)
    store_projection_tile(
        grouped,
        stored_ptr,
        partial_ptr,
        batch,
        head_offsets,
        sequence_offsets,
        tiles,
        storage_length,
        heads,
        heads_per_program,
        head_dim,
        block_m,
        tile_rows,
    )
    if summary_ptr is not None:
        batch_heads = batch * heads + head_offsets.to(tl.int64)
        tile_mask = (head_offsets[:, None] < heads) & (tiles[None, :] < storage_length // tile_rows)
        tile_offsets = batch_heads[:, None] * (storage_length // tile_rows) + tiles[None, :]
        summary_offsets = tile_offsets[:, :, None] * head_dim + feature_offsets[None, None, :]
        summary, auxiliary = summarize_block_tiles(
            grouped,
            tl.reshape(valid, (block_m // tile_rows, tile_rows)),
            mean_pool_summary,
            tl.constexpr(False),
        )
        tl.store(summary_ptr + summary_offsets, summary, tile_mask[:, :, None])
        if not mean_pool_summary:
            tl.store(auxiliary_ptr + summary_offsets, auxiliary, tile_mask[:, :, None])
