"""Projection storage and represented-value statistics, independent of attention roles."""

# Triton device parameters are not Python runtime values.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false

import triton
import triton.language as tl

from piper_kernels._triton.reductions import store_mean_from_partials


@triton.jit
def store_projection_tile(
    projection,
    stored_ptr,
    partial_ptr,
    batch,
    group_offsets,
    sequence_offsets,
    tile_offsets,
    storage_length,
    groups: tl.constexpr,
    groups_per_program: tl.constexpr,
    features: tl.constexpr,
    block_m: tl.constexpr,
    tile_rows: tl.constexpr,
):
    """Store a masked [group, tile, row, feature] tile and sum its BF16 values in FP32.

    The caller supplies any transforms and zeros invalid rows before this step.
    Statistics describe the stored values; centering and encoding happen later.
    """
    tl.static_assert(block_m % tile_rows == 0)
    # Reducing before rounding would describe different values from those read
    # by a subsequent centering pass.
    stored = projection.to(tl.bfloat16)
    sums = tl.sum(stored.to(tl.float32), axis=2)
    batch_groups = batch * groups + group_offsets.to(tl.int64)
    offsets_d = tl.arange(0, features)
    partial_offsets = (
        batch_groups[:, None] * (storage_length // tile_rows) + tile_offsets[None, :]
    )[:, :, None] * features + offsets_d[None, None, :]
    tl.store(
        partial_ptr + partial_offsets,
        sums,
        (group_offsets[:, None, None] < groups)
        & (tile_offsets[None, :, None] < storage_length // tile_rows),
    )
    offsets = (
        batch_groups[:, None, None] * storage_length * features
        + sequence_offsets[None, :, None] * features
        + offsets_d[None, None, :]
    )
    tl.store(
        stored_ptr + offsets,
        tl.reshape(stored, (groups_per_program, block_m, features)),
        (group_offsets[:, None, None] < groups)
        & (sequence_offsets[None, :, None] < storage_length),
    )


@triton.jit(do_not_specialize=["sequence_length", "num_chunks"])
def _mean_finalize_kernel(
    partial_ptr,
    mean_ptr,
    sequence_length,
    num_chunks,
    features: tl.constexpr,
    block_chunks: tl.constexpr,
    block_d: tl.constexpr,
):
    store_mean_from_partials(
        partial_ptr, mean_ptr, sequence_length, num_chunks, features, block_chunks, block_d
    )
