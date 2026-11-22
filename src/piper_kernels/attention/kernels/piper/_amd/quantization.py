"""Shared vectorized 8-warp RDNA4 Gluon quantization primitives and Q preparation."""

# Gluon device parameters, layouts, and static tuples are not Python runtime values.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.amd_compiler_workarounds import LDS_ONLY_FENCE_ATTRS
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels._grid import select_block_grid
from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import (
    SIGNED_HADAMARD_MASK,
)

TILE_ROWS = 64
QUERY_SCALE_ROWS = 32

GL_TILE_ROWS = gl.constexpr(TILE_ROWS)
GL_TILE_ROWS_F32 = gl.constexpr(float(TILE_ROWS))
GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)
_GL_HADAMARD_W0 = gl.constexpr(SIGNED_HADAMARD_MASK[0])
_GL_HADAMARD_W1 = gl.constexpr(SIGNED_HADAMARD_MASK[1])
_GL_HADAMARD_W2 = gl.constexpr(SIGNED_HADAMARD_MASK[2])
_GL_HADAMARD_W3 = gl.constexpr(SIGNED_HADAMARD_MASK[3])
GL_LOG2_E = gl.constexpr(1.4426950408889634)
GL_INT8_RANGE = gl.constexpr(127.0)
GL_SCALE_EPSILON = gl.constexpr(1e-7)
GL_VALUE_MEAN_SCALE_FLOOR = gl.constexpr(2.0**-80)
GL_P_UINT8_RANGE = gl.constexpr(255.0)
GL_D128_NORM = gl.constexpr(0.08838834764831845)
GL_D64_NORM = gl.constexpr(0.125)

VALUE_STORE_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0]))

# Coalesced [64, 128] tile layout (8 wave32 warps, 8 contiguous columns per register block -> b128).
VEC_TILE_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [4, 0]],
        warp_bases=[[8, 0], [16, 0], [32, 0]],
        block_bases=[],
        shape=[64, 128],
    )
)
MEAN_BCAST_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [4, 0]],
        warp_bases=[[0, 0], [0, 0], [0, 0]],
        block_bases=[],
        shape=[8, 128],
    )
)
MINMAX_REDUCE_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [1, 0, 0]],
        lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 0]],
        warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]],
        block_bases=[],
        shape=[2, 8, 128],
    )
)

PREP_LLVM_FN_ATTRS = LDS_ONLY_FENCE_ATTRS
PREP_COMMON_DO_NOT_SPECIALIZE = (
    "logical_sequence_length",
    "storage_tiles",
    "stride_b",
    "stride_h",
    "stride_n",
    "heads",
    "heads_inner",
)

_SWIZZLE_SWAP_ASM = gl.constexpr(
    tuple(
        "\n\t".join(
            [f"ds_swizzle_b32 ${i}, ${i + 8} offset:swizzle(SWAP,{mask})" for i in range(8)]
            # LLVM SIInsertWaitcnts does not track DS_CNT across INLINEASM.
            + ["s_wait_dscnt 0x0"]
        )
        for mask in (1, 2, 4, 8)
    )
)
_SWIZZLE_SWAP_CONSTRAINTS = gl.constexpr(",".join(["=&v"] * 8 + ["v"] * 8))


def is_aligned_16b(tensor: torch.Tensor) -> bool:
    """Return whether vector loads along features satisfy 16-byte address alignment."""
    element_bytes = tensor.element_size()
    return (
        (8 * element_bytes) % 16 == 0
        and tensor.stride(3) == 1
        and tensor.data_ptr() % 16 == 0
        and all((stride * element_bytes) % 16 == 0 for stride in tensor.stride()[:3])
    )


@gluon.jit
def resolve_vec_tile_ids(heads, heads_inner, head_dim: gl.constexpr):
    """Decode ``(batch_index, tile_index, head_base, batch_head_index)`` from grid IDs."""
    program_id_0 = gl.program_id(0)
    program_id_1 = gl.program_id(1)
    batch_index = gl.program_id(2)
    heads_per_program: gl.constexpr = 128 // head_dim
    column_block = gl.where(heads_inner != 0, program_id_0, program_id_1)
    tile_index = gl.where(heads_inner != 0, program_id_1, program_id_0)
    head_base = column_block * heads_per_program
    batch_head_index = batch_index * heads + head_base
    return batch_index, tile_index, head_base, batch_head_index


@gluon.jit
def round_to_int8(values):
    """Round symmetrically and clamp to INT8 codes in ``[-127, 127]``."""
    rounded = values + 0.5 * gl.where(values >= 0, 1.0, -1.0)
    return gl.maximum(-GL_INT8_RANGE, gl.minimum(GL_INT8_RANGE, rounded)).to(gl.int8)


@gluon.jit
def compute_int8_scale(maximum):
    """Compute positive INT8 quantization scale from an absolute maximum."""
    return maximum / GL_INT8_RANGE + GL_SCALE_EPSILON


@gluon.jit
def compute_hadamard_signs(column_offsets):
    """Return ``+1.0``/``-1.0`` signs for ``column_offsets`` in ``[0, 128)``."""
    word_group = column_offsets // 32
    words = gl.where(
        word_group == 0,
        _GL_HADAMARD_W0,
        gl.where(
            word_group == 1,
            _GL_HADAMARD_W1,
            gl.where(word_group == 2, _GL_HADAMARD_W2, _GL_HADAMARD_W3),
        ),
    ).to(gl.uint32)
    return gl.where(((words >> (column_offsets % 32).to(gl.uint32)) & 1) != 0, 1.0, -1.0)


@gluon.jit
def mask_query_group_scales(
    raw_scale,
    group_offsets,
    seq_first_row,
    query_sequence_end,
    block_length,
    scale_factor,
    mask_rows: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    two_heads: gl.constexpr,
):
    """Keep positive denominators while zeroing stored Q32 scales for empty groups."""
    if not mask_rows:
        return raw_scale, raw_scale * scale_factor
    group_valid = (seq_first_row + group_offsets * GL_QUERY_SCALE_ROWS) < query_sequence_end
    if mask_block_lengths:
        group_valid = group_valid & (group_offsets * GL_QUERY_SCALE_ROWS < block_length)
    valid_mask = group_valid[:, None] if two_heads else group_valid
    return gl.where(valid_mask, raw_scale, 1.0), gl.where(valid_mask, raw_scale * scale_factor, 0.0)


@gluon.jit
def load_tile_vec(
    input_ptr,
    batch_index,
    head_base,
    tile_index,
    stride_b,
    stride_h,
    stride_n,
    block_lengths_ptr,
    logical_sequence_length,
    heads,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    mask_heads: gl.constexpr,
    aligned_16b: gl.constexpr,
):
    """Load one ``[64, 128]`` tile (1x D128 or 2x D64 heads) into ``VEC_TILE_LAYOUT``."""
    rows_local = gl.arange(0, 64, gl.SliceLayout(1, VEC_TILE_LAYOUT))
    columns = gl.arange(0, 128, gl.SliceLayout(0, VEC_TILE_LAYOUT))
    sequence_rows = tile_index.to(gl.int64) * GL_TILE_ROWS + rows_local

    if head_dim == 64:
        local_head = columns // 64
        features = columns % 64
        offsets = (
            batch_index.to(gl.int64) * stride_b
            + (head_base + local_head[None, :]).to(gl.int64) * stride_h
            + sequence_rows[:, None] * stride_n
            + features[None, :]
        )
    else:
        offsets = (
            batch_index.to(gl.int64) * stride_b
            + head_base.to(gl.int64) * stride_h
            + sequence_rows[:, None] * stride_n
            + columns[None, :]
        )

    if aligned_16b:
        ptrs = gl.max_contiguous(gl.multiple_of(input_ptr + offsets, [1, 16]), [1, 8])
    else:
        ptrs = input_ptr + offsets

    if mask_rows:
        if mask_block_lengths:
            block_length = gl.load(block_lengths_ptr + tile_index)
            row_valid = rows_local < block_length
        else:
            row_valid = sequence_rows < logical_sequence_length
        if mask_heads:
            head_valid = (head_base + (columns // 64)) < heads
            valid_mask = row_valid[:, None] & head_valid[None, :]
        else:
            valid_mask = row_valid[:, None]
        values = gl.load(ptrs, mask=valid_mask, other=0.0).to(gl.float32)
    else:
        row_valid = gl.full([64], True, gl.int1, gl.SliceLayout(1, VEC_TILE_LAYOUT))
        if mask_heads:
            head_valid = (head_base + (columns // 64)) < heads
            values = gl.load(ptrs, mask=head_valid[None, :], other=0.0).to(gl.float32)
        else:
            values = gl.load(ptrs).to(gl.float32)
    return values, row_valid


@gluon.jit
def reduce_column_minmax_vec(
    values,
    row_valid,
    mask_rows: gl.constexpr,
    broadcast_to_tile: gl.constexpr,
):
    """Reduce ``[64, 128]`` across 8 warps into per-column ``(max, min)``."""
    if mask_rows:
        valid = gl.convert_layout(row_valid, gl.SliceLayout(1, values.type.layout))
        inf_pad = gl.where(valid[:, None], 0.0, float("inf"))
        max_input = gl.reshape(values - inf_pad, [8, 8, 128])
        min_input = gl.reshape(values + inf_pad, [8, 8, 128])
        warp_max = gl.max(max_input, axis=1)
        warp_min = gl.min(min_input, axis=1)
    else:
        grouped = gl.reshape(values, [8, 8, 128])
        warp_max = gl.max(grouped, axis=1)
        warp_min = gl.min(grouped, axis=1)

    packed = gl.permute(gl.join(warp_max, warp_min), [2, 0, 1])
    per_column = gl.convert_layout(packed, MINMAX_REDUCE_LAYOUT)
    reg_max, reg_min = gl.split(gl.permute(per_column, [1, 2, 0]))
    col_max = gl.max(reg_max, axis=0)
    col_min = gl.min(reg_min, axis=0)
    if broadcast_to_tile:
        tile_row_layout: gl.constexpr = gl.SliceLayout(0, VEC_TILE_LAYOUT)
        col_max = gl.convert_layout(col_max, tile_row_layout)
        col_min = gl.convert_layout(col_min, tile_row_layout)
    return col_max, col_min


@gluon.jit
def reduce_column_mean_vec(values, valid_count_f32):
    """Compute exact columnwise mean of zero-padded ``[64, 128]`` in ``SliceLayout(0, tile)``."""
    warp_sum = gl.sum(gl.reshape(values, [8, 8, 128]), axis=1)
    per_column = gl.convert_layout(warp_sum, MEAN_BCAST_LAYOUT)
    return gl.sum(per_column, axis=0) / valid_count_f32


@gluon.jit
def _swizzle_swap(values, stage_index: gl.constexpr):
    """Wave32 intra-lane exchange via batched ``ds_swizzle_b32`` + explicit DS wait."""
    return gl.inline_asm_elementwise(
        asm=_SWIZZLE_SWAP_ASM[stage_index],
        constraints=_SWIZZLE_SWAP_CONSTRAINTS,
        args=[values],
        dtype=gl.float32,
        is_pure=True,
        pack=8,
    )


@gluon.jit
def apply_signed_hadamard_vec(values, head_dim: gl.constexpr):
    """Apply signed Hadamard-64 or Hadamard-128 in registers and lane swizzles."""
    columns = gl.arange(0, 128, gl.SliceLayout(0, values.type.layout))
    rotated = values * compute_hadamard_signs(columns % head_dim)[None, :]

    # Stages 1, 2, and 4 stay in thread-local registers.
    low, high = gl.split(gl.reshape(rotated, [64, 64, 2]))
    rotated = gl.reshape(gl.join(low + high, low - high), [64, 128])
    low, high = gl.split(gl.permute(gl.reshape(rotated, [64, 32, 2, 2]), [0, 1, 3, 2]))
    rotated = gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 3, 2]), [64, 128])
    low, high = gl.split(gl.permute(gl.reshape(rotated, [64, 16, 2, 4]), [0, 1, 3, 2]))
    rotated = gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 3, 2]), [64, 128])

    # Stages 8, 16, 32 (and 64 for D128) map directly onto lane bits 0..3 via ds_swizzle_b32.
    peer8 = _swizzle_swap(rotated, 0)
    rotated = gl.where(((columns & 8) != 0)[None, :], peer8 - rotated, rotated + peer8)
    peer16 = _swizzle_swap(rotated, 1)
    rotated = gl.where(((columns & 16) != 0)[None, :], peer16 - rotated, rotated + peer16)
    peer32 = _swizzle_swap(rotated, 2)
    rotated = gl.where(((columns & 32) != 0)[None, :], peer32 - rotated, rotated + peer32)

    if head_dim == 64:
        return rotated * GL_D64_NORM
    peer64 = _swizzle_swap(rotated, 3)
    rotated = gl.where(((columns & 64) != 0)[None, :], peer64 - rotated, rotated + peer64)
    return rotated * GL_D128_NORM


@gluon.jit
def store_head_major_vec(
    output_ptr,
    tile_int8,
    batch_head_index,
    tile_index,
    storage_tiles,
    head_base,
    heads,
    head_dim: gl.constexpr,
    mask_heads: gl.constexpr,
):
    """Store one ``[64, 128]`` INT8 tile into ``[batch, heads, storage_rows, head_dim]``."""
    layout: gl.constexpr = tile_int8.type.layout
    rows = gl.arange(0, 64, gl.SliceLayout(1, layout))
    columns = gl.arange(0, 128, gl.SliceLayout(0, layout))
    batch_head_64 = batch_head_index.to(gl.int64)
    storage_rows = storage_tiles.to(gl.int64) * GL_TILE_ROWS
    row_base = tile_index.to(gl.int64) * GL_TILE_ROWS + rows[:, None]
    if head_dim == 64:
        local_head = columns // 64
        offsets = (
            (batch_head_64 + local_head[None, :].to(gl.int64)) * storage_rows + row_base
        ) * 64 + (columns % 64)[None, :]
        if mask_heads:
            gl.store(
                output_ptr + offsets,
                tile_int8,
                mask=((head_base + local_head) < heads)[None, :],
            )
        else:
            gl.store(output_ptr + offsets, tile_int8)
    else:
        offsets = (batch_head_64 * storage_rows + row_base) * 128 + columns[None, :]
        gl.store(output_ptr + offsets, tile_int8)


@gluon.jit
def store_summary_vec(
    output_ptr,
    summary,
    batch_head_index,
    tile_index,
    tiles_stride,
    head_base,
    heads,
    head_dim: gl.constexpr,
    mask_heads: gl.constexpr,
):
    """Store one ``[128]`` summary/mean vector into ``[batch, heads, tiles, head_dim]``."""
    columns = gl.arange(0, 128, summary.type.layout)
    batch_head_64 = batch_head_index.to(gl.int64)
    tiles_64 = tiles_stride.to(gl.int64)
    tile_index_64 = tile_index.to(gl.int64)
    if head_dim == 64:
        local_head = columns // 64
        offsets = ((batch_head_64 + local_head.to(gl.int64)) * tiles_64 + tile_index_64) * 64 + (
            columns % 64
        )
        if mask_heads:
            gl.store(output_ptr + offsets, summary, mask=(head_base + local_head) < heads)
        else:
            gl.store(output_ptr + offsets, summary)
    else:
        offsets = (batch_head_64 * tiles_64 + tile_index_64) * 128 + columns
        gl.store(output_ptr + offsets, summary)


@gluon.jit(do_not_specialize=PREP_COMMON_DO_NOT_SPECIALIZE)
def _prepare_query_kernel(
    query_ptr,
    query_int8_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    storage_tiles,
    stride_b,
    stride_h,
    stride_n,
    heads,
    heads_inner,
    softmax_scale,
    head_dim: gl.constexpr,
    emit_summaries: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    mask_heads: gl.constexpr,
    aligned_16b: gl.constexpr,
):
    """Quantize queries with signed Hadamard rotation and optional min+max summaries."""
    batch_index, tile_index, head_base, batch_head_index = resolve_vec_tile_ids(
        heads, heads_inner, head_dim
    )
    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    tile_index_64 = tile_index.to(gl.int64)

    values, row_valid = load_tile_vec(
        query_ptr,
        batch_index,
        head_base,
        tile_index,
        stride_b,
        stride_h,
        stride_n,
        block_lengths_ptr,
        logical_sequence_length,
        heads,
        head_dim,
        mask_block_lengths,
        mask_rows,
        mask_heads,
        aligned_16b,
    )

    if emit_summaries:
        col_max, col_min = reduce_column_minmax_vec(
            values,
            row_valid,
            mask_rows=mask_rows,
            broadcast_to_tile=False,
        )
        store_summary_vec(
            query_summary_ptr,
            col_max + col_min,
            batch_head_index,
            tile_index,
            storage_tiles,
            head_base,
            heads,
            head_dim=head_dim,
            mask_heads=mask_heads,
        )

    rotated = apply_signed_hadamard_vec(values, head_dim)
    scale_factor = softmax_scale * GL_LOG2_E
    group_stride = storage_tiles_64 * 2
    sequence_first_row = tile_index_64 * GL_TILE_ROWS

    if head_dim == 64:
        grouped = gl.reshape(rotated, [2, GL_QUERY_SCALE_ROWS, 2, 64])
        raw_scale = compute_int8_scale(gl.max(gl.max(gl.abs(grouped), axis=3), axis=1))
        group_offsets = gl.arange(0, 2, gl.SliceLayout(1, raw_scale.type.layout))
        head_offsets = gl.arange(0, 2, gl.SliceLayout(0, raw_scale.type.layout))
        # Standalone Q32 scale masking follows logical sequence bounds even under block_lengths.
        raw_scale, stored_scale = mask_query_group_scales(
            raw_scale,
            group_offsets,
            sequence_first_row,
            logical_sequence_length,
            GL_TILE_ROWS,
            scale_factor,
            mask_rows=mask_rows,
            mask_block_lengths=False,
            two_heads=True,
        )
        query_codes = gl.reshape(round_to_int8(grouped / raw_scale[:, None, :, None]), [64, 128])
        store_head_major_vec(
            query_int8_ptr,
            query_codes,
            batch_head_index,
            tile_index,
            storage_tiles,
            head_base,
            heads,
            head_dim=head_dim,
            mask_heads=mask_heads,
        )
        scale_offsets = (batch_head_64 + head_offsets[None, :].to(gl.int64)) * group_stride + (
            tile_index_64 * 2 + group_offsets[:, None]
        )
        if mask_heads:
            gl.store(
                query_scale_ptr + scale_offsets,
                stored_scale,
                mask=(head_base + head_offsets[None, :]) < heads,
            )
        else:
            gl.store(query_scale_ptr + scale_offsets, stored_scale)
    else:
        grouped = gl.reshape(rotated, [2, GL_QUERY_SCALE_ROWS, 128])
        raw_scale = compute_int8_scale(gl.max(gl.max(gl.abs(grouped), axis=2), axis=1))
        group_offsets = gl.arange(0, 2, raw_scale.type.layout)
        raw_scale, stored_scale = mask_query_group_scales(
            raw_scale,
            group_offsets,
            sequence_first_row,
            logical_sequence_length,
            GL_TILE_ROWS,
            scale_factor,
            mask_rows=mask_rows,
            mask_block_lengths=False,
            two_heads=False,
        )
        query_codes = gl.reshape(round_to_int8(grouped / raw_scale[:, None, None]), [64, 128])
        store_head_major_vec(
            query_int8_ptr,
            query_codes,
            batch_head_index,
            tile_index,
            storage_tiles,
            head_base,
            heads,
            head_dim=head_dim,
            mask_heads=False,
        )
        scale_offsets = batch_head_64 * group_stride + (tile_index_64 * 2 + group_offsets)
        gl.store(query_scale_ptr + scale_offsets, stored_scale)


def _prepare_query_impl(
    query: torch.Tensor,
    scale: float,
    *,
    storage_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
    emit_summaries: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch, heads, logical_sequence_length, head_dim = query.shape
    storage_tiles = storage_sequence_length // TILE_ROWS
    query_int8 = torch.empty(
        (batch, heads, storage_sequence_length, head_dim),
        device=query.device,
        dtype=torch.int8,
    )
    query_scale = torch.empty(
        (batch, heads, storage_tiles * 2),
        device=query.device,
        dtype=torch.float32,
    )
    query_summary = (
        torch.empty(
            (batch, heads, storage_tiles, head_dim),
            device=query.device,
            dtype=torch.float32,
        )
        if emit_summaries
        else None
    )
    if batch == 0 or heads == 0 or storage_tiles == 0:
        return query_int8, query_scale, query_summary
    if logical_sequence_length == 0:
        with device_context(query.device):
            query_int8.zero_()
            query_scale.zero_()
            if query_summary is not None:
                query_summary.zero_()
        return query_int8, query_scale, query_summary

    grid, heads_inner = select_block_grid(query, storage_tiles, heads_per_program=128 // head_dim)
    mask_block_lengths = block_lengths is not None
    mask_rows = (logical_sequence_length != storage_sequence_length) or mask_block_lengths
    mask_heads = head_dim == 64 and (heads % 2 != 0)
    aligned_16b = is_aligned_16b(query)
    summary_buffer = query_summary if query_summary is not None else query_scale
    lengths_buffer = block_lengths if block_lengths is not None else query_scale
    with device_context(query.device):
        _prepare_query_kernel[grid](
            query,
            query_int8,
            query_scale,
            summary_buffer,
            lengths_buffer,
            logical_sequence_length,
            storage_tiles,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            heads,
            heads_inner,
            float(scale),
            head_dim,
            emit_summaries,
            mask_block_lengths,
            mask_rows,
            mask_heads,
            aligned_16b,
            num_warps=8,
            num_stages=1,
            llvm_fn_attrs=PREP_LLVM_FN_ATTRS,
        )
    return query_int8, query_scale, query_summary


def prepare_query(
    query: torch.Tensor,
    scale: float,
    *,
    storage_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize queries with signed Hadamard rotation into padded head-major storage."""
    query_int8, query_scale, _summary = _prepare_query_impl(
        query,
        scale,
        storage_sequence_length=storage_sequence_length,
        block_lengths=block_lengths,
        emit_summaries=False,
    )
    return query_int8, query_scale


def prepare_query_with_summary(
    query: torch.Tensor,
    scale: float,
    *,
    storage_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize queries and emit per-K64-tile min+max routing summaries."""
    query_int8, query_scale, query_summary = _prepare_query_impl(
        query,
        scale,
        storage_sequence_length=storage_sequence_length,
        block_lengths=block_lengths,
        emit_summaries=True,
    )
    assert query_summary is not None
    return query_int8, query_scale, query_summary
