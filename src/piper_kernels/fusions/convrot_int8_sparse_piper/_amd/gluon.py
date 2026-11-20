"""RDNA4 Gluon kernels for fused ConvRot INT8 sparse-Piper Q/K/V projections.

Each workgroup projects a 128-row by 128-column tile (`4x2` wave grid, `16 KB` LDS)
using two K64 stages of bank-conflict-free LDS feeding `v_wmma_i32_16x16x16_iu8`:
one D128 head or two D64 heads per program. Swapping feature bit 4 (`16`) and bit 6
(`64`) during B loads keeps each wave on contiguous 64-column feature halves and places
accumulator halves `[64, 128]` into separate register ranges for zero-cost top/bottom
splitting. Six stages of the signed Hadamard run in registers, stage 8 runs via
`v_permlanex16_b32`, and only D128 stage 64 (and D128 R96/R128 RoPE) exchanges across
the two N-waves via epilogue LDS. Columnwise routing summaries and K/V statistics use
within-warp reductions followed by a single cross-wave `convert_layout` barrier.
"""

# Gluon exposes low-level signatures and layouts that are not modeled by static type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3, rdna4

from piper_kernels._triton.amd_compiler_workarounds import LDS_ONLY_FENCE_ATTRS
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.piper._amd.fragments import AMDWMMALayout
from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import (
    SIGNED_HADAMARD_MASK,
)
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MEAN_ROUTING
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from .._layout import (
    QUERY_SCALE_ROWS,
    TILE_ROWS,
    resolve_row_block_launches,
)
from .policy import (
    GLUON_ACCUMULATOR_SEED,
    GLUON_ACCUMULATOR_SEED_MIN_K,
    GLUON_BLOCK_K,
    GLUON_BLOCK_M,
)

# ---------------------------------------------------------------------------
# RDNA4 wave32 4x2 (128x128x64) Gluon layouts
# ---------------------------------------------------------------------------

_WMMA_4X2_LAYOUT = gl.constexpr(
    AMDWMMALayout(version=2, transposed=True, warp_bases=[[1, 0], [2, 0], [0, 1]])
)
_INPUT_SHARED_LAYOUT = gl.constexpr(
    gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[1, 0])
)
_WEIGHT_SHARED_LAYOUT = gl.constexpr(
    gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[0, 1])
)
_INPUT_LOAD_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0]))
_WEIGHT_LOAD_LAYOUT = gl.constexpr(gl.BlockedLayout([8, 1], [8, 4], [1, 8], [0, 1]))
_INPUT_DOT_LAYOUT = gl.constexpr(gl.DotOperandLayout(0, _WMMA_4X2_LAYOUT, k_width=8))
_WEIGHT_DOT_LAYOUT = gl.constexpr(gl.DotOperandLayout(1, _WMMA_4X2_LAYOUT, k_width=8))

# Physical [128, 128] WMMA layout, before reindexing swapped B columns [4] <-> [6].
_WMMA_PHYSICAL_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 32], [0, 64], [64, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        warp_bases=[[16, 0], [32, 0], [0, 16]],
        block_bases=[],
        shape=[128, 128],
    )
)
# One K64 half [64, 128] after splitting M=64 and unswapping column bit 4 <-> bit 6:
# N-warp 0 owns columns 0..63, N-warp 1 owns columns 64..127.
_TILE_HALF_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        warp_bases=[[16, 0], [32, 0], [0, 64]],
        block_bases=[],
        shape=[64, 128],
    )
)
# Cross-N-warp exchange over [64, 2, 32] (16 KB FP32): moves axis-1 (size 2) into registers.
_WARP_EXCHANGE_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 1, 0]],
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 8]],
        warp_bases=[[16, 0, 0], [32, 0, 0], [0, 0, 16]],
        block_bases=[],
        shape=[64, 2, 32],
    )
)
# One-way post-exchange [64, 128] layout (avoids a return LDS trip for D128 Hadamard).
_POST_EXCHANGE_HALF_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 64], [0, 32]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        warp_bases=[[16, 0], [32, 0], [0, 16]],
        block_bases=[],
        shape=[64, 128],
    )
)
# One-barrier cross-M-warp column reduction layouts (distributes 128 columns across warps/lanes).
_MINMAX_REDUCE_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0]],
        lane_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 0]],
        warp_bases=[[0, 0, 16], [0, 0, 32], [0, 0, 64]],
        block_bases=[],
        shape=[2, 4, 128],
    )
)
_MEAN_REDUCE_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0]],
        lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        warp_bases=[[0, 32], [0, 64], [0, 0]],
        block_bases=[],
        shape=[4, 128],
    )
)
_VALUE_STORE_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 8], [4, 8], [8, 1], [1, 0]))

_GL_BLOCK_M = gl.constexpr(GLUON_BLOCK_M)
_GL_BLOCK_K = gl.constexpr(GLUON_BLOCK_K)
_GL_TILE_ROWS = gl.constexpr(TILE_ROWS)
_GL_TILE_ROWS_F32 = gl.constexpr(float(TILE_ROWS))
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)
_GL_SEED = gl.constexpr(GLUON_ACCUMULATOR_SEED)
_GL_SEED_MIN_K = gl.constexpr(GLUON_ACCUMULATOR_SEED_MIN_K)
_GL_HADAMARD_W0 = gl.constexpr(SIGNED_HADAMARD_MASK[0])
_GL_HADAMARD_W1 = gl.constexpr(SIGNED_HADAMARD_MASK[1])
_GL_HADAMARD_W2 = gl.constexpr(SIGNED_HADAMARD_MASK[2])
_GL_HADAMARD_W3 = gl.constexpr(SIGNED_HADAMARD_MASK[3])
_GL_LOG2_E = gl.constexpr(1.4426950408889634)
_GL_INT8_RANGE = gl.constexpr(127.0)
_GL_SCALE_EPSILON = gl.constexpr(1e-7)
_GL_VALUE_MEAN_SCALE_FLOOR = gl.constexpr(2.0**-80)
_GL_P_UINT8_RANGE = gl.constexpr(255.0)
_GL_D128_NORM = gl.constexpr(0.08838834764831845)
_GL_D64_NORM = gl.constexpr(0.125)

_DO_NOT_SPECIALIZE = (
    "logical_sequence_length",
    "row_block_offset",
    "storage_tiles",
    "num_row_blocks",
)
_DO_NOT_SPECIALIZE_QUERY = (*_DO_NOT_SPECIALIZE, "chunk_start", "query_sequence_end")

_BUFFER_OFFSET_LIMIT = 1 << 31


def _build_kernel_options(*, waves_per_eu: str | None = None) -> dict[str, object]:
    """Build LLVM function attributes keeping LDS-only fences and optional occupancy hints."""
    attrs: tuple[tuple[str, str], ...] = LDS_ONLY_FENCE_ATTRS
    if waves_per_eu is not None:
        attrs = (("amdgpu-waves-per-eu", waves_per_eu), *attrs)
    return {"llvm_fn_attrs": attrs} if attrs else {}


def _build_key_kernel_options(
    head_dim: int,
    rotary_dim: int,
    *,
    mask_rows: bool,
    mask_block_lengths: bool,
    mean_pool_summary: bool,
) -> dict[str, object]:
    """Keep 6-8 waves/EU on D64 and D128 R<=96 fast paths; relax ragged minmax tails and R128."""
    use_occupancy_hint = head_dim == 64 or (
        rotary_dim < 128 and (not mask_rows or mask_block_lengths or mean_pool_summary)
    )
    return _build_kernel_options(waves_per_eu="6,8" if use_occupancy_hint else None)


def operands_aligned(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    *,
    storage_sequence_length: int,
    rotary_dim: int = 0,
    head_dim: int = 128,
) -> bool:
    """Return whether operands are contiguous and fit signed 32-bit buffer offsets."""
    if not (input_qdata.is_contiguous() and weight_qdata.is_contiguous()):
        return False
    if head_dim not in (64, 128):
        return False
    _batch, sequence_length, input_features = input_qdata.shape
    output_features = weight_qdata.shape[0]
    heads_per_program = 128 // head_dim
    # Per-batch / per-head-block buffer_{load,store} uses signed 32-bit byte/element offsets.
    return (
        sequence_length * input_features < _BUFFER_OFFSET_LIMIT
        and 128 * input_features < _BUFFER_OFFSET_LIMIT
        and output_features < _BUFFER_OFFSET_LIMIT
        and (sequence_length * max(rotary_dim, 1) + head_dim) * 4 < _BUFFER_OFFSET_LIMIT
        and heads_per_program * storage_sequence_length * head_dim < _BUFFER_OFFSET_LIMIT
    )


@gluon.jit
def _resolve_tile_ids(
    group_m: gl.constexpr, num_row_blocks, heads: gl.constexpr, head_dim: gl.constexpr
):
    """Map linear program IDs to ``(row_block, column_block)`` with optional L2 swizzle."""
    program_id = gl.program_id(0)
    num_col_blocks: gl.constexpr = (heads * head_dim) // 128
    if group_m > 0:
        group_size = group_m * num_col_blocks
        group_index = program_id // group_size
        first_row_block = group_index * group_m
        group_row_blocks = gl.minimum(num_row_blocks - first_row_block, group_m)
        within_group = program_id % group_size
        return first_row_block + (within_group % group_row_blocks), within_group // group_row_blocks
    return program_id % num_row_blocks, program_id // num_row_blocks


@gluon.jit
def _swap_feature_bit_4_6(column_index):
    bit4 = (column_index >> 4) & 1
    bit6 = (column_index >> 6) & 1
    diff = bit4 ^ bit6
    return column_index ^ (diff << 4) ^ (diff << 6)


@gluon.jit
def _exchange_lanes_x16(values):
    """Swap 16-lane halves inside wave32 via RDNA ``v_permlanex16_b32``."""
    return gl.inline_asm_elementwise(
        asm="v_permlanex16_b32 $0, $1, $2, $3",
        constraints="=v,v,s,s",
        args=[values, gl.to_tensor(0x76543210).to(gl.int32), gl.to_tensor(-19088744).to(gl.int32)],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _round_to_int8(values):
    rounded = values + 0.5 * gl.where(values >= 0, 1.0, -1.0)
    return gl.maximum(-_GL_INT8_RANGE, gl.minimum(_GL_INT8_RANGE, rounded)).to(gl.int8)


@gluon.jit
def _compute_int8_scale(maximum):
    return maximum / _GL_INT8_RANGE + _GL_SCALE_EPSILON


@gluon.jit
def _compute_hadamard_signs(column_offsets):
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
def _project_int8_tile(
    input_base_ptr,
    weight_base_ptr,
    first_sequence_row,
    logical_sequence_length,
    input_features: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project 128x128 INT8 tiles through two K64 LDS stages using 4x2 wave32 WMMA."""
    input_shared = gl.allocate_shared_memory(gl.int8, [128, 64], _INPUT_SHARED_LAYOUT)
    weight_shared = gl.allocate_shared_memory(gl.int8, [64, 128], _WEIGHT_SHARED_LAYOUT)

    input_rows = first_sequence_row + gl.arange(0, 128, gl.SliceLayout(1, _INPUT_LOAD_LAYOUT))
    if mask_rows:
        input_rows = gl.minimum(input_rows, logical_sequence_length - 1)
    input_k = gl.arange(0, 64, gl.SliceLayout(0, _INPUT_LOAD_LAYOUT))
    input_offsets = input_rows[:, None] * input_features + input_k[None, :]

    weight_k = gl.arange(0, 64, gl.SliceLayout(1, _WEIGHT_LOAD_LAYOUT))
    weight_columns = _swap_feature_bit_4_6(
        gl.arange(0, 128, gl.SliceLayout(0, _WEIGHT_LOAD_LAYOUT))
    )
    weight_offsets = weight_columns[None, :] * input_features + weight_k[:, None]

    use_seed: gl.constexpr = input_features >= _GL_SEED_MIN_K
    if use_seed:
        accumulator = gl.full([128, 128], _GL_SEED, gl.int32, layout=_WMMA_4X2_LAYOUT)
    else:
        accumulator = gl.zeros([128, 128], gl.int32, layout=_WMMA_4X2_LAYOUT)

    input_shared.store(cdna3.buffer_load(input_base_ptr, input_offsets))
    weight_shared.store(cdna3.buffer_load(weight_base_ptr, weight_offsets))
    gl.barrier()

    num_k_blocks: gl.constexpr = input_features // _GL_BLOCK_K
    for k_block_index in range(1, num_k_blocks):
        input_next = cdna3.buffer_load(input_base_ptr, input_offsets + k_block_index * _GL_BLOCK_K)
        weight_next = cdna3.buffer_load(
            weight_base_ptr, weight_offsets + k_block_index * _GL_BLOCK_K
        )
        accumulator = rdna4.wmma(
            input_shared.load(_INPUT_DOT_LAYOUT),
            weight_shared.load(_WEIGHT_DOT_LAYOUT),
            accumulator,
        )
        gl.barrier()
        input_shared.store(input_next)
        weight_shared.store(weight_next)
        gl.barrier()

    accumulator = rdna4.wmma(
        input_shared.load(_INPUT_DOT_LAYOUT),
        weight_shared.load(_WEIGHT_DOT_LAYOUT),
        accumulator,
    )
    if use_seed:
        accumulator = accumulator - _GL_SEED
    return accumulator


@gluon.jit
def _split_top_bot_logical(accumulator_wmma):
    """Reinterpret WMMA [128, 128] (col4<->col6 swapped) into two [64, 128] halves."""
    physical = gl.convert_layout(accumulator_wmma, _WMMA_PHYSICAL_LAYOUT)
    reshaped = gl.reshape(physical, [2, 64, 2, 2, 2, 16])
    reordered = gl.permute(reshaped, [1, 4, 3, 2, 5, 0])
    half_top, half_bot = gl.split(gl.reshape(reordered, [64, 128, 2]))
    return gl.convert_layout(half_top, _TILE_HALF_LAYOUT), gl.convert_layout(
        half_bot, _TILE_HALF_LAYOUT
    )


@gluon.jit
def _split_quarters_16(values):
    """Split [64, 128] along its 16-column chunks inside _TILE_HALF_LAYOUT."""
    chunked = gl.reshape(values, [64, 2, 2, 2, 16])
    half_01, half_23 = gl.split(gl.permute(chunked, [0, 1, 3, 4, 2]))
    quarter_0, quarter_1 = gl.split(gl.permute(half_01, [0, 1, 3, 2]))
    quarter_2, quarter_3 = gl.split(gl.permute(half_23, [0, 1, 3, 2]))
    return quarter_0, quarter_1, quarter_2, quarter_3


@gluon.jit
def _join_quarters_16(quarter_0, quarter_1, quarter_2, quarter_3):
    """Reassemble four [64, 2, 16] quarter-chunks into _TILE_HALF_LAYOUT."""
    half_01 = gl.permute(gl.join(quarter_0, quarter_1), [0, 1, 3, 2])
    half_23 = gl.permute(gl.join(quarter_2, quarter_3), [0, 1, 3, 2])
    chunked = gl.permute(gl.join(half_01, half_23), [0, 1, 4, 2, 3])
    return gl.convert_layout(gl.reshape(chunked, [64, 128]), _TILE_HALF_LAYOUT)


@gluon.jit
def _rotate_half(values, head_dim: gl.constexpr, rotary_dim: gl.constexpr):
    """Return split-half RoPE partners ``(-second_half, first_half)`` in ``_TILE_HALF_LAYOUT``."""
    if rotary_dim == 64:
        # Both D64 (within each head) and D128 (N-warp 0) hold both 32-col halves in registers.
        half_lo, half_hi = gl.split(gl.permute(gl.reshape(values, [64, 2, 2, 32]), [0, 1, 3, 2]))
        joined = gl.permute(gl.join(-half_hi, half_lo), [0, 1, 3, 2])
        return gl.convert_layout(gl.reshape(joined, [64, 128]), _TILE_HALF_LAYOUT)
    if rotary_dim == 32:
        quarter_0, quarter_1, quarter_2, quarter_3 = _split_quarters_16(values)
        return _join_quarters_16(-quarter_1, quarter_0, quarter_2, quarter_3)
    if head_dim == 64 and rotary_dim == 48:
        quarter_0, quarter_1, quarter_2, quarter_3 = _split_quarters_16(values)
        peer_0 = _exchange_lanes_x16(quarter_0)
        peer_1 = _exchange_lanes_x16(quarter_1)
        peer_2 = _exchange_lanes_x16(quarter_2)
        in_high_eight = (
            gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, quarter_0.type.layout))) & 8
        ) != 0
        high_mask = in_high_eight[None, None, :]
        rot_0 = gl.where(high_mask, -peer_2, -peer_1)
        rot_1 = gl.where(high_mask, peer_0, -peer_2)
        rot_2 = gl.where(high_mask, peer_1, peer_0)
        return _join_quarters_16(rot_0, rot_1, rot_2, quarter_3)
    if head_dim == 128 and rotary_dim == 96:
        # One 16 KB cross-N-warp exchange swaps [16..47] on N-warp 0 with [64..95] on N-warp 1.
        quarter_0, quarter_1, quarter_2, quarter_3 = _split_quarters_16(values)
        is_nwarp_0 = (
            gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, quarter_0.type.layout))) == 0
        )[None, :, None]
        send_0 = gl.where(is_nwarp_0, quarter_1, quarter_0)
        send_1 = gl.where(is_nwarp_0, quarter_2, quarter_1)
        packed_send = gl.reshape(gl.permute(gl.join(send_0, send_1), [0, 1, 3, 2]), [64, 2, 32])
        exchanged = gl.convert_layout(packed_send, _WARP_EXCHANGE_LAYOUT)
        from_warp_0, from_warp_1 = gl.split(gl.permute(exchanged, [0, 2, 1]))
        packed_recv = gl.convert_layout(
            gl.permute(gl.join(-from_warp_1, from_warp_0), [0, 2, 1]), packed_send.type.layout
        )
        recv_0, recv_1 = gl.split(gl.permute(gl.reshape(packed_recv, [64, 2, 2, 16]), [0, 1, 3, 2]))
        out_0 = gl.where(is_nwarp_0, -quarter_3, recv_0)
        out_1 = gl.where(is_nwarp_0, recv_0, recv_1)
        out_2 = gl.where(is_nwarp_0, recv_1, quarter_2)
        out_3 = gl.where(is_nwarp_0, quarter_0, quarter_3)
        return _join_quarters_16(out_0, out_1, out_2, out_3)
    # Full 128-channel RoPE exchanges both halves across the two N-warps.
    chunk_0, chunk_1 = gl.split(gl.permute(gl.reshape(values, [64, 2, 2, 32]), [0, 1, 3, 2]))
    exch_0 = gl.convert_layout(chunk_0, _WARP_EXCHANGE_LAYOUT)
    lo_0, hi_0 = gl.split(gl.permute(exch_0, [0, 2, 1]))
    rot_chunk_0 = gl.convert_layout(
        gl.permute(gl.join(-hi_0, lo_0), [0, 2, 1]), chunk_0.type.layout
    )
    exch_1 = gl.convert_layout(chunk_1, _WARP_EXCHANGE_LAYOUT)
    lo_1, hi_1 = gl.split(gl.permute(exch_1, [0, 2, 1]))
    rot_chunk_1 = gl.convert_layout(
        gl.permute(gl.join(-hi_1, lo_1), [0, 2, 1]), chunk_1.type.layout
    )
    return gl.convert_layout(
        gl.reshape(gl.permute(gl.join(rot_chunk_0, rot_chunk_1), [0, 1, 3, 2]), [64, 128]),
        _TILE_HALF_LAYOUT,
    )


@gluon.jit
def _apply_hadamard_64(values):
    """Apply 6-stage Walsh-Hadamard to ``[64, 2, 64]`` without LDS."""
    low, high = gl.split(gl.reshape(values, [64, 2, 32, 2]))
    values = gl.reshape(gl.join(low + high, low - high), [64, 2, 64])
    low, high = gl.split(gl.permute(gl.reshape(values, [64, 2, 16, 2, 2]), [0, 1, 2, 4, 3]))
    values = gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 2, 4, 3]), [64, 2, 64])
    low, high = gl.split(gl.permute(gl.reshape(values, [64, 2, 8, 2, 4]), [0, 1, 2, 4, 3]))
    values = gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 2, 4, 3]), [64, 2, 64])
    peer = _exchange_lanes_x16(values)
    high_lane = (
        gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(1, values.type.layout))) & 8
    ) != 0
    values = gl.where(high_lane[None, None, :], peer - values, values + peer)
    low, high = gl.split(gl.permute(gl.reshape(values, [64, 2, 2, 2, 16]), [0, 1, 2, 4, 3]))
    values = gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 2, 4, 3]), [64, 2, 64])
    low, high = gl.split(gl.permute(gl.reshape(values, [64, 2, 2, 32]), [0, 1, 3, 2]))
    return gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 3, 2]), [64, 2, 64])


@gluon.jit
def _apply_hadamard_stage64(values_64x2x64):
    """Apply stage-64 Hadamard across the two N-warps and stay in ``_POST_EXCHANGE_HALF_LAYOUT``."""
    chunk_0, chunk_1 = gl.split(
        gl.permute(gl.reshape(values_64x2x64, [64, 2, 2, 32]), [0, 1, 3, 2])
    )
    exch_0 = gl.convert_layout(chunk_0, _WARP_EXCHANGE_LAYOUT)
    low_0, high_0 = gl.split(gl.permute(exch_0, [0, 2, 1]))
    out_0 = gl.permute(gl.join(low_0 + high_0, low_0 - high_0), [0, 2, 1])
    exch_1 = gl.convert_layout(chunk_1, _WARP_EXCHANGE_LAYOUT)
    low_1, high_1 = gl.split(gl.permute(exch_1, [0, 2, 1]))
    out_1 = gl.permute(gl.join(low_1 + high_1, low_1 - high_1), [0, 2, 1])
    merged = gl.reshape(gl.permute(gl.join(out_0, out_1), [0, 1, 3, 2]), [64, 128])
    return gl.convert_layout(merged, _POST_EXCHANGE_HALF_LAYOUT)


@gluon.jit
def _reduce_column_minmax(values_64x128, row_valid, mask_rows: gl.constexpr):
    """Exact columnwise ``(max, min)`` of ``[64, 128]`` using one cross-M-warp barrier."""
    if mask_rows:
        valid = gl.convert_layout(row_valid, gl.SliceLayout(1, values_64x128.type.layout))
        inf_pad = gl.where(valid[:, None], 0.0, float("inf"))
        max_input = gl.reshape(values_64x128 - inf_pad, [4, 16, 128])
        min_input = gl.reshape(values_64x128 + inf_pad, [4, 16, 128])
        warp_max = gl.max(max_input, axis=1)
        warp_min = gl.min(min_input, axis=1)
    else:
        grouped = gl.reshape(values_64x128, [4, 16, 128])
        warp_max = gl.max(grouped, axis=1)
        warp_min = gl.min(grouped, axis=1)
    packed = gl.permute(gl.join(warp_max, warp_min), [2, 0, 1])
    per_column = gl.convert_layout(packed, _MINMAX_REDUCE_LAYOUT)
    reg_max, reg_min = gl.split(gl.permute(per_column, [1, 2, 0]))
    return gl.max(reg_max, axis=0), gl.min(reg_min, axis=0)


@gluon.jit
def _reduce_column_mean(values_64x128, valid_count_f32):
    """Exact columnwise mean of zero-padded ``[64, 128]`` using one cross-M-warp barrier."""
    warp_sum = gl.sum(gl.reshape(values_64x128, [4, 16, 128]), axis=1)
    per_column = gl.convert_layout(warp_sum, _MEAN_REDUCE_LAYOUT)
    return gl.sum(per_column, axis=0) / valid_count_f32


@gluon.jit
def _apply_rmsnorm(values, head_dim: gl.constexpr, norm_epsilon: gl.constexpr):
    if head_dim == 64:
        grouped = gl.reshape(values, [64, 2, 64])
        inv_rms = gl.rsqrt(gl.sum(grouped * grouped, axis=2) * (1.0 / 64.0) + norm_epsilon)
        return gl.reshape(grouped * inv_rms[:, :, None], [64, 128])
    inv_rms = gl.rsqrt(gl.sum(values * values, axis=1) * (1.0 / 128.0) + norm_epsilon)
    return values * inv_rms[:, None]


@gluon.jit
def _dequantize_halves(
    acc_top,
    acc_bot,
    input_scale_base_ptr,
    weight_scale_base_ptr,
    bias_base_ptr,
    norm_weight_ptr,
    seq_first_row,
    logical_sequence_length,
    head_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    apply_rmsnorm: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Dequantize, add bias, and optionally RMSNorm both halves with disjoint vector lifetimes."""
    rows_local = gl.arange(0, 64, gl.SliceLayout(1, _TILE_HALF_LAYOUT))
    column_offsets = gl.arange(0, 128, gl.SliceLayout(0, _TILE_HALF_LAYOUT))
    rows_top = seq_first_row + rows_local
    rows_bot = seq_first_row + _GL_TILE_ROWS + rows_local
    if mask_rows:
        rows_top = gl.minimum(rows_top, logical_sequence_length - 1)
        rows_bot = gl.minimum(rows_bot, logical_sequence_length - 1)
    input_scale_top = cdna3.buffer_load(input_scale_base_ptr, rows_top).to(gl.float32)
    input_scale_bot = cdna3.buffer_load(input_scale_base_ptr, rows_bot).to(gl.float32)

    weight_scale = cdna3.buffer_load(weight_scale_base_ptr, column_offsets).to(gl.float32)
    half_top = acc_top.to(gl.float32) * (input_scale_top[:, None] * weight_scale[None, :])
    half_bot = acc_bot.to(gl.float32) * (input_scale_bot[:, None] * weight_scale[None, :])

    if bias_base_ptr is not None:
        bias = cdna3.buffer_load(bias_base_ptr, column_offsets).to(gl.float32)
        half_top = half_top + bias[None, :]
        half_bot = half_bot + bias[None, :]

    if apply_rmsnorm:
        half_top = _apply_rmsnorm(half_top, head_dim, norm_epsilon)
        half_bot = _apply_rmsnorm(half_bot, head_dim, norm_epsilon)
        if norm_weight_ptr is not None:
            norm_weight = cdna3.buffer_load(norm_weight_ptr, column_offsets % head_dim).to(
                gl.float32
            )
            half_top = half_top * norm_weight[None, :]
            half_bot = half_bot * norm_weight[None, :]
    return half_top, half_bot


@gluon.jit
def _project_dequantized_halves(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    row_block_offset,
    sequence_row_offset,
    logical_sequence_length,
    storage_tiles,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    apply_rmsnorm: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Run the 128x128 GEMM, split top/bot K64 halves, and return dequantized FP32 tiles."""
    local_row_block, column_block = _resolve_tile_ids(group_m, num_row_blocks, heads, head_dim)
    row_block = row_block_offset + local_row_block
    batch = gl.program_id(1)

    heads_per_program: gl.constexpr = 128 // head_dim
    first_col = column_block * 128
    batch_head_index = batch * heads + column_block * heads_per_program
    tile_top = row_block * 2
    seq_first_row = sequence_row_offset + row_block * _GL_BLOCK_M

    input_base_ptr = input_ptr + batch.to(gl.int64) * logical_sequence_length * input_features
    input_scale_base_ptr = input_scale_ptr + batch.to(gl.int64) * logical_sequence_length
    weight_base_ptr = weight_ptr + first_col.to(gl.int64) * input_features
    weight_scale_base_ptr = weight_scale_ptr + first_col.to(gl.int64)
    bias_base_ptr = bias_ptr + first_col.to(gl.int64) if bias_ptr is not None else None
    batch_head_stride = batch_head_index.to(gl.int64) * storage_tiles.to(gl.int64)

    accumulator = _project_int8_tile(
        input_base_ptr,
        weight_base_ptr,
        seq_first_row,
        logical_sequence_length,
        input_features,
        mask_rows,
    )
    acc_top, acc_bot = _split_top_bot_logical(accumulator)
    half_top, half_bot = _dequantize_halves(
        acc_top,
        acc_bot,
        input_scale_base_ptr,
        weight_scale_base_ptr,
        bias_base_ptr,
        norm_weight_ptr,
        seq_first_row,
        logical_sequence_length,
        head_dim,
        norm_epsilon,
        apply_rmsnorm,
        mask_rows,
    )
    return half_top, half_bot, batch_head_index, batch_head_stride, tile_top, seq_first_row


@gluon.jit
def _rope_and_mask_half(
    values,
    cos_ptr,
    sin_ptr,
    seq_first_row,
    sequence_end,
    logical_sequence_length,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    mask_rows: gl.constexpr,
    block_lengths_ptr,
    global_tile_index,
    mask_block_lengths: gl.constexpr,
):
    """Apply staged RoPE, compute validity mask, and zero invalid rows for one [64, 128] half."""
    rows_local = gl.arange(0, 64, gl.SliceLayout(1, _TILE_HALF_LAYOUT))
    sequence_rows = seq_first_row + rows_local
    if rotary_dim > 0:
        feature_offsets = gl.arange(0, 128, gl.SliceLayout(0, _TILE_HALF_LAYOUT)) % head_dim
        rows_clamped = (
            gl.minimum(sequence_rows, logical_sequence_length - 1) if mask_rows else sequence_rows
        )
        rotary_mask = feature_offsets < rotary_dim
        rotated = _rotate_half(values, head_dim, rotary_dim)
        rope_offsets = rows_clamped[:, None] * rotary_dim + feature_offsets[None, :]
        cos_values = cdna3.buffer_load(
            cos_ptr, rope_offsets, mask=rotary_mask[None, :], other=1.0
        ).to(gl.float32)
        values = gl.where(rotary_mask[None, :], values * cos_values, values)
        sin_values = cdna3.buffer_load(
            sin_ptr, rope_offsets, mask=rotary_mask[None, :], other=0.0
        ).to(gl.float32)
        values = gl.where(rotary_mask[None, :], values + rotated * sin_values, values)

    if mask_rows:
        row_valid = sequence_rows < sequence_end
        if mask_block_lengths:
            block_length = gl.load(
                block_lengths_ptr + global_tile_index,
                mask=seq_first_row < sequence_end,
                other=0,
            )
            row_valid = row_valid & (rows_local < block_length)
        else:
            block_length = _GL_TILE_ROWS
        values = gl.where(row_valid[:, None], values, 0.0)
    else:
        block_length = _GL_TILE_ROWS
        row_valid = sequence_rows < sequence_end
    return values, row_valid, block_length


@gluon.jit
def _store_tile_summary(
    summary_ptr,
    values,
    batch_head_index,
    tile_index,
    storage_tiles,
    head_dim: gl.constexpr,
):
    column_offsets = gl.arange(0, 128, values.type.layout)
    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    if head_dim == 64:
        head_offset = (column_offsets // 64).to(gl.int64)
        feature_offset = column_offsets % 64
        offsets = ((batch_head_64 + head_offset) * storage_tiles_64 + tile_index) * 64 + (
            feature_offset
        )
    else:
        offsets = (batch_head_64 * storage_tiles_64 + tile_index) * 128 + column_offsets
    gl.store(summary_ptr + offsets, values)


@gluon.jit
def _store_head_major_i8(
    base_ptr,
    int8_tile,
    tile_index,
    storage_rows,
    head_dim: gl.constexpr,
):
    layout: gl.constexpr = int8_tile.type.layout
    rows = gl.arange(0, 64, gl.SliceLayout(1, layout))
    columns = gl.arange(0, 128, gl.SliceLayout(0, layout))
    if head_dim == 64:
        offsets = (
            (columns // 64)[None, :] * storage_rows + (tile_index * _GL_TILE_ROWS + rows[:, None])
        ) * 64 + (columns % 64)[None, :]
    else:
        offsets = (tile_index * _GL_TILE_ROWS + rows[:, None]) * 128 + columns[None, :]
    cdna3.buffer_store(int8_tile, base_ptr, offsets)


@gluon.jit
def _mask_query_group_scales(
    raw_scale,
    group_offsets,
    seq_first_row,
    query_sequence_end,
    block_length,
    scale_factor: gl.constexpr,
    mask_rows: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    two_heads: gl.constexpr,
):
    """Keep positive denominators while zeroing stored Q32 scales for empty groups."""
    if not mask_rows:
        return raw_scale, raw_scale * scale_factor
    group_valid = (seq_first_row + group_offsets * _GL_QUERY_SCALE_ROWS) < query_sequence_end
    if mask_block_lengths:
        group_valid = group_valid & (group_offsets * _GL_QUERY_SCALE_ROWS < block_length)
    valid_mask = group_valid[:, None] if two_heads else group_valid
    return gl.where(valid_mask, raw_scale, 1.0), gl.where(valid_mask, raw_scale * scale_factor, 0.0)


@gluon.jit
def _store_query_half(
    values,
    cos_ptr,
    sin_ptr,
    query_base_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    batch_head_index,
    storage_tile,
    global_tile,
    storage_tiles,
    seq_first_row,
    query_sequence_end,
    logical_sequence_length,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    softmax_scale: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    values, row_valid, block_length = _rope_and_mask_half(
        values,
        cos_ptr,
        sin_ptr,
        seq_first_row,
        query_sequence_end,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mask_rows,
        block_lengths_ptr,
        global_tile,
        mask_block_lengths,
    )
    if mean_pool_summary:
        valid_count = (
            gl.sum(row_valid.to(gl.int32), axis=0).to(gl.float32)
            if mask_rows
            else _GL_TILE_ROWS_F32
        )
        summary = _reduce_column_mean(values, valid_count)
    else:
        column_max, column_min = _reduce_column_minmax(values, row_valid, mask_rows)
        summary = column_max + column_min
    _store_tile_summary(
        query_summary_ptr, summary, batch_head_index, storage_tile, storage_tiles, head_dim
    )

    column_offsets = gl.arange(0, 128, gl.SliceLayout(0, _TILE_HALF_LAYOUT))
    values = values * _compute_hadamard_signs(column_offsets % head_dim)[None, :]
    rotated = _apply_hadamard_64(gl.reshape(values, [64, 2, 64]))

    scale_factor: gl.constexpr = softmax_scale * _GL_LOG2_E
    batch_head_64 = batch_head_index.to(gl.int64)
    group_stride = storage_tiles.to(gl.int64) * 2
    storage_rows = storage_tiles * _GL_TILE_ROWS

    if head_dim == 64:
        grouped = gl.reshape(rotated * _GL_D64_NORM, [2, _GL_QUERY_SCALE_ROWS, 2, 64])
        raw_scale = _compute_int8_scale(gl.max(gl.max(gl.abs(grouped), axis=3), axis=1))
        group_offsets = gl.arange(0, 2, gl.SliceLayout(1, raw_scale.type.layout))
        head_offsets = gl.arange(0, 2, gl.SliceLayout(0, raw_scale.type.layout))
        raw_scale, stored_scale = _mask_query_group_scales(
            raw_scale,
            group_offsets,
            seq_first_row,
            query_sequence_end,
            block_length,
            scale_factor,
            mask_rows=mask_rows,
            mask_block_lengths=mask_block_lengths,
            two_heads=True,
        )
        query_codes = gl.reshape(_round_to_int8(grouped / raw_scale[:, None, :, None]), [64, 128])
        _store_head_major_i8(query_base_ptr, query_codes, storage_tile, storage_rows, head_dim)
        query_scale_offsets = (
            batch_head_64 + head_offsets[None, :].to(gl.int64)
        ) * group_stride + (storage_tile * 2 + group_offsets[:, None])
        gl.store(query_scale_ptr + query_scale_offsets, stored_scale)
    else:
        grouped = gl.reshape(
            _apply_hadamard_stage64(rotated) * _GL_D128_NORM, [2, _GL_QUERY_SCALE_ROWS, 128]
        )
        raw_scale = _compute_int8_scale(gl.max(gl.max(gl.abs(grouped), axis=2), axis=1))
        group_offsets = gl.arange(0, 2, raw_scale.type.layout)
        raw_scale, stored_scale = _mask_query_group_scales(
            raw_scale,
            group_offsets,
            seq_first_row,
            query_sequence_end,
            block_length,
            scale_factor,
            mask_rows=mask_rows,
            mask_block_lengths=mask_block_lengths,
            two_heads=False,
        )
        query_codes = gl.reshape(_round_to_int8(grouped / raw_scale[:, None, None]), [64, 128])
        _store_head_major_i8(query_base_ptr, query_codes, storage_tile, storage_rows, head_dim)
        query_scale_offsets = batch_head_64 * group_stride + (storage_tile * 2 + group_offsets)
        gl.store(query_scale_ptr + query_scale_offsets, stored_scale)


@gluon.jit
def _store_key_half(
    values,
    cos_ptr,
    sin_ptr,
    key_base_ptr,
    key_scale_ptr,
    key_center_base_ptr,
    key_center_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    batch_head_index,
    tile_index,
    storage_tiles,
    seq_first_row,
    logical_sequence_length,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    values, row_valid, _block_length = _rope_and_mask_half(
        values,
        cos_ptr,
        sin_ptr,
        seq_first_row,
        logical_sequence_length,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mask_rows,
        block_lengths_ptr,
        tile_index,
        mask_block_lengths,
    )
    if mean_pool_summary:
        valid_count = (
            gl.sum(row_valid.to(gl.int32), axis=0).to(gl.float32)
            if mask_rows
            else _GL_TILE_ROWS_F32
        )
        key_summary = _reduce_column_mean(values, valid_count)
        _store_tile_summary(
            key_summary_ptr, key_summary, batch_head_index, tile_index, storage_tiles, head_dim
        )
    else:
        column_max, column_min = _reduce_column_minmax(values, row_valid, mask_rows)
        _store_tile_summary(
            key_summary_ptr, column_max, batch_head_index, tile_index, storage_tiles, head_dim
        )
        _store_tile_summary(
            key_aux_ptr, column_min, batch_head_index, tile_index, storage_tiles, head_dim
        )

    column_offsets = gl.arange(0, 128, gl.SliceLayout(0, _TILE_HALF_LAYOUT))
    values = values * _compute_hadamard_signs(column_offsets % head_dim)[None, :]
    hadamard_64 = _apply_hadamard_64(gl.reshape(values, [64, 2, 64]))
    if head_dim == 64:
        rotated = gl.reshape(hadamard_64 * _GL_D64_NORM, [64, 128])
    else:
        rotated = _apply_hadamard_stage64(hadamard_64) * _GL_D128_NORM

    rotated_max, rotated_min = _reduce_column_minmax(rotated, row_valid, mask_rows)
    if mask_rows:
        center = gl.where(rotated_max >= rotated_min, 0.5 * (rotated_max + rotated_min), 0.0)
    else:
        center = 0.5 * (rotated_max + rotated_min)

    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    storage_rows = storage_tiles * _GL_TILE_ROWS

    # Quantize & store the 2-term rotated center in 1-VGPR/thread layout before centered keys.
    if head_dim == 64:
        center_heads = gl.reshape(center, [2, 64])
        center_scale_0 = _compute_int8_scale(gl.max(gl.abs(center_heads), axis=1))
        center_codes_0 = _round_to_int8(center_heads / center_scale_0[:, None])
        residual = center_heads - center_codes_0.to(gl.float32) * center_scale_0[:, None]
        center_scale_1 = _compute_int8_scale(gl.max(gl.abs(residual), axis=1))
        center_codes_1 = _round_to_int8(residual / center_scale_1[:, None])
        center_layout: gl.constexpr = center_codes_0.type.layout
        head_indices = gl.arange(0, 2, gl.SliceLayout(1, center_layout))
        feature_indices = gl.arange(0, 64, gl.SliceLayout(0, center_layout))
        center_offsets = (head_indices[:, None] * storage_tiles_64 + tile_index) * (
            2 * 64
        ) + feature_indices[None, :]
        cdna3.buffer_store(center_codes_0, key_center_base_ptr, center_offsets.to(gl.int32))
        cdna3.buffer_store(center_codes_1, key_center_base_ptr, (center_offsets + 64).to(gl.int32))
        scale_heads = gl.arange(0, 2, center_scale_0.type.layout)
        center_scale_offsets = (
            (batch_head_64 + scale_heads.to(gl.int64)) * storage_tiles_64 + tile_index
        ) * 2
        gl.store(key_center_scale_ptr + center_scale_offsets, center_scale_0)
        gl.store(key_center_scale_ptr + center_scale_offsets + 1, center_scale_1)
    else:
        center_scale_0 = _compute_int8_scale(gl.max(gl.abs(center), axis=0))
        center_codes_0 = _round_to_int8(center / center_scale_0)
        residual = center - center_codes_0.to(gl.float32) * center_scale_0
        center_scale_1 = _compute_int8_scale(gl.max(gl.abs(residual), axis=0))
        center_codes_1 = _round_to_int8(residual / center_scale_1)
        feature_indices = gl.arange(0, 128, center_codes_0.type.layout)
        center_offsets = tile_index * (2 * 128) + feature_indices
        cdna3.buffer_store(center_codes_0, key_center_base_ptr, center_offsets)
        cdna3.buffer_store(center_codes_1, key_center_base_ptr, center_offsets + 128)
        center_scale_offsets = (batch_head_64 * storage_tiles_64 + tile_index) * 2
        gl.store(key_center_scale_ptr + center_scale_offsets, center_scale_0)
        gl.store(key_center_scale_ptr + center_scale_offsets + 1, center_scale_1)

    center_row = gl.convert_layout(center, gl.SliceLayout(0, rotated.type.layout))
    centered = rotated - center_row[None, :]
    if mask_rows:
        row_valid_rot = gl.convert_layout(row_valid, gl.SliceLayout(1, centered.type.layout))
        centered = gl.where(row_valid_rot[:, None], centered, 0.0)
    if head_dim == 64:
        centered_heads = gl.reshape(centered, [64, 2, 64])
        key_scale = _compute_int8_scale(gl.max(gl.max(gl.abs(centered_heads), axis=2), axis=0))
        key_codes = gl.reshape(_round_to_int8(centered_heads / key_scale[None, :, None]), [64, 128])
        _store_head_major_i8(key_base_ptr, key_codes, tile_index, storage_rows, head_dim)
        scale_heads = gl.arange(0, 2, key_scale.type.layout)
        gl.store(
            key_scale_ptr
            + (batch_head_64 + scale_heads.to(gl.int64)) * storage_tiles_64
            + tile_index,
            key_scale,
        )
    else:
        key_scale = _compute_int8_scale(gl.max(gl.max(gl.abs(centered), axis=1), axis=0))
        key_codes = _round_to_int8(centered / key_scale)
        _store_head_major_i8(key_base_ptr, key_codes, tile_index, storage_rows, head_dim)
        gl.store(key_scale_ptr + batch_head_64 * storage_tiles_64 + tile_index, key_scale)


@gluon.jit
def _store_value_half(
    values,
    value_base_ptr,
    value_scale_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    batch_head_index,
    tile_index,
    storage_tiles,
    seq_first_row,
    logical_sequence_length,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    if mask_rows:
        rows_local = gl.arange(0, 64, gl.SliceLayout(1, _TILE_HALF_LAYOUT))
        row_valid = (seq_first_row + rows_local) < logical_sequence_length
        if mask_block_lengths:
            block_length = gl.load(
                block_lengths_ptr + tile_index,
                mask=seq_first_row < logical_sequence_length,
                other=0,
            )
            row_valid = row_valid & (rows_local < block_length)
        values = gl.where(row_valid[:, None], values, 0.0)
        valid_count = gl.maximum(gl.sum(row_valid.to(gl.int32), axis=0), 1).to(gl.float32)
        mean_bf16 = _reduce_column_mean(values, valid_count).to(gl.bfloat16)
    else:
        row_valid = None
        mean_bf16 = _reduce_column_mean(values, _GL_TILE_ROWS_F32).to(gl.bfloat16)
    _store_tile_summary(
        value_mean_ptr, mean_bf16, batch_head_index, tile_index, storage_tiles, head_dim
    )

    mean_f32 = mean_bf16.to(gl.float32)
    centered = values - gl.convert_layout(mean_f32, gl.SliceLayout(0, _TILE_HALF_LAYOUT))[None, :]
    if mask_rows and row_valid is not None:
        centered = gl.where(row_valid[:, None], centered, 0.0)

    batch_head_64 = batch_head_index.to(gl.int64)
    storage_tiles_64 = storage_tiles.to(gl.int64)
    if head_dim == 64:
        centered_heads = gl.reshape(centered, [64, 2, 64])
        residual_scale = _compute_int8_scale(gl.max(gl.max(gl.abs(centered_heads), axis=2), axis=0))
        mean_magnitude = gl.convert_layout(
            gl.max(gl.abs(gl.reshape(mean_f32, [2, 64])), axis=1), residual_scale.type.layout
        )
        value_scale = gl.maximum(residual_scale, mean_magnitude * _GL_VALUE_MEAN_SCALE_FLOOR)
        value_codes = gl.reshape(
            _round_to_int8(centered_heads / value_scale[None, :, None]), [64, 128]
        )
        scale_heads = gl.arange(0, 2, value_scale.type.layout)
        gl.store(
            value_scale_ptr
            + (batch_head_64 + scale_heads.to(gl.int64)) * storage_tiles_64
            + tile_index,
            value_scale * _GL_P_UINT8_RANGE,
        )
    else:
        residual_scale = _compute_int8_scale(gl.max(gl.max(gl.abs(centered), axis=1), axis=0))
        mean_magnitude = gl.max(gl.abs(mean_f32), axis=0)
        value_scale = gl.maximum(residual_scale, mean_magnitude * _GL_VALUE_MEAN_SCALE_FLOOR)
        value_codes = _round_to_int8(centered / value_scale)
        gl.store(
            value_scale_ptr + batch_head_64 * storage_tiles_64 + tile_index,
            value_scale * _GL_P_UINT8_RANGE,
        )

    # Fold AMD-WMMA token bit3<->bit4 swizzle into the transpose so stores vectorize to b64.
    value_swizzled = gl.reshape(
        gl.permute(gl.reshape(value_codes, [2, 2, 2, 8, 128]), [4, 0, 2, 1, 3]),
        [128, 64],
    )
    value_transposed = gl.convert_layout(value_swizzled, _VALUE_STORE_LAYOUT)
    feature_offsets = gl.arange(0, 128, gl.SliceLayout(1, _VALUE_STORE_LAYOUT))
    token_offsets = gl.arange(0, 64, gl.SliceLayout(0, _VALUE_STORE_LAYOUT))
    if head_dim == 64:
        value_offsets = (
            (
                ((feature_offsets // 64)[:, None] * storage_tiles + tile_index) * 64
                + (feature_offsets % 64)[:, None]
            )
            * _GL_TILE_ROWS
        ) + token_offsets[None, :]
    else:
        value_offsets = (tile_index * 128 + feature_offsets[:, None]) * _GL_TILE_ROWS + (
            token_offsets[None, :]
        )
    cdna3.buffer_store(value_transposed, value_base_ptr, value_offsets)


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_QUERY)
def _query_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    chunk_start,
    query_sequence_end,
    storage_tiles,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    softmax_scale: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a Q window tile and emit Q32 INT8 rows, scales, and Q64 summaries."""
    half_top, half_bot, batch_head_index, batch_head_stride, storage_tile_top, seq_first_row = (
        _project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            norm_weight_ptr,
            row_block_offset,
            chunk_start,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            norm_epsilon,
            True,
            mask_rows,
            group_m,
        )
    )
    query_base_ptr = query_ptr + batch_head_stride * _GL_TILE_ROWS * head_dim
    chunk_tile_base = chunk_start // _GL_TILE_ROWS
    _store_query_half(
        half_top,
        cos_ptr,
        sin_ptr,
        query_base_ptr,
        query_scale_ptr,
        query_summary_ptr,
        block_lengths_ptr,
        batch_head_index,
        storage_tile_top,
        chunk_tile_base + storage_tile_top,
        storage_tiles,
        seq_first_row,
        query_sequence_end,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        softmax_scale,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
    )
    storage_tile_bot = storage_tile_top + 1
    if (not mask_rows) or (
        (storage_tile_bot < storage_tiles) and (seq_first_row + _GL_TILE_ROWS < query_sequence_end)
    ):
        _store_query_half(
            half_bot,
            cos_ptr,
            sin_ptr,
            query_base_ptr,
            query_scale_ptr,
            query_summary_ptr,
            block_lengths_ptr,
            batch_head_index,
            storage_tile_bot,
            chunk_tile_base + storage_tile_bot,
            storage_tiles,
            seq_first_row + _GL_TILE_ROWS,
            query_sequence_end,
            logical_sequence_length,
            head_dim,
            rotary_dim,
            softmax_scale,
            mean_pool_summary,
            mask_block_lengths,
            mask_rows,
        )


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE)
def _key_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    key_ptr,
    key_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a K tile and emit K64-centered INT8 keys, scales, centers, and summaries."""
    half_top, half_bot, batch_head_index, batch_head_stride, tile_top, seq_first_row = (
        _project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            norm_weight_ptr,
            row_block_offset,
            0,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            norm_epsilon,
            True,
            mask_rows,
            group_m,
        )
    )
    key_base_ptr = key_ptr + batch_head_stride * _GL_TILE_ROWS * head_dim
    key_center_base_ptr = key_center_ptr + batch_head_stride * (2 * head_dim)
    _store_key_half(
        half_top,
        cos_ptr,
        sin_ptr,
        key_base_ptr,
        key_scale_ptr,
        key_center_base_ptr,
        key_center_scale_ptr,
        key_summary_ptr,
        key_aux_ptr,
        block_lengths_ptr,
        batch_head_index,
        tile_top,
        storage_tiles,
        seq_first_row,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
    )
    tile_bot = tile_top + 1
    if (not mask_rows) or (
        (tile_bot < storage_tiles) and (seq_first_row + _GL_TILE_ROWS < logical_sequence_length)
    ):
        _store_key_half(
            half_bot,
            cos_ptr,
            sin_ptr,
            key_base_ptr,
            key_scale_ptr,
            key_center_base_ptr,
            key_center_scale_ptr,
            key_summary_ptr,
            key_aux_ptr,
            block_lengths_ptr,
            batch_head_index,
            tile_bot,
            storage_tiles,
            seq_first_row + _GL_TILE_ROWS,
            logical_sequence_length,
            head_dim,
            rotary_dim,
            mean_pool_summary,
            mask_block_lengths,
            mask_rows,
        )


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE)
def _value_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    value_ptr,
    value_scale_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a V tile and emit K64 tile-centered AMD-WMMA INT8 values and BF16 means."""
    half_top, half_bot, batch_head_index, batch_head_stride, tile_top, seq_first_row = (
        _project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            None,
            row_block_offset,
            0,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            1e-5,
            False,
            mask_rows,
            group_m,
        )
    )
    value_base_ptr = value_ptr + batch_head_stride * _GL_TILE_ROWS * head_dim
    _store_value_half(
        half_top,
        value_base_ptr,
        value_scale_ptr,
        value_mean_ptr,
        block_lengths_ptr,
        batch_head_index,
        tile_top,
        storage_tiles,
        seq_first_row,
        logical_sequence_length,
        head_dim,
        mask_block_lengths,
        mask_rows,
    )
    tile_bot = tile_top + 1
    if (not mask_rows) or (
        (tile_bot < storage_tiles) and (seq_first_row + _GL_TILE_ROWS < logical_sequence_length)
    ):
        _store_value_half(
            half_bot,
            value_base_ptr,
            value_scale_ptr,
            value_mean_ptr,
            block_lengths_ptr,
            batch_head_index,
            tile_bot,
            storage_tiles,
            seq_first_row + _GL_TILE_ROWS,
            logical_sequence_length,
            head_dim,
            mask_block_lengths,
            mask_rows,
        )


def project_query(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    chunk_start: int,
    chunk_rows: int,
    out: QueryOutput,
    bias: torch.Tensor | None = None,
    execution_plan: ProjectionExecutionPlan,
) -> None:
    """Fill Q32 INT8 queries, scales, and Q64 summaries via RDNA4 Gluon kernels."""
    query, query_scale, query_summary = out
    batch, heads, storage_sequence_length, head_dim = query.shape
    if batch == 0 or chunk_rows == 0:
        return
    mask_block_lengths = block_lengths is not None
    col_blocks = (heads * head_dim) // 128
    options = _build_kernel_options()

    with device_context(input_qdata.device):
        for row_blocks, row_block_offset, aligned_rows in resolve_row_block_launches(
            chunk_rows, GLUON_BLOCK_M
        ):
            _query_kernel[(row_blocks * col_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                norm_weight,
                cos,
                sin,
                query,
                query_scale,
                query_summary,
                block_lengths if mask_block_lengths else query_scale,
                input_qdata.shape[1],
                row_block_offset,
                chunk_start,
                chunk_start + chunk_rows,
                storage_sequence_length // TILE_ROWS,
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=mask_block_lengths,
                mask_rows=(not aligned_rows) or mask_block_lengths,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                **options,
            )


def project_key(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    out: KeyOutput,
    bias: torch.Tensor | None = None,
    execution_plan: ProjectionExecutionPlan,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    """Fill K64-centered INT8 keys, scales, 2-term centers, and summaries via RDNA4 Gluon."""
    key, key_scale, key_center, key_center_scale, key_summary, key_aux = out
    batch, heads, storage_sequence_length, head_dim = key.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0 or sequence_length == 0:
        return
    mask_block_lengths = block_lengths is not None
    col_blocks = (heads * head_dim) // 128
    rotary_dim = cos.shape[1]
    mean_pool_summary = routing_mode == _MEAN_ROUTING

    with device_context(input_qdata.device):
        for row_blocks, row_block_offset, aligned_rows in resolve_row_block_launches(
            sequence_length,
            GLUON_BLOCK_M,
            start_row_block=start_row_block,
            num_row_blocks=num_row_blocks,
        ):
            mask_rows = (not aligned_rows) or mask_block_lengths
            options = _build_key_kernel_options(
                head_dim,
                rotary_dim,
                mask_rows=mask_rows,
                mask_block_lengths=mask_block_lengths,
                mean_pool_summary=mean_pool_summary,
            )
            _key_kernel[(row_blocks * col_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                norm_weight,
                cos,
                sin,
                key,
                key_scale,
                key_center,
                key_center_scale,
                key_summary,
                key_aux,
                block_lengths if mask_block_lengths else key_scale,
                sequence_length,
                row_block_offset,
                storage_sequence_length // TILE_ROWS,
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                norm_epsilon=norm_epsilon,
                mean_pool_summary=mean_pool_summary,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                **options,
            )


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    value_format: ValueFormat,
    out: ValueOutput,
    bias: torch.Tensor | None = None,
    execution_plan: ProjectionExecutionPlan,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    """Fill K64 tile-centered INT8 values, multipliers, and BF16 means via RDNA4 Gluon."""
    if value_format is not ValueFormat.AMD_WMMA:
        raise ValueError("RDNA4 Gluon V projection requires ValueFormat.AMD_WMMA storage")
    value, value_scale_multiplier, value_mean = out
    batch, heads, storage_tiles, head_dim = value_mean.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0 or sequence_length == 0:
        return
    mask_block_lengths = block_lengths is not None
    col_blocks = (heads * head_dim) // 128
    options = _build_kernel_options()

    with device_context(input_qdata.device):
        for row_blocks, row_block_offset, aligned_rows in resolve_row_block_launches(
            sequence_length,
            GLUON_BLOCK_M,
            start_row_block=start_row_block,
            num_row_blocks=num_row_blocks,
        ):
            _value_kernel[(row_blocks * col_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                value,
                value_scale_multiplier,
                value_mean,
                block_lengths if mask_block_lengths else value_scale_multiplier,
                sequence_length,
                row_block_offset,
                storage_tiles,
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                mask_block_lengths=mask_block_lengths,
                mask_rows=(not aligned_rows) or mask_block_lengths,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                **options,
            )
