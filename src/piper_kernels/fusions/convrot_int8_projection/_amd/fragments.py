"""RDNA4 Gluon fragments shared by the fused ConvRot INT8 dense and sparse Piper projections.

Each workgroup projects a 128-row by 128-column tile (`4x2` wave grid, `16 KB` LDS)
using two K64 stages of bank-conflict-free LDS feeding `v_wmma_i32_16x16x16_iu8`:
one D128 head or two D64 heads per program. Swapping feature bit 4 (`16`) and bit 6
(`64`) during B loads keeps each wave on contiguous 64-column feature halves and places
accumulator halves `[64, 128]` into separate register ranges for zero-cost top/bottom
splitting. Six stages of the signed Hadamard run in registers, stage 8 runs via
`v_permlanex16_b32`, and only D128 stage 64 (and D128 R96/R128 RoPE) exchanges across
the two N-waves via epilogue LDS.
"""

# Gluon exposes low-level signatures and layouts that are not modeled by static type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3, rdna4

from piper_kernels._triton.amd_compiler_workarounds import LDS_ONLY_FENCE_ATTRS
from piper_kernels.attention.kernels.piper._amd.fragments import AMDWMMALayout
from piper_kernels.attention.kernels.piper._amd.quantization import (
    GL_D64_NORM,
    GL_D128_NORM,
    GL_INT8_RANGE,  # noqa: F401 - re-exported for downstream fusion kernels
    GL_LOG2_E,
    GL_P_UINT8_RANGE,  # noqa: F401 - re-exported for downstream fusion kernels
    GL_QUERY_SCALE_ROWS,
    GL_SCALE_EPSILON,  # noqa: F401 - re-exported for downstream fusion kernels
    GL_TILE_ROWS,
    GL_TILE_ROWS_F32,  # noqa: F401 - re-exported for downstream fusion kernels
    GL_VALUE_MEAN_SCALE_FLOOR,  # noqa: F401 - re-exported for downstream fusion kernels
    VALUE_STORE_LAYOUT,
    compute_hadamard_signs,
    compute_int8_scale,
    mask_query_group_scales,
    round_to_int8,
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
TILE_HALF_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        warp_bases=[[16, 0], [32, 0], [0, 64]],
        block_bases=[],
        shape=[64, 128],
    )
)
# Cross-N-warp exchange over [64, 2, 32] (16 KB FP32): moves axis-1 (size 2) into registers.
WARP_EXCHANGE_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 1, 0]],
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 8]],
        warp_bases=[[16, 0, 0], [32, 0, 0], [0, 0, 16]],
        block_bases=[],
        shape=[64, 2, 32],
    )
)
# One-way post-exchange [64, 128] layout (avoids a return LDS trip for D128 Hadamard).
POST_EXCHANGE_HALF_LAYOUT = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 64], [0, 32]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        warp_bases=[[16, 0], [32, 0], [0, 16]],
        block_bases=[],
        shape=[64, 128],
    )
)
GL_BLOCK_M = gl.constexpr(GLUON_BLOCK_M)
GL_BLOCK_K = gl.constexpr(GLUON_BLOCK_K)
_GL_SEED = gl.constexpr(GLUON_ACCUMULATOR_SEED)
_GL_SEED_MIN_K = gl.constexpr(GLUON_ACCUMULATOR_SEED_MIN_K)

DO_NOT_SPECIALIZE = (
    "logical_sequence_length",
    "row_block_offset",
    "storage_tiles",
    "num_row_blocks",
)
DO_NOT_SPECIALIZE_QUERY = (*DO_NOT_SPECIALIZE, "chunk_start", "query_sequence_end")

_BUFFER_OFFSET_LIMIT = 1 << 31


def build_kernel_options(*, waves_per_eu: str | None = None) -> dict[str, object]:
    """Build LLVM function attributes keeping LDS-only fences and optional occupancy hints."""
    attrs: tuple[tuple[str, str], ...] = LDS_ONLY_FENCE_ATTRS
    if waves_per_eu is not None:
        attrs = (("amdgpu-waves-per-eu", waves_per_eu), *attrs)
    return {"llvm_fn_attrs": attrs} if attrs else {}


def operands_aligned(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    *,
    storage_sequence_length: int,
    rotary_dim: int = 0,
    head_dim: int = 128,
    output_element_bytes: int = 1,
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
        and heads_per_program * storage_sequence_length * head_dim * output_element_bytes
        < _BUFFER_OFFSET_LIMIT
    )


@gluon.jit
def resolve_tile_ids(
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
def exchange_lanes_x16(values):
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
def project_int8_tile(
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

    num_k_blocks: gl.constexpr = input_features // GL_BLOCK_K
    for k_block_index in range(1, num_k_blocks):
        input_next = cdna3.buffer_load(input_base_ptr, input_offsets + k_block_index * GL_BLOCK_K)
        weight_next = cdna3.buffer_load(
            weight_base_ptr, weight_offsets + k_block_index * GL_BLOCK_K
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
def split_top_bot_logical(accumulator_wmma):
    """Reinterpret WMMA [128, 128] (col4<->col6 swapped) into two [64, 128] halves."""
    physical = gl.convert_layout(accumulator_wmma, _WMMA_PHYSICAL_LAYOUT)
    reshaped = gl.reshape(physical, [2, 64, 2, 2, 2, 16])
    reordered = gl.permute(reshaped, [1, 4, 3, 2, 5, 0])
    half_top, half_bot = gl.split(gl.reshape(reordered, [64, 128, 2]))
    return gl.convert_layout(half_top, TILE_HALF_LAYOUT), gl.convert_layout(
        half_bot, TILE_HALF_LAYOUT
    )


@gluon.jit
def _split_quarters_16(values):
    """Split [64, 128] along its 16-column chunks inside TILE_HALF_LAYOUT."""
    chunked = gl.reshape(values, [64, 2, 2, 2, 16])
    half_01, half_23 = gl.split(gl.permute(chunked, [0, 1, 3, 4, 2]))
    quarter_0, quarter_1 = gl.split(gl.permute(half_01, [0, 1, 3, 2]))
    quarter_2, quarter_3 = gl.split(gl.permute(half_23, [0, 1, 3, 2]))
    return quarter_0, quarter_1, quarter_2, quarter_3


@gluon.jit
def _join_quarters_16(quarter_0, quarter_1, quarter_2, quarter_3):
    """Reassemble four [64, 2, 16] quarter-chunks into TILE_HALF_LAYOUT."""
    half_01 = gl.permute(gl.join(quarter_0, quarter_1), [0, 1, 3, 2])
    half_23 = gl.permute(gl.join(quarter_2, quarter_3), [0, 1, 3, 2])
    chunked = gl.permute(gl.join(half_01, half_23), [0, 1, 4, 2, 3])
    return gl.convert_layout(gl.reshape(chunked, [64, 128]), TILE_HALF_LAYOUT)


@gluon.jit
def rotate_half(values, head_dim: gl.constexpr, rotary_dim: gl.constexpr):
    """Return split-half RoPE partners ``(-second_half, first_half)`` in ``TILE_HALF_LAYOUT``."""
    if rotary_dim == 64:
        # Both D64 (within each head) and D128 (N-warp 0) hold both 32-col halves in registers.
        half_lo, half_hi = gl.split(gl.permute(gl.reshape(values, [64, 2, 2, 32]), [0, 1, 3, 2]))
        joined = gl.permute(gl.join(-half_hi, half_lo), [0, 1, 3, 2])
        return gl.convert_layout(gl.reshape(joined, [64, 128]), TILE_HALF_LAYOUT)
    if rotary_dim == 32:
        quarter_0, quarter_1, quarter_2, quarter_3 = _split_quarters_16(values)
        return _join_quarters_16(-quarter_1, quarter_0, quarter_2, quarter_3)
    if head_dim == 64 and rotary_dim == 48:
        quarter_0, quarter_1, quarter_2, quarter_3 = _split_quarters_16(values)
        peer_0 = exchange_lanes_x16(quarter_0)
        peer_1 = exchange_lanes_x16(quarter_1)
        peer_2 = exchange_lanes_x16(quarter_2)
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
        exchanged = gl.convert_layout(packed_send, WARP_EXCHANGE_LAYOUT)
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
    exch_0 = gl.convert_layout(chunk_0, WARP_EXCHANGE_LAYOUT)
    lo_0, hi_0 = gl.split(gl.permute(exch_0, [0, 2, 1]))
    rot_chunk_0 = gl.convert_layout(
        gl.permute(gl.join(-hi_0, lo_0), [0, 2, 1]), chunk_0.type.layout
    )
    exch_1 = gl.convert_layout(chunk_1, WARP_EXCHANGE_LAYOUT)
    lo_1, hi_1 = gl.split(gl.permute(exch_1, [0, 2, 1]))
    rot_chunk_1 = gl.convert_layout(
        gl.permute(gl.join(-hi_1, lo_1), [0, 2, 1]), chunk_1.type.layout
    )
    return gl.convert_layout(
        gl.reshape(gl.permute(gl.join(rot_chunk_0, rot_chunk_1), [0, 1, 3, 2]), [64, 128]),
        TILE_HALF_LAYOUT,
    )


@gluon.jit
def apply_hadamard_64(values):
    """Apply 6-stage Walsh-Hadamard to ``[64, 2, 64]`` without LDS."""
    low, high = gl.split(gl.reshape(values, [64, 2, 32, 2]))
    values = gl.reshape(gl.join(low + high, low - high), [64, 2, 64])
    low, high = gl.split(gl.permute(gl.reshape(values, [64, 2, 16, 2, 2]), [0, 1, 2, 4, 3]))
    values = gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 2, 4, 3]), [64, 2, 64])
    low, high = gl.split(gl.permute(gl.reshape(values, [64, 2, 8, 2, 4]), [0, 1, 2, 4, 3]))
    values = gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 2, 4, 3]), [64, 2, 64])
    peer = exchange_lanes_x16(values)
    high_lane = (
        gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(1, values.type.layout))) & 8
    ) != 0
    values = gl.where(high_lane[None, None, :], peer - values, values + peer)
    low, high = gl.split(gl.permute(gl.reshape(values, [64, 2, 2, 2, 16]), [0, 1, 2, 4, 3]))
    values = gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 2, 4, 3]), [64, 2, 64])
    low, high = gl.split(gl.permute(gl.reshape(values, [64, 2, 2, 32]), [0, 1, 3, 2]))
    return gl.reshape(gl.permute(gl.join(low + high, low - high), [0, 1, 3, 2]), [64, 2, 64])


@gluon.jit
def apply_hadamard_stage64(values_64x2x64):
    """Apply stage-64 Hadamard across the two N-warps and stay in ``POST_EXCHANGE_HALF_LAYOUT``."""
    chunk_0, chunk_1 = gl.split(
        gl.permute(gl.reshape(values_64x2x64, [64, 2, 2, 32]), [0, 1, 3, 2])
    )
    exch_0 = gl.convert_layout(chunk_0, WARP_EXCHANGE_LAYOUT)
    low_0, high_0 = gl.split(gl.permute(exch_0, [0, 2, 1]))
    out_0 = gl.permute(gl.join(low_0 + high_0, low_0 - high_0), [0, 2, 1])
    exch_1 = gl.convert_layout(chunk_1, WARP_EXCHANGE_LAYOUT)
    low_1, high_1 = gl.split(gl.permute(exch_1, [0, 2, 1]))
    out_1 = gl.permute(gl.join(low_1 + high_1, low_1 - high_1), [0, 2, 1])
    merged = gl.reshape(gl.permute(gl.join(out_0, out_1), [0, 1, 3, 2]), [64, 128])
    return gl.convert_layout(merged, POST_EXCHANGE_HALF_LAYOUT)


@gluon.jit
def apply_rmsnorm(values, head_dim: gl.constexpr, norm_epsilon: gl.constexpr):
    if head_dim == 64:
        grouped = gl.reshape(values, [64, 2, 64])
        inv_rms = gl.rsqrt(gl.sum(grouped * grouped, axis=2) * (1.0 / 64.0) + norm_epsilon)
        return gl.reshape(grouped * inv_rms[:, :, None], [64, 128])
    inv_rms = gl.rsqrt(gl.sum(values * values, axis=1) * (1.0 / 128.0) + norm_epsilon)
    return values * inv_rms[:, None]


@gluon.jit
def dequantize_halves(
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
    apply_rmsnorm_flag: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Dequantize, add bias, and optionally RMSNorm both halves with disjoint vector lifetimes."""
    rows_local = gl.arange(0, 64, gl.SliceLayout(1, TILE_HALF_LAYOUT))
    column_offsets = gl.arange(0, 128, gl.SliceLayout(0, TILE_HALF_LAYOUT))
    rows_top = seq_first_row + rows_local
    rows_bot = seq_first_row + GL_TILE_ROWS + rows_local
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

    if apply_rmsnorm_flag:
        half_top = apply_rmsnorm(half_top, head_dim, norm_epsilon)
        half_bot = apply_rmsnorm(half_bot, head_dim, norm_epsilon)
        if norm_weight_ptr is not None:
            norm_weight = cdna3.buffer_load(norm_weight_ptr, column_offsets % head_dim).to(
                gl.float32
            )
            half_top = half_top * norm_weight[None, :]
            half_bot = half_bot * norm_weight[None, :]
    return half_top, half_bot


@gluon.jit
def project_dequantized_halves(
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
    apply_rmsnorm_flag: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Run the 128x128 GEMM, split top/bot K64 halves, and return dequantized FP32 tiles."""
    local_row_block, column_block = resolve_tile_ids(group_m, num_row_blocks, heads, head_dim)
    row_block = row_block_offset + local_row_block
    batch = gl.program_id(1)

    heads_per_program: gl.constexpr = 128 // head_dim
    first_col = column_block * 128
    batch_head_index = batch * heads + column_block * heads_per_program
    tile_top = row_block * 2
    seq_first_row = sequence_row_offset + row_block * GL_BLOCK_M

    input_base_ptr = input_ptr + batch.to(gl.int64) * logical_sequence_length * input_features
    input_scale_base_ptr = input_scale_ptr + batch.to(gl.int64) * logical_sequence_length
    weight_base_ptr = weight_ptr + first_col.to(gl.int64) * input_features
    weight_scale_base_ptr = weight_scale_ptr + first_col.to(gl.int64)
    bias_base_ptr = bias_ptr + first_col.to(gl.int64) if bias_ptr is not None else None
    batch_head_stride = batch_head_index.to(gl.int64) * storage_tiles.to(gl.int64)

    accumulator = project_int8_tile(
        input_base_ptr,
        weight_base_ptr,
        seq_first_row,
        logical_sequence_length,
        input_features,
        mask_rows,
    )
    acc_top, acc_bot = split_top_bot_logical(accumulator)
    half_top, half_bot = dequantize_halves(
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
        apply_rmsnorm_flag,
        mask_rows,
    )
    return half_top, half_bot, batch_head_index, batch_head_stride, tile_top, seq_first_row


@gluon.jit
def rope_and_mask_half(
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
    rows_local = gl.arange(0, 64, gl.SliceLayout(1, TILE_HALF_LAYOUT))
    sequence_rows = seq_first_row + rows_local
    if rotary_dim > 0:
        feature_offsets = gl.arange(0, 128, gl.SliceLayout(0, TILE_HALF_LAYOUT)) % head_dim
        rows_clamped = (
            gl.minimum(sequence_rows, logical_sequence_length - 1) if mask_rows else sequence_rows
        )
        rotary_mask = feature_offsets < rotary_dim
        rotated = rotate_half(values, head_dim, rotary_dim)
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
            block_length = GL_TILE_ROWS
        values = gl.where(row_valid[:, None], values, 0.0)
    else:
        block_length = GL_TILE_ROWS
        row_valid = sequence_rows < sequence_end
    return values, row_valid, block_length


@gluon.jit
def store_head_major(
    base_ptr,
    tile,
    tile_index,
    storage_rows,
    head_dim: gl.constexpr,
):
    """Store one [64, 128] half (1 or 2 heads) into ``[heads, storage_rows, head_dim]``."""
    layout: gl.constexpr = tile.type.layout
    rows = gl.arange(0, 64, gl.SliceLayout(1, layout))
    columns = gl.arange(0, 128, gl.SliceLayout(0, layout))
    if head_dim == 64:
        offsets = (
            (columns // 64)[None, :] * storage_rows + (tile_index * GL_TILE_ROWS + rows[:, None])
        ) * 64 + (columns % 64)[None, :]
    else:
        offsets = (tile_index * GL_TILE_ROWS + rows[:, None]) * 128 + columns[None, :]
    cdna3.buffer_store(tile, base_ptr, offsets)


@gluon.jit
def quantize_and_store_query_half(
    values,
    query_base_ptr,
    query_scale_ptr,
    batch_head_index,
    storage_tile,
    storage_tiles,
    seq_first_row,
    query_sequence_end,
    block_length,
    head_dim: gl.constexpr,
    softmax_scale: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Apply signed Hadamard, quantize Q32 groups, and store query codes and scales."""
    column_offsets = gl.arange(0, 128, gl.SliceLayout(0, TILE_HALF_LAYOUT))
    values = values * compute_hadamard_signs(column_offsets % head_dim)[None, :]
    rotated = apply_hadamard_64(gl.reshape(values, [64, 2, 64]))

    scale_factor: gl.constexpr = softmax_scale * GL_LOG2_E
    batch_head_64 = batch_head_index.to(gl.int64)
    group_stride = storage_tiles.to(gl.int64) * 2
    storage_rows = storage_tiles * GL_TILE_ROWS

    if head_dim == 64:
        grouped = gl.reshape(rotated * GL_D64_NORM, [2, GL_QUERY_SCALE_ROWS, 2, 64])
        raw_scale = compute_int8_scale(gl.max(gl.max(gl.abs(grouped), axis=3), axis=1))
        group_offsets = gl.arange(0, 2, gl.SliceLayout(1, raw_scale.type.layout))
        head_offsets = gl.arange(0, 2, gl.SliceLayout(0, raw_scale.type.layout))
        raw_scale, stored_scale = mask_query_group_scales(
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
        query_codes = gl.reshape(round_to_int8(grouped / raw_scale[:, None, :, None]), [64, 128])
        store_head_major(query_base_ptr, query_codes, storage_tile, storage_rows, head_dim)
        query_scale_offsets = (
            batch_head_64 + head_offsets[None, :].to(gl.int64)
        ) * group_stride + (storage_tile * 2 + group_offsets[:, None])
        gl.store(query_scale_ptr + query_scale_offsets, stored_scale)
    else:
        grouped = gl.reshape(
            apply_hadamard_stage64(rotated) * GL_D128_NORM, [2, GL_QUERY_SCALE_ROWS, 128]
        )
        raw_scale = compute_int8_scale(gl.max(gl.max(gl.abs(grouped), axis=2), axis=1))
        group_offsets = gl.arange(0, 2, raw_scale.type.layout)
        raw_scale, stored_scale = mask_query_group_scales(
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
        query_codes = gl.reshape(round_to_int8(grouped / raw_scale[:, None, None]), [64, 128])
        store_head_major(query_base_ptr, query_codes, storage_tile, storage_rows, head_dim)
        query_scale_offsets = batch_head_64 * group_stride + (storage_tile * 2 + group_offsets)
        gl.store(query_scale_ptr + query_scale_offsets, stored_scale)


@gluon.jit
def store_amd_wmma_value_codes(
    value_base_ptr,
    value_codes,
    tile_index,
    storage_tiles,
    head_dim: gl.constexpr,
):
    """Pack and store INT8 value codes in RDNA4 WMMA ``[tile, feature, packed_token]`` order."""
    # Fold AMD-WMMA token bit3<->bit4 swizzle into the transpose so stores vectorize to b64.
    value_swizzled = gl.reshape(
        gl.permute(gl.reshape(value_codes, [2, 2, 2, 8, 128]), [4, 0, 2, 1, 3]),
        [128, 64],
    )
    value_transposed = gl.convert_layout(value_swizzled, VALUE_STORE_LAYOUT)
    feature_offsets = gl.arange(0, 128, gl.SliceLayout(1, VALUE_STORE_LAYOUT))
    token_offsets = gl.arange(0, 64, gl.SliceLayout(0, VALUE_STORE_LAYOUT))
    if head_dim == 64:
        value_offsets = (
            (
                ((feature_offsets // 64)[:, None] * storage_tiles + tile_index) * 64
                + (feature_offsets % 64)[:, None]
            )
            * GL_TILE_ROWS
        ) + token_offsets[None, :]
    else:
        value_offsets = (tile_index * 128 + feature_offsets[:, None]) * GL_TILE_ROWS + (
            token_offsets[None, :]
        )
    cdna3.buffer_store(value_transposed, value_base_ptr, value_offsets)
