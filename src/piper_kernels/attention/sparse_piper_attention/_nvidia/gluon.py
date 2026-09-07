"""Paired-K128 Gluon kernel for sparse Piper Attention on SM120."""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS

from .._prepared import _PreparedSparsePiperAttention

_BLOCK_M = 64
_BLOCK_N = 64
_HEAD_DIM = 128
_LOG2_255 = 7.994353436858858

_GL_BLOCK_M = gl.constexpr(_BLOCK_M)
_GL_BLOCK_N = gl.constexpr(_BLOCK_N)
_GL_HEAD_DIM = gl.constexpr(_HEAD_DIM)
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)
_GL_LOG2_255 = gl.constexpr(_LOG2_255)
_GL_VALUE_LOG_BOUND_CORRECTION = gl.constexpr(0.086085)


@gluon.jit
def _uint8_int8_mma(lhs, rhs, accumulator):
    gl.static_assert(lhs.dtype == gl.uint8, "lhs must be UINT8")
    gl.static_assert(rhs.dtype == gl.int8, "rhs must be INT8")
    lhs_bits = lhs.to(gl.int8, bitcast=True)
    result = mma_v2(lhs_bits, rhs, accumulator)
    return gl.inline_asm_elementwise(
        asm="piper_attention_u8s8_dot_marker $0, $1;",
        constraints="=r,r",
        args=[result],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _packed_float32_to_uint8(values):
    return gl.inline_asm_elementwise(
        asm="""
        {
            .reg .s32 a, b, c, d;
            .reg .b32 lo;
            cvt.rzi.s32.f32 a, $1;
            cvt.rzi.s32.f32 b, $2;
            cvt.rzi.s32.f32 c, $3;
            cvt.rzi.s32.f32 d, $4;
            cvt.pack.sat.u8.s32.b32 lo, d, c, 0;
            cvt.pack.sat.u8.s32.b32 $0, b, a, lo;
        }
        """,
        constraints="=r,f,f,f,f",
        args=[values],
        dtype=gl.uint8,
        is_pure=True,
        pack=4,
    )


@gluon.jit
def _fma_fp32(lhs, rhs, addend):
    return gl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[lhs, rhs, addend],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _issue_tma(descriptor, offsets, shared, barrier):
    mbarrier.expect(barrier, descriptor.block_type.nbytes)
    tma.async_copy_global_to_shared(descriptor, offsets, barrier, shared)


@gluon.jit
def _issue_tma_pair(
    descriptor,
    offsets_0,
    offsets_1,
    shared_0,
    shared_1,
    barrier,
):
    mbarrier.expect(barrier, descriptor.block_type.nbytes * 2)
    tma.async_copy_global_to_shared(descriptor, offsets_0, barrier, shared_0)
    tma.async_copy_global_to_shared(descriptor, offsets_1, barrier, shared_1)


@gluon.jit
def _piper_probability_pair(
    query,
    key_shared_pair,
    query_scale,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    block_lengths_ptr,
    denominator,
    running_max,
    batch_head,
    start_n_0,
    start_n_1,
    has_second,
    sequence_tiles,
    logical_sequence_length,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
    probability_layout: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    mask_duplicate: gl.constexpr,
):
    """Advance one shared Piper coordinate over two independently scaled K64 tiles."""
    key = key_shared_pair.permute([1, 0]).load(key_layout)
    integer_scores = mma_v2(
        query,
        key,
        gl.zeros([_GL_BLOCK_M, 2 * _GL_BLOCK_N], gl.int32, mma_layout),
    )
    integer_scores_0, integer_scores_1 = gl.split(
        gl.permute(
            gl.reshape(integer_scores, [_GL_BLOCK_M, 2, _GL_BLOCK_N]),
            [0, 2, 1],
        )
    )
    integer_scores_0 = gl.convert_layout(integer_scores_0, mma_layout)
    integer_scores_1 = gl.convert_layout(integer_scores_1, mma_layout)
    key_scale_0 = gl.load(key_scale_ptr + batch_head * sequence_tiles + start_n_0 // _GL_BLOCK_N)
    key_scale_1 = gl.load(key_scale_ptr + batch_head * sequence_tiles + start_n_1 // _GL_BLOCK_N)
    score_scale_0 = query_scale * key_scale_0
    score_scale_1 = query_scale * key_scale_1
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    offsets_n = gl.arange(0, _GL_BLOCK_N, column_layout)
    valid_keys_0 = gl.full([_GL_BLOCK_N], True, gl.int1, column_layout)
    valid_keys_1 = valid_keys_0
    if mask_block_lengths:
        block_length_0 = gl.load(block_lengths_ptr + start_n_0 // _GL_BLOCK_N)
        block_length_1 = gl.load(block_lengths_ptr + start_n_1 // _GL_BLOCK_N)
        valid_keys_0 = offsets_n < block_length_0
        valid_keys_1 = offsets_n < block_length_1
    elif mask_ragged_tail:
        valid_keys_0 = start_n_0 + offsets_n < logical_sequence_length
        valid_keys_1 = start_n_1 + offsets_n < logical_sequence_length
    if mask_block_lengths or mask_ragged_tail:
        if mask_duplicate:
            valid_keys_1 &= has_second
        integer_scores_0 = gl.where(valid_keys_0[None, :], integer_scores_0, -2147483648)
        integer_scores_1 = gl.where(valid_keys_1[None, :], integer_scores_1, -2147483648)
    # A duplicate second tile is entirely invalid, so its maximum is
    # replaced by -inf below without masking all of its integer scores.

    # Q/K scales are nonnegative. Their FP32 conversion and multiplication
    # are monotonic, so the row maximum can be reduced exactly in INT32
    # before scaling, keeping both full FP32 score tiles out of this stage.
    score_max_0 = gl.max(integer_scores_0, axis=1).to(gl.float32) * score_scale_0
    score_max_1 = gl.max(integer_scores_1, axis=1).to(gl.float32) * score_scale_1
    # Every selected physical K64 has at least one valid key; only the
    # duplicated final tile can have no active keys in this kernel's contract.
    if mask_duplicate:
        score_max_1 = gl.where(has_second, score_max_1, -float("inf"))

    value_scale_multiplier_0 = gl.load(
        value_scale_multiplier_ptr + batch_head * sequence_tiles + start_n_0 // _GL_BLOCK_N
    ).to(gl.float32)
    value_scale_multiplier_1 = gl.load(
        value_scale_multiplier_ptr + batch_head * sequence_tiles + start_n_1 // _GL_BLOCK_N
    ).to(gl.float32)
    multiplier_bits_0 = value_scale_multiplier_0.to(gl.int32, bitcast=True)
    multiplier_bits_1 = value_scale_multiplier_1.to(gl.int32, bitcast=True)
    value_log_scale_0 = multiplier_bits_0.to(gl.float32) * (1.0 / 8388608.0) - (
        127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION
    )
    value_log_scale_1 = multiplier_bits_1.to(gl.float32) * (1.0 / 8388608.0) - (
        127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION
    )
    block_max = gl.maximum(
        score_max_0 + value_log_scale_0,
        score_max_1 + value_log_scale_1,
    )
    next_max = gl.maximum(running_max, block_max)
    old_weight = gl.exp2(running_max - next_max)
    current_weight = gl.exp2(block_max - next_max)
    scores_0 = integer_scores_0.to(gl.float32) * score_scale_0[:, None]
    scores_1 = integer_scores_1.to(gl.float32) * score_scale_1[:, None]
    # Join the independent K64 coordinates along a register dimension before
    # probability work, retaining the original shared coordinate and K64 scales.
    scores = gl.reshape(
        gl.permute(gl.join(scores_0, scores_1), [0, 2, 1]),
        [_GL_BLOCK_M, 2 * _GL_BLOCK_N],
    )
    scores = gl.convert_layout(scores, mma_layout)
    paired_columns = gl.arange(0, 2 * _GL_BLOCK_N, column_layout)
    value_scale_multiplier = gl.where(
        paired_columns < _GL_BLOCK_N,
        value_scale_multiplier_0,
        value_scale_multiplier_1,
    )
    # Mask the final exponent argument, permitting FMA for valid score shifts.
    shifted_scores = scores - block_max[:, None]
    if mask_block_lengths or mask_ragged_tail:
        valid_keys = gl.reshape(
            gl.permute(gl.join(valid_keys_0, valid_keys_1), [1, 0]), [2 * _GL_BLOCK_N]
        )
        valid_keys = gl.convert_layout(valid_keys, column_layout)
        shifted_scores = gl.where(valid_keys[None, :], shifted_scores, -float("inf"))
    elif mask_duplicate:
        valid_keys = (paired_columns < _GL_BLOCK_N) | has_second
        shifted_scores = gl.where(valid_keys[None, :], shifted_scores, -float("inf"))
    probabilities = gl.exp2(shifted_scores)
    probability_uint8 = _packed_float32_to_uint8(
        probabilities * value_scale_multiplier[None, :] + 0.5
    )
    probability_uint8 = gl.convert_layout(probability_uint8, probability_layout)
    probability_sum = gl.sum(probabilities, axis=1)
    denominator = denominator * old_weight + probability_sum * current_weight
    return (
        probability_uint8,
        denominator,
        next_max,
        old_weight,
        current_weight,
    )


@gluon.jit
def _rescale_packed(partial, accumulator, old_weight, current_weight):
    """Update the FP32 numerator, skipping rescaling when both row weights are one.

    Requires M64/D128 MMA[4,1] register order A,A,B,B repeated within each
    32-element pack. Checking elements 0 and 2 covers both rows. No MMA
    instruction or collective synchronization occurs inside the branch.
    """
    return gl.inline_asm_elementwise(
        asm="""
            {
            .reg .pred keep_a, keep_b, keep_pair;
            .reg .f32 product;
            setp.eq.f32 keep_a, $96, 0f3f800000;
            setp.eq.f32 keep_b, $98, 0f3f800000;
            and.pred keep_pair, keep_a, keep_b;
            @keep_pair bra PIPER_RESCALE_DONE;
            mul.rn.f32 $0, $0, $96;
            mul.rn.f32 $1, $1, $97;
            mul.rn.f32 $2, $2, $98;
            mul.rn.f32 $3, $3, $99;
            mul.rn.f32 $4, $4, $100;
            mul.rn.f32 $5, $5, $101;
            mul.rn.f32 $6, $6, $102;
            mul.rn.f32 $7, $7, $103;
            mul.rn.f32 $8, $8, $104;
            mul.rn.f32 $9, $9, $105;
            mul.rn.f32 $10, $10, $106;
            mul.rn.f32 $11, $11, $107;
            mul.rn.f32 $12, $12, $108;
            mul.rn.f32 $13, $13, $109;
            mul.rn.f32 $14, $14, $110;
            mul.rn.f32 $15, $15, $111;
            mul.rn.f32 $16, $16, $112;
            mul.rn.f32 $17, $17, $113;
            mul.rn.f32 $18, $18, $114;
            mul.rn.f32 $19, $19, $115;
            mul.rn.f32 $20, $20, $116;
            mul.rn.f32 $21, $21, $117;
            mul.rn.f32 $22, $22, $118;
            mul.rn.f32 $23, $23, $119;
            mul.rn.f32 $24, $24, $120;
            mul.rn.f32 $25, $25, $121;
            mul.rn.f32 $26, $26, $122;
            mul.rn.f32 $27, $27, $123;
            mul.rn.f32 $28, $28, $124;
            mul.rn.f32 $29, $29, $125;
            mul.rn.f32 $30, $30, $126;
            mul.rn.f32 $31, $31, $127;
            PIPER_RESCALE_DONE:
            cvt.rn.f32.s32 product, $32;
            fma.rn.f32 $0, product, $128, $0;
            cvt.rn.f32.s32 product, $33;
            fma.rn.f32 $1, product, $129, $1;
            cvt.rn.f32.s32 product, $34;
            fma.rn.f32 $2, product, $130, $2;
            cvt.rn.f32.s32 product, $35;
            fma.rn.f32 $3, product, $131, $3;
            cvt.rn.f32.s32 product, $36;
            fma.rn.f32 $4, product, $132, $4;
            cvt.rn.f32.s32 product, $37;
            fma.rn.f32 $5, product, $133, $5;
            cvt.rn.f32.s32 product, $38;
            fma.rn.f32 $6, product, $134, $6;
            cvt.rn.f32.s32 product, $39;
            fma.rn.f32 $7, product, $135, $7;
            cvt.rn.f32.s32 product, $40;
            fma.rn.f32 $8, product, $136, $8;
            cvt.rn.f32.s32 product, $41;
            fma.rn.f32 $9, product, $137, $9;
            cvt.rn.f32.s32 product, $42;
            fma.rn.f32 $10, product, $138, $10;
            cvt.rn.f32.s32 product, $43;
            fma.rn.f32 $11, product, $139, $11;
            cvt.rn.f32.s32 product, $44;
            fma.rn.f32 $12, product, $140, $12;
            cvt.rn.f32.s32 product, $45;
            fma.rn.f32 $13, product, $141, $13;
            cvt.rn.f32.s32 product, $46;
            fma.rn.f32 $14, product, $142, $14;
            cvt.rn.f32.s32 product, $47;
            fma.rn.f32 $15, product, $143, $15;
            cvt.rn.f32.s32 product, $48;
            fma.rn.f32 $16, product, $144, $16;
            cvt.rn.f32.s32 product, $49;
            fma.rn.f32 $17, product, $145, $17;
            cvt.rn.f32.s32 product, $50;
            fma.rn.f32 $18, product, $146, $18;
            cvt.rn.f32.s32 product, $51;
            fma.rn.f32 $19, product, $147, $19;
            cvt.rn.f32.s32 product, $52;
            fma.rn.f32 $20, product, $148, $20;
            cvt.rn.f32.s32 product, $53;
            fma.rn.f32 $21, product, $149, $21;
            cvt.rn.f32.s32 product, $54;
            fma.rn.f32 $22, product, $150, $22;
            cvt.rn.f32.s32 product, $55;
            fma.rn.f32 $23, product, $151, $23;
            cvt.rn.f32.s32 product, $56;
            fma.rn.f32 $24, product, $152, $24;
            cvt.rn.f32.s32 product, $57;
            fma.rn.f32 $25, product, $153, $25;
            cvt.rn.f32.s32 product, $58;
            fma.rn.f32 $26, product, $154, $26;
            cvt.rn.f32.s32 product, $59;
            fma.rn.f32 $27, product, $155, $27;
            cvt.rn.f32.s32 product, $60;
            fma.rn.f32 $28, product, $156, $28;
            cvt.rn.f32.s32 product, $61;
            fma.rn.f32 $29, product, $157, $29;
            cvt.rn.f32.s32 product, $62;
            fma.rn.f32 $30, product, $158, $30;
            cvt.rn.f32.s32 product, $63;
            fma.rn.f32 $31, product, $159, $31;
            }
        """,
        constraints=(
            # $0..31: FP32 outputs, written before all inputs are consumed.
            "=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,"
            "=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,"
            # $32..63: INT32 PV products.
            "r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,"
            # $64..95: FP32 accumulator inputs tied to the output registers.
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,"
            # $96..127: old row weights (elements 0 and 2 are $96 and $98).
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,"
            # $128..159: current row weights.
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f"
        ),
        args=[partial, accumulator, old_weight[:, None], current_weight[:, None]],
        dtype=gl.float32,
        is_pure=True,
        pack=32,
    )


@gluon.jit
def _piper_pv_pair(
    probability_uint8,
    value_shared_pair,
    accumulator,
    old_weight,
    current_weight,
    mma_layout: gl.constexpr,
    value_layout: gl.constexpr,
):
    """Accumulate two K64 PV tiles in INT32 and update the FP32 numerator."""
    value_0 = value_shared_pair.index(0).permute([1, 0]).load(value_layout)
    value_1 = value_shared_pair.index(1).permute([1, 0]).load(value_layout)
    value = gl.reshape(
        gl.permute(gl.join(value_0, value_1), [2, 0, 1]),
        [2 * _GL_BLOCK_N, _GL_HEAD_DIM],
    )
    value = gl.convert_layout(value, value_layout)
    # Each product sum is bounded by 128 * 255 * 128 = 4,177,920 in magnitude:
    # safe in INT32 and exactly representable when converted to FP32.
    partial = _uint8_int8_mma(
        probability_uint8,
        value,
        gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.int32, mma_layout),
    )
    return _rescale_packed(partial, accumulator, old_weight, current_weight)


@gluon.jit
def _native_tile_start(
    route_base,
    tile_position,
    routed_sparse_tile_count,
    selected_sparse_tile_count,
    sparse_key_blocks,
    stride_rr,
    use_sparse_routes,
):
    safe_route_position = gl.minimum(tile_position, routed_sparse_tile_count - 1)
    route = gl.load(route_base + safe_route_position * stride_rr).to(gl.int32)
    sparse_tile = gl.where(use_sparse_routes, route, tile_position)
    sparse_start = sparse_tile * _GL_BLOCK_N
    dense_start = (
        sparse_key_blocks * _GL_BLOCK_N + (tile_position - selected_sparse_tile_count) * _GL_BLOCK_N
    )
    return gl.where(tile_position < selected_sparse_tile_count, sparse_start, dense_start)


@gluon.jit(
    do_not_specialize=[
        "logical_sequence_length",
        "query_block_offset",
        "global_query_block_offset",
        "sparse_key_blocks",
        "sparse_query_blocks",
        "stride_rb",
        "stride_rq",
    ]
)
def _sparse_piper_attention_kernel(
    query_desc,
    key_desc,
    value_desc,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_mean_ptr,
    coarse_output_ptr,
    coarse_gate_ptr,
    block_lengths_ptr,
    routes_ptr,
    head_keep_blocks_ptr,
    route_head_offsets_ptr,
    output_ptr,
    query_block_offset,
    global_query_block_offset,
    query_storage_sequence_length,
    storage_sequence_length,
    logical_sequence_length,
    sparse_key_blocks,
    sparse_query_blocks,
    stride_rb,
    stride_rq,
    stride_rr,
    stride_ob,
    stride_oh,
    stride_on,
    stride_cb,
    stride_ch,
    stride_cq,
    stride_gb,
    stride_gh,
    stride_gn,
    heads,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
    ragged_tail_is_routed: gl.constexpr,
):
    """Pair native logical K64 tiles in one shared Piper probability coordinate."""
    local_query_block = gl.program_id(0)
    query_block = query_block_offset + local_query_block
    global_query_block = global_query_block_offset + local_query_block
    head = gl.program_id(1)
    batch = gl.program_id(2)
    batch_head = batch * heads + head
    start_m = query_block * _GL_BLOCK_M
    output_start_m = local_query_block * _GL_BLOCK_M
    route_head_offset = gl.load(route_head_offsets_ptr + head)
    route_base = (
        routes_ptr + batch * stride_rb + query_block * stride_rq + route_head_offset * stride_rr
    )

    # _rescale_packed relies on this four-warp MMA register layout.
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 8],
    )
    query_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    key_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)

    query_shared = gl.allocate_shared_memory(
        query_desc.dtype, [_GL_BLOCK_M, _GL_HEAD_DIM], query_desc.layout
    )
    key_shared_pair = gl.allocate_shared_memory(
        key_desc.dtype, [2 * _GL_BLOCK_N, _GL_HEAD_DIM], key_desc.layout
    )
    key_shared_0 = key_shared_pair.slice(0, _GL_BLOCK_N)
    key_shared_1 = key_shared_pair.slice(_GL_BLOCK_N, _GL_BLOCK_N)
    value_shared_pair = gl.allocate_shared_memory(
        value_desc.dtype, [2, _GL_HEAD_DIM, _GL_BLOCK_N], value_desc.layout
    )
    value_shared_0 = value_shared_pair.index(0)
    value_shared_1 = value_shared_pair.index(1)
    query_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    key_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    value_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(query_barrier, count=1)
    mbarrier.init(key_barrier, count=1)
    mbarrier.init(value_barrier, count=1)
    gl.barrier()

    routed_sparse_tile_count = gl.load(head_keep_blocks_ptr + head)
    if has_dense_query_suffix:
        use_sparse_routes = global_query_block < sparse_query_blocks
        selected_sparse_tile_count = gl.where(
            use_sparse_routes,
            routed_sparse_tile_count,
            sparse_key_blocks,
        )
    else:
        use_sparse_routes = True
        selected_sparse_tile_count = routed_sparse_tile_count
    sequence_tiles = storage_sequence_length // _GL_BLOCK_N
    dense_tile_count = sequence_tiles - sparse_key_blocks
    tile_count = selected_sparse_tile_count + dense_tile_count
    pair_count = gl.cdiv(tile_count, 2)
    initial_position_1 = gl.minimum(1, tile_count - 1)
    initial_n_0 = _native_tile_start(
        route_base,
        0,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
    )
    initial_n_1 = _native_tile_start(
        route_base,
        initial_position_1,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
    )

    _issue_tma(
        query_desc,
        [batch_head * query_storage_sequence_length + start_m, 0],
        query_shared,
        query_barrier,
    )
    _issue_tma_pair(
        key_desc,
        [batch_head * storage_sequence_length + initial_n_0, 0],
        [batch_head * storage_sequence_length + initial_n_1, 0],
        key_shared_0,
        key_shared_1,
        key_barrier,
    )
    _issue_tma_pair(
        value_desc,
        [batch_head * _GL_HEAD_DIM, initial_n_0],
        [batch_head * _GL_HEAD_DIM, initial_n_1],
        value_shared_0,
        value_shared_1,
        value_barrier,
    )
    mbarrier.wait(query_barrier, phase=0)
    query = query_shared.load(query_layout)

    offsets_m = gl.arange(0, _GL_BLOCK_M, row_layout)
    query_scale_stride = query_storage_sequence_length // _GL_QUERY_SCALE_ROWS
    query_scale = gl.load(
        query_scale_ptr
        + batch_head * query_scale_stride
        + (start_m + offsets_m) // _GL_QUERY_SCALE_ROWS
    )
    accumulator = gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.float32, mma_layout)
    denominator = gl.zeros([_GL_BLOCK_M], gl.float32, row_layout)
    running_max = gl.full([_GL_BLOCK_M], -float("inf"), gl.float32, row_layout)
    start_n_0 = initial_n_0
    start_n_1 = initial_n_1

    for pair_index in range(pair_count - 1):
        phase = pair_index & 1
        tile_position_0 = pair_index * 2
        mbarrier.wait(key_barrier, phase=phase)
        (
            probability,
            denominator,
            running_max,
            old_weight,
            current_weight,
        ) = _piper_probability_pair(
            query,
            key_shared_pair,
            query_scale,
            key_scale_ptr,
            value_scale_multiplier_ptr,
            block_lengths_ptr,
            denominator,
            running_max,
            batch_head,
            start_n_0,
            start_n_1,
            True,
            sequence_tiles,
            logical_sequence_length,
            mma_layout,
            key_layout,
            probability_layout,
            mask_block_lengths,
            mask_ragged_tail and ragged_tail_is_routed,
            False,
        )

        next_position_0 = tile_position_0 + 2
        next_position_1 = gl.minimum(next_position_0 + 1, tile_count - 1)
        next_n_0 = _native_tile_start(
            route_base,
            next_position_0,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
        )
        next_n_1 = _native_tile_start(
            route_base,
            next_position_1,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
        )
        gl.barrier()
        _issue_tma_pair(
            key_desc,
            [batch_head * storage_sequence_length + next_n_0, 0],
            [batch_head * storage_sequence_length + next_n_1, 0],
            key_shared_0,
            key_shared_1,
            key_barrier,
        )
        mbarrier.wait(value_barrier, phase=phase)
        accumulator = _piper_pv_pair(
            probability,
            value_shared_pair,
            accumulator,
            old_weight,
            current_weight,
            mma_layout,
            value_layout,
        )
        gl.barrier()
        _issue_tma_pair(
            value_desc,
            [batch_head * _GL_HEAD_DIM, next_n_0],
            [batch_head * _GL_HEAD_DIM, next_n_1],
            value_shared_0,
            value_shared_1,
            value_barrier,
        )
        start_n_0 = next_n_0
        start_n_1 = next_n_1

    final_pair = pair_count - 1
    final_phase = final_pair & 1
    final_position_0 = final_pair * 2
    has_second = final_position_0 + 1 < tile_count
    mbarrier.wait(key_barrier, phase=final_phase)
    (
        probability,
        denominator,
        running_max,
        old_weight,
        current_weight,
    ) = _piper_probability_pair(
        query,
        key_shared_pair,
        query_scale,
        key_scale_ptr,
        value_scale_multiplier_ptr,
        block_lengths_ptr,
        denominator,
        running_max,
        batch_head,
        start_n_0,
        start_n_1,
        has_second,
        sequence_tiles,
        logical_sequence_length,
        mma_layout,
        key_layout,
        probability_layout,
        mask_block_lengths,
        mask_ragged_tail,
        True,
    )
    mbarrier.wait(value_barrier, phase=final_phase)
    accumulator = _piper_pv_pair(
        probability,
        value_shared_pair,
        accumulator,
        old_weight,
        current_weight,
        mma_layout,
        value_layout,
    )

    offsets_d = gl.arange(0, _GL_HEAD_DIM, column_layout)
    output = accumulator / (gl.maximum(denominator, 1e-30) * 255.0)[:, None]
    value_mean = gl.load(value_mean_ptr + batch_head * _GL_HEAD_DIM + offsets_d).to(gl.float32)
    output += value_mean[None, :]
    valid_queries = global_query_block * _GL_BLOCK_M + offsets_m < logical_sequence_length
    if apply_coarse_residual:
        coarse = gl.load(
            coarse_output_ptr
            + batch * stride_cb
            + head * stride_ch
            + query_block * stride_cq
            + offsets_d
        ).to(gl.float32)
        gate_offsets = (
            batch * stride_gb
            + head * stride_gh
            + (output_start_m + offsets_m[:, None]) * stride_gn
            + offsets_d[None, :]
        )
        if mask_ragged_tail:
            gate = gl.load(
                coarse_gate_ptr + gate_offsets,
                mask=valid_queries[:, None],
                other=0.0,
            ).to(gl.float32)
        else:
            gate = gl.load(coarse_gate_ptr + gate_offsets).to(gl.float32)
        # Round once at the output store, after combining both FP32 terms.
        output = _fma_fp32(gate, coarse[None, :], output)
    output_offsets = (
        batch * stride_ob
        + head * stride_oh
        + (output_start_m + offsets_m[:, None]) * stride_on
        + offsets_d[None, :]
    )
    if mask_ragged_tail:
        gl.store(
            output_ptr + output_offsets,
            output.to(gl.bfloat16),
            mask=valid_queries[:, None],
        )
    else:
        gl.store(output_ptr + output_offsets, output.to(gl.bfloat16))

    # Every warp must finish its final waits before any warp invalidates the
    # shared barriers.
    gl.barrier()
    mbarrier.invalidate(query_barrier)
    mbarrier.invalidate(key_barrier)
    mbarrier.invalidate(value_barrier)


def _make_gluon_descriptors(
    prepared: _PreparedSparsePiperAttention,
) -> tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor]:
    query = prepared.query.data
    key = prepared.context.key
    value = prepared.context.value
    query_layout = gl.NVMMASharedLayout.get_default_for([_BLOCK_M, _HEAD_DIM], gl.int8)
    key_layout = gl.NVMMASharedLayout.get_default_for([_BLOCK_N, _HEAD_DIM], gl.int8)
    value_layout = gl.NVMMASharedLayout.get_default_for([_HEAD_DIM, _BLOCK_N], gl.int8)
    batch_heads = int(query.shape[0] * query.shape[1])
    query_storage_sequence_length = int(query.shape[2])
    storage_sequence_length = int(key.shape[2])
    if (
        key.shape[:2] != query.shape[:2]
        or key.shape[3] != query.shape[3]
        or value.shape
        != (
            query.shape[0],
            query.shape[1],
            query.shape[3],
            storage_sequence_length,
        )
    ):
        raise ValueError("paired Gluon routed Piper requires compatible Q and K/V storage")
    with device_context(query.device):
        return (
            TensorDescriptor(
                query,
                [batch_heads * query_storage_sequence_length, _HEAD_DIM],
                [_HEAD_DIM, 1],
                [_BLOCK_M, _HEAD_DIM],
                query_layout,
            ),
            TensorDescriptor(
                key,
                [batch_heads * storage_sequence_length, _HEAD_DIM],
                [_HEAD_DIM, 1],
                [_BLOCK_N, _HEAD_DIM],
                key_layout,
            ),
            TensorDescriptor(
                value,
                [batch_heads * _HEAD_DIM, storage_sequence_length],
                [storage_sequence_length, 1],
                [_HEAD_DIM, _BLOCK_N],
                value_layout,
            ),
        )


def _resolve_query_block_range(
    prepared: _PreparedSparsePiperAttention,
    query_block_offset: int,
    query_block_count: int | None,
) -> tuple[int, int, int]:
    """Validate a local launch range and return its size and global offset."""
    query_state = prepared.query
    stored_query_blocks = query_state.data.shape[2] // _BLOCK_M
    if isinstance(query_block_offset, bool) or not isinstance(query_block_offset, int):
        raise TypeError("sparse Piper query block offset must be an integer")
    if query_block_count is not None and (
        isinstance(query_block_count, bool) or not isinstance(query_block_count, int)
    ):
        raise TypeError("sparse Piper query block count must be an integer or None")
    if not 0 <= query_block_offset < stored_query_blocks:
        raise ValueError("sparse Piper query block offset must fit the prepared query storage")
    resolved_query_block_count = (
        stored_query_blocks - query_block_offset if query_block_count is None else query_block_count
    )
    if (
        resolved_query_block_count < 1
        or query_block_offset + resolved_query_block_count > stored_query_blocks
    ):
        raise ValueError("sparse Piper query block range must fit the prepared query storage")
    return (
        stored_query_blocks,
        resolved_query_block_count,
        query_state.global_block_offset + query_block_offset,
    )


def _launch_sparse_piper_attention(
    prepared: _PreparedSparsePiperAttention,
    output: torch.Tensor,
    *,
    query_block_offset: int = 0,
    query_block_count: int | None = None,
    coarse_output: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
) -> None:
    """Launch one caller-owned query-block range over the complete K/V sequence.

    When present, ``coarse_gate`` contains exactly the local output rows
    covered by this launch. ``coarse_output`` is indexed within the prepared
    query storage, whose ``global_block_offset`` locates it in the sequence.
    """
    query_state = prepared.query
    context = prepared.context
    query = query_state.data
    batch, heads, query_storage_sequence_length, head_dim = query.shape
    logical_sequence_length = context.logical_sequence_length
    storage_sequence_length = context.key.shape[2]
    has_block_lengths = context.block_lengths is not None
    # A compact ragged tile outside the sparse prefix is visited last by the
    # dense suffix. Caller-supplied routes may put it anywhere within the
    # sparse prefix, requiring masks in the ordinary loop as well.
    ragged_tail_is_routed = (
        not has_block_lengths and context.sparse_key_blocks * _BLOCK_N > logical_sequence_length
    )
    if (
        head_dim != _HEAD_DIM
        or query_storage_sequence_length < _BLOCK_M
        or query_storage_sequence_length % _BLOCK_M
        or storage_sequence_length < _BLOCK_N
        or storage_sequence_length % _BLOCK_N
        or (
            not has_block_lengths
            and (logical_sequence_length + _BLOCK_M - 1) // _BLOCK_M * _BLOCK_M
            != storage_sequence_length
        )
    ):
        raise ValueError("paired Gluon routed Piper requires padded M64/D128 storage")
    total_query_blocks = storage_sequence_length // _BLOCK_M
    sparse_query_blocks = context.sparse_query_blocks
    stored_query_blocks, resolved_query_block_count, global_query_block_offset = (
        _resolve_query_block_range(
            prepared,
            query_block_offset,
            query_block_count,
        )
    )
    output_sequence_length = (
        resolved_query_block_count * _BLOCK_M
        if has_block_lengths
        else min(
            resolved_query_block_count * _BLOCK_M,
            logical_sequence_length - global_query_block_offset * _BLOCK_M,
        )
    )
    if (
        output.shape != (batch, heads, output_sequence_length, head_dim)
        or output.dtype is not torch.bfloat16
        or output.device != query.device
        or output.stride(-1) != 1
    ):
        raise ValueError("paired Gluon routed Piper output must match the query block range")
    if context.value_scale_multiplier.shape[-1] != 1:
        raise ValueError("paired Gluon routed Piper requires one folded scale per K64 tile")
    has_coarse_residual = coarse_output is not None or coarse_gate is not None
    if (coarse_output is None) != (coarse_gate is None):
        raise ValueError("coarse output and coarse gate must be supplied together")
    if has_coarse_residual:
        assert coarse_output is not None
        assert coarse_gate is not None
        if (
            coarse_output.shape != (batch, heads, stored_query_blocks, head_dim)
            or coarse_output.dtype is not torch.float32
            or coarse_output.device != query.device
            or coarse_output.stride(-1) != 1
        ):
            raise ValueError("Gluon coarse output must be FP32 [batch,heads,Q64,D128]")
        if (
            coarse_gate.shape != (batch, output_sequence_length, heads, head_dim)
            or coarse_gate.dtype is not torch.bfloat16
            or coarse_gate.device != query.device
            or coarse_gate.stride(-1) != 1
        ):
            raise ValueError("Gluon coarse gate must match the local attention output")
    with device_context(output.device):
        install_uint8_int8_dot_hook()

        query_desc, key_desc, value_desc = _make_gluon_descriptors(prepared)
        routes = query_state.routes
        route_head_offsets = context.route_head_offsets
        stride_rb = routes.stride(0)
        stride_rq = routes.stride(1)
        stride_rr = routes.stride(2)
        coarse_tensor = context.value_mean if coarse_output is None else coarse_output
        gate_tensor = output if coarse_gate is None else coarse_gate
        coarse_strides = (0, 0, 0) if coarse_output is None else coarse_output.stride()[:3]
        gate_strides = (
            (0, 0, 0)
            if coarse_gate is None
            else (
                coarse_gate.stride(0),
                coarse_gate.stride(2),
                coarse_gate.stride(1),
            )
        )
        _sparse_piper_attention_kernel[(resolved_query_block_count, heads, batch)](
            query_desc,
            key_desc,
            value_desc,
            query_state.scale,
            context.key_scale,
            context.value_scale_multiplier,
            context.value_mean,
            coarse_tensor,
            gate_tensor,
            (
                context.block_lengths
                if context.block_lengths is not None
                else context.head_keep_blocks
            ),
            routes,
            context.head_keep_blocks,
            route_head_offsets,
            output,
            query_block_offset,
            global_query_block_offset,
            query_storage_sequence_length,
            storage_sequence_length,
            logical_sequence_length,
            context.sparse_key_blocks,
            total_query_blocks if sparse_query_blocks is None else sparse_query_blocks,
            stride_rb,
            stride_rq,
            stride_rr,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            *coarse_strides,
            *gate_strides,
            heads,
            has_block_lengths,
            not has_block_lengths and logical_sequence_length != storage_sequence_length,
            sparse_query_blocks is not None,
            has_coarse_residual,
            ragged_tail_is_routed,
            num_warps=4,
            num_stages=1,
        )
