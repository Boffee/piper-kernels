"""Paired-K128 recurrence arithmetic shared by the NVIDIA sparse Piper kernels.

Every function here consumes register tensors or shared-memory descriptors that
the calling kernel has already filled, so the SM120 TMA kernel and the SM89
``cp.async`` kernel differ only in how operands reach shared memory. Keeping the
arithmetic in one place preserves one numerical contract across targets:
grouped INT8 Q/K, centered tile-scaled INT8 V, UINT8 probabilities, paired K128
online softmax, and an FP32 numerator and denominator rounded once at the store.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS

_LOG2_255 = 7.994353436858858

_GL_BLOCK_N = gl.constexpr(TILE_ROWS)
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
def piper_probability_pair(
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
    block_m: gl.constexpr = query.shape[0]
    key = key_shared_pair.permute([1, 0]).load(key_layout)
    integer_scores = mma_v2(
        query,
        key,
        gl.zeros([block_m, 2 * _GL_BLOCK_N], gl.int32, mma_layout),
    )
    integer_scores_0, integer_scores_1 = gl.split(
        gl.permute(
            gl.reshape(integer_scores, [block_m, 2, _GL_BLOCK_N]),
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
    score_max_0 = gl.max(integer_scores_0, axis=1).to(gl.float32) * score_scale_0  # pyright: ignore[reportAttributeAccessIssue]
    score_max_1 = gl.max(integer_scores_1, axis=1).to(gl.float32) * score_scale_1  # pyright: ignore[reportAttributeAccessIssue]
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
        [block_m, 2 * _GL_BLOCK_N],
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

    Validated for M64/D64 MMA[2,1] or [4,1], M128/D64 MMA[4,1], and
    M64/D128 MMA[4,1]: register order A,A,B,B repeats within each 32-element
    pack. Checking elements 0 and 2 covers both rows. No MMA instruction or
    collective synchronization occurs inside the branch.
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
def piper_pv_pair(
    probability_uint8,
    value_shared_pair,
    accumulator,
    old_weight,
    current_weight,
    mma_layout: gl.constexpr,
    value_layout: gl.constexpr,
):
    """Accumulate two K64 PV tiles in INT32 and update the FP32 numerator."""
    head_dim: gl.constexpr = accumulator.shape[1]
    block_m: gl.constexpr = accumulator.shape[0]
    value_0 = value_shared_pair.index(0).permute([1, 0]).load(value_layout)
    value_1 = value_shared_pair.index(1).permute([1, 0]).load(value_layout)
    value = gl.reshape(
        gl.permute(gl.join(value_0, value_1), [2, 0, 1]),
        [2 * _GL_BLOCK_N, head_dim],
    )
    value = gl.convert_layout(value, value_layout)
    # Each product sum is bounded by 128 * 255 * 128 = 4,177,920 in magnitude:
    # safe in INT32 and exactly representable when converted to FP32.
    partial = _uint8_int8_mma(
        probability_uint8,
        value,
        gl.zeros([block_m, head_dim], gl.int32, mma_layout),
    )
    return _rescale_packed(partial, accumulator, old_weight, current_weight)


@gluon.jit
def store_attention_output(
    accumulator,
    denominator,
    value_mean_ptr,
    coarse_output_ptr,
    coarse_gate_ptr,
    output_ptr,
    batch,
    head,
    kv_batch_head,
    query_block,
    global_query_block,
    output_start_m,
    offsets_m,
    logical_sequence_length,
    output_sequence_length,
    stride_ob,
    stride_oh,
    stride_on,
    stride_cb,
    stride_ch,
    stride_cq,
    stride_gb,
    stride_gh,
    stride_gn,
    mma_layout: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
    mask_output_tail: gl.constexpr,
):
    """Normalize, restore the V mean, add an optional coarse residual, and store once."""
    block_m: gl.constexpr = accumulator.shape[0]
    head_dim: gl.constexpr = accumulator.shape[1]
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    offsets_d = gl.arange(0, head_dim, column_layout)
    output = accumulator / (gl.maximum(denominator, 1e-30) * 255.0)[:, None]
    value_mean = gl.load(value_mean_ptr + kv_batch_head * head_dim + offsets_d).to(gl.float32)
    output += value_mean[None, :]
    valid_queries = global_query_block * _GL_BLOCK_N + offsets_m < logical_sequence_length
    if block_m == 128:
        valid_queries = output_start_m + offsets_m < output_sequence_length
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
    if mask_output_tail:
        gl.store(
            output_ptr + output_offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=valid_queries[:, None],
        )
    else:
        gl.store(output_ptr + output_offsets, output.to(output_ptr.dtype.element_ty))
