"""Gluon kernel for dense Piper Attention with NVIDIA async copies.

Q, K, and V tiles reach shared memory through Ampere-style ``cp.async`` copies
tracked by commit groups; dispatch currently selects this load pipeline on SM89,
which has no Tensor Memory Accelerator. Each CTA owns 64 or 128 query rows on
four MMA warps and advances dense Piper's K64 recurrence: per-row Q scales,
per-key K scales and V multipliers, UINT8 probability codes, and an FP32
numerator rescaled once per tile.

Only the final K64 tile of a row block carries masks, so ragged lengths reuse
one compiled kernel; padded query rows are computed from zero Q and discarded at
the store. Per-thread Q/K quantization gives each MMA thread exactly one K scale
per tile, and each tile's V multipliers travel with its K copy. The kernel can
also quantize its query tile in the prologue, bit for bit as the Sage per-thread
Q kernel does.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false, reportAttributeAccessIssue=false
# Masks exist only in the masked constexpr branch that also reads them.
# pyright: reportPossiblyUnboundVariable=false

from __future__ import annotations

from typing import TYPE_CHECKING

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import (
    SIGNED_HADAMARD_MASK,
)

if TYPE_CHECKING:
    import torch

BLOCK_Q_VALUES = (64, 128)
_BLOCK_N = 64
_NUM_WARPS = 4
# ``cp.async`` moves at most 16 bytes per thread and instruction.
_COPY_BYTES = 16

_GL_BLOCK = gl.constexpr(_BLOCK_N)
_GL_NUM_WARPS = gl.constexpr(_NUM_WARPS)
_GL_COPY_BYTES = gl.constexpr(_COPY_BYTES)
_GL_LOG2_255 = gl.constexpr(7.994353436858858)
# Pads the analytical maximum so the derived bound stays conservative after
# integer-to-FP32 rounding; shared with the Triton kernel's derivation.
_GL_VALUE_LOG_BOUND_CORRECTION = gl.constexpr(0.086085)
# Sage Q quantization constants, matching the per-thread Triton Q kernel.
_GL_LOG2_E = gl.constexpr(1.4426950408889634)
_GL_SCALE_EPSILON = gl.constexpr(1e-7)
_GL_HADAMARD_WORD_0 = gl.constexpr(SIGNED_HADAMARD_MASK[0])
_GL_HADAMARD_WORD_1 = gl.constexpr(SIGNED_HADAMARD_MASK[1])
_GL_HADAMARD_WORD_2 = gl.constexpr(SIGNED_HADAMARD_MASK[2])
_GL_HADAMARD_WORD_3 = gl.constexpr(SIGNED_HADAMARD_MASK[3])


@gluon.jit
def _uint8_int8_mma(lhs, rhs, accumulator):
    gl.static_assert(lhs.dtype == gl.uint8, "lhs must be UINT8")
    gl.static_assert(rhs.dtype == gl.int8, "rhs must be INT8")
    result = mma_v2(lhs.to(gl.int8, bitcast=True), rhs, accumulator)
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
    """Truncate and saturate four probability codes with packed SM72+ PTX."""
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
def _rescale_packed(partial, accumulator, old_weight, current_weight):
    """Update the FP32 numerator, skipping rescaling when both row weights are one.

    Validated with MMA warps [4, 1] for M64 at D64 and D128 and for M128 at D64:
    register order A,A,B,B repeats within each 32-element pack, so elements 0
    and 2 cover both rows. No MMA instruction or collective synchronization occurs in the branch.
    """
    return gl.inline_asm_elementwise(
        asm="""
            {
            .reg .pred keep_a, keep_b, keep_pair;
            .reg .f32 product;
            setp.eq.f32 keep_a, $96, 0f3f800000;
            setp.eq.f32 keep_b, $98, 0f3f800000;
            and.pred keep_pair, keep_a, keep_b;
            @keep_pair bra PIPER_DENSE_RESCALE_DONE;
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
            PIPER_DENSE_RESCALE_DONE:
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
def _hadamard_stage(values, head_dim: gl.constexpr, distance: gl.constexpr):
    """Apply one butterfly stage to columns held in one thread's registers."""
    rows: gl.constexpr = values.shape[0]
    outer: gl.constexpr = head_dim // (2 * distance)
    layout: gl.constexpr = values.type.layout
    grouped = gl.reshape(values, [rows, outer, 2, distance])
    low, high = gl.split(gl.permute(grouped, [0, 1, 3, 2]))
    transformed = gl.join(low + high, low - high)
    result = gl.reshape(gl.permute(transformed, [0, 1, 3, 2]), [rows, head_dim])
    return gl.convert_layout(result, layout)


@gluon.jit
def _rotate_signed_hadamard(values, head_dim: gl.constexpr, final_layout: gl.constexpr):
    """Apply the Sage Q/K signed, normalized Hadamard in the Triton kernel's stage order.

    Every stage but the last pairs columns inside one thread's contiguous run.
    The last pairs columns c and c + head_dim / 2, which ``final_layout`` places
    in one thread; the result is returned in that layout.
    """
    offsets = gl.arange(0, head_dim, gl.SliceLayout(0, values.type.layout))
    group = offsets // 32
    words = gl.where(
        group == 0,
        _GL_HADAMARD_WORD_0,
        gl.where(
            group == 1,
            _GL_HADAMARD_WORD_1,
            gl.where(group == 2, _GL_HADAMARD_WORD_2, _GL_HADAMARD_WORD_3),
        ),
    ).to(gl.uint32)
    signs = gl.where(((words >> (offsets % 32).to(gl.uint32)) & 1) != 0, 1.0, -1.0)
    values = values * signs[None, :]
    values = _hadamard_stage(values, head_dim, 1)
    values = _hadamard_stage(values, head_dim, 2)
    values = _hadamard_stage(values, head_dim, 4)
    values = _hadamard_stage(values, head_dim, 8)
    values = _hadamard_stage(values, head_dim, 16)
    if head_dim == 128:
        values = _hadamard_stage(values, head_dim, 32)
    values = _hadamard_stage(gl.convert_layout(values, final_layout), head_dim, head_dim // 2)
    return values * (head_dim**-0.5)


@gluon.jit
def _round_to_int8(values):
    rounded = values + 0.5 * gl.where(values >= 0, 1.0, -1.0)
    return gl.maximum(-127.0, gl.minimum(127.0, rounded)).to(gl.int8)


@gluon.jit
def _quantize_query_rows(values, softmax_scale, head_dim: gl.constexpr, final_layout: gl.constexpr):
    """Quantize Q rows as the per-thread Sage Q kernel does.

    Rows 32b + 8i + t (i = 0..3) share one scale. The returned row scales
    include the softmax scale and the base-2 conversion.
    """
    rows: gl.constexpr = values.shape[0]
    values = _rotate_signed_hadamard(values, head_dim, final_layout)
    row_max = gl.max(gl.abs(values), axis=1)
    grouped_max = gl.reshape(row_max, [rows // 32, 4, 8])
    scale = gl.max(grouped_max, axis=1) / 127.0 + _GL_SCALE_EPSILON  # pyright: ignore[reportOperatorIssue]
    row_scale, _ = gl.broadcast(gl.expand_dims(scale, 1), grouped_max)
    row_scale = gl.convert_layout(gl.reshape(row_scale, [rows]), row_max.type.layout)
    quantized = _round_to_int8(values / row_scale[:, None])
    return quantized, row_scale * (softmax_scale * _GL_LOG2_E)


@gluon.jit
def _copy_rows(base_ptr, first_row, shared, copy_layout: gl.constexpr):
    """Copy consecutive INT8 rows of one contiguous feature width into shared memory."""
    rows: gl.constexpr = shared.shape[0]
    width: gl.constexpr = shared.shape[1]
    offsets_r = first_row + gl.arange(0, rows, gl.SliceLayout(1, copy_layout))
    offsets_c = gl.arange(0, width, gl.SliceLayout(0, copy_layout))
    async_copy.async_load(shared, base_ptr + offsets_r[:, None] * width + offsets_c[None, :])


@gluon.jit
def _copy_value_tile(value_base, key_storage, start_n, value_shared, copy_layout: gl.constexpr):
    """Copy one transposed [D, K64] V tile."""
    head_dim: gl.constexpr = value_shared.shape[0]
    features = gl.arange(0, head_dim, gl.SliceLayout(1, copy_layout))
    keys = gl.arange(0, _GL_BLOCK, gl.SliceLayout(0, copy_layout))
    async_copy.async_load(
        value_shared,
        value_base + features[:, None] * key_storage + start_n + keys[None, :],
    )


@gluon.jit
def _copy_key_metadata(base, start_n, key_length, shared, copy_layout: gl.constexpr):
    """Copy one K64 tile of per-key FP32 metadata, zero-filling keys past the length."""
    keys = start_n + gl.arange(0, _GL_BLOCK, copy_layout)
    async_copy.async_load(shared, base + keys, mask=keys < key_length)


@gluon.jit
def _probability_tile(
    query,
    key_shared,
    query_scale,
    key_scale_base,
    multiplier_shared,
    denominator,
    running_max,
    start_n,
    query_rows,
    key_length,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
    probability_layout: gl.constexpr,
    masked: gl.constexpr,
    is_causal: gl.constexpr,
):
    """Advance the running maximum and denominator over one K64 tile."""
    block_m: gl.constexpr = query.shape[0]
    key = key_shared.permute([1, 0]).load(key_layout)
    integer_scores = mma_v2(query, key, gl.zeros([block_m, _GL_BLOCK], gl.int32, mma_layout))
    columns = gl.arange(0, _GL_BLOCK, gl.SliceLayout(0, mma_layout))
    keys = start_n + columns
    # Per-thread K quantization gives keys 8j + 2t and 8j + 2t + 1 of a tile one
    # scale: exactly the columns that MMA thread t holds. Reading every column's
    # scale at its group's first key lets the compiler keep one load per thread.
    # A group's first key is valid whenever any of its keys is.
    scale_keys = start_n + (columns % 8) // 2 * 2
    # Keys past the length were zero-filled by their multiplier copy.
    multiplier = multiplier_shared.load(gl.SliceLayout(0, mma_layout))
    if masked:
        valid_keys = keys < key_length
        if block_m > _GL_BLOCK:
            # A masked load per column spills 128-row tiles. Clamping instead keeps
            # one in-bounds read per thread; groups past the length are masked
            # below, so their scale is unused.
            key_scale = gl.load(key_scale_base + gl.minimum(scale_keys, key_length - 1))
        else:
            key_scale = gl.load(
                key_scale_base + scale_keys, mask=scale_keys < key_length, other=0.0
            )
    else:
        key_scale = gl.load(key_scale_base + scale_keys)
    # Both scales are uniform per thread, so their product is formed once per row.
    scores = integer_scores.to(gl.float32) * (query_scale[:, None] * key_scale[None, :])
    # A conservative log2 bound of each V scale, read from its multiplier's bits.
    log_scale = multiplier.to(gl.int32, bitcast=True).to(gl.float32) * (1.0 / 8388608.0) - (
        127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION
    )
    shifted = scores + log_scale[None, :]
    if masked:
        valid = valid_keys[None, :]
        if is_causal:
            valid = valid & (keys[None, :] <= query_rows[:, None])
        shifted = gl.where(valid, shifted, -float("inf"))
    # The final tile of every row holds at least one valid key, so each block
    # maximum is finite.
    block_max = gl.max(shifted, axis=1)
    next_max = gl.maximum(running_max, block_max)
    old_weight = gl.exp2(running_max - next_max)
    current_weight = gl.exp2(block_max - next_max)
    exponent = scores - block_max[:, None]
    if masked:
        exponent = gl.where(valid, exponent, -float("inf"))
    probabilities = gl.exp2(exponent)
    denominator = denominator * old_weight + gl.sum(probabilities, axis=1) * current_weight
    codes = _packed_float32_to_uint8(probabilities * multiplier[None, :] + 0.5)
    codes = gl.convert_layout(codes, probability_layout)
    return codes, denominator, next_max, old_weight, current_weight


@gluon.jit
def _pv_tile(
    codes,
    value_shared,
    accumulator,
    old_weight,
    current_weight,
    mma_layout: gl.constexpr,
    value_layout: gl.constexpr,
):
    """Accumulate one K64 PV product in INT32 and update the FP32 numerator."""
    value = value_shared.permute([1, 0]).load(value_layout)
    partial = _uint8_int8_mma(codes, value, gl.zeros(accumulator.shape, gl.int32, mma_layout))
    return _rescale_packed(partial, accumulator, old_weight, current_weight)


# Storage lengths are whole K64/Q64 blocks, so their specialization never varies
# and proves the 16-byte alignment every ``cp.async`` copy needs.
@gluon.jit(do_not_specialize=["query_length", "key_length", "heads"])
def _dense_piper_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    multiplier_ptr,
    value_mean_ptr,
    output_ptr,
    softmax_scale,
    query_length,
    key_length,
    query_storage,
    key_storage,
    heads,
    stride_qb,
    stride_qh,
    stride_qn,
    head_groups: gl.constexpr,
    head_dim: gl.constexpr,
    is_causal: gl.constexpr,
    block_q: gl.constexpr,
    quantize_query: gl.constexpr,
):
    """Fused UINT8-P/INT8-V online attention for one query tile.

    ``query_ptr`` holds FP16/BF16 Q when ``quantize_query`` is set, and prepared
    INT8 Q with ``query_scale_ptr`` otherwise; the strides describe either.

    Each warp owns ``block_q // 4`` rows: 128-row tiles reuse every K and V
    operand fragment for two 16-row MMA tiles and halve per-key work per row.

    The next K tile and its V multipliers are copied while the current
    probabilities are computed, and the next V tile while the current PV product
    accumulates, so one K and one V commit group are in flight across each wait.
    A CTA barrier after each wait publishes every thread's copies, and one before
    each reissue retires the reads of the buffer being overwritten.
    """
    query_block = gl.program_id(0)
    if is_causal:
        # Launch the longest causal rows first.
        query_block = gl.num_programs(0) - 1 - query_block
    head = gl.program_id(1)
    batch = gl.program_id(2)
    batch_head = (batch * heads + head).to(gl.int64)
    kv_batch_head = (batch * (heads // head_groups) + head // head_groups).to(gl.int64)
    start_m = query_block * block_q
    query_base = query_ptr + batch.to(gl.int64) * stride_qb + head.to(gl.int64) * stride_qh
    key_base = key_ptr + kv_batch_head * key_storage * head_dim
    value_base = value_ptr + kv_batch_head * head_dim * key_storage
    key_scale_base = key_scale_ptr + kv_batch_head * key_length
    multiplier_base = multiplier_ptr + kv_batch_head * key_length

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[_GL_NUM_WARPS, 1],
        instr_shape=[16, 8],
    )
    query_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    key_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    # One 16-byte copy per thread and row segment, with rows spread over warps.
    row_copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, _GL_COPY_BYTES],
        [32 // (head_dim // _GL_COPY_BYTES), head_dim // _GL_COPY_BYTES],
        [_GL_NUM_WARPS, 1],
        [1, 0],
    )
    value_copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, _GL_COPY_BYTES],
        [32 // (_GL_BLOCK // _GL_COPY_BYTES), _GL_BLOCK // _GL_COPY_BYTES],
        [_GL_NUM_WARPS, 1],
        [1, 0],
    )
    metadata_copy_layout: gl.constexpr = gl.BlockedLayout([1], [32], [_GL_NUM_WARPS], [0])
    row_shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [_GL_BLOCK, head_dim], gl.int8
    )
    value_shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [head_dim, _GL_BLOCK], gl.int8
    )
    query_shared = gl.allocate_shared_memory(
        gl.int8,
        [block_q, head_dim],
        gl.NVMMASharedLayout.get_default_for([block_q, head_dim], gl.int8),
    )
    key_shared = gl.allocate_shared_memory(gl.int8, [_GL_BLOCK, head_dim], row_shared_layout)
    value_shared = gl.allocate_shared_memory(gl.int8, [head_dim, _GL_BLOCK], value_shared_layout)
    multiplier_shared = gl.allocate_shared_memory(
        gl.float32, [_GL_BLOCK], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )

    tile_count = gl.cdiv(key_length, _GL_BLOCK)
    mask_from = tile_count - 1
    if is_causal:
        tile_count = gl.minimum(tile_count, (start_m + block_q) // _GL_BLOCK)
        # Every K64 tile that meets this block's rows on the diagonal is masked.
        mask_from = gl.minimum(tile_count - 1, start_m // _GL_BLOCK)

    if quantize_query:
        # Each Q row spans 128 // block_q threads. Loading Q before the first K
        # and V copies keeps its latency ahead of theirs.
        row_threads: gl.constexpr = 32 * _GL_NUM_WARPS // block_q
        quantize_layout: gl.constexpr = gl.BlockedLayout(
            [1, head_dim // row_threads],
            [32 // row_threads, row_threads],
            [_GL_NUM_WARPS, 1],
            [1, 0],
        )
        # The last Hadamard stage pairs columns c and c + head_dim / 2 in one thread.
        final_layout: gl.constexpr = gl.BlockedLayout(
            [1, head_dim // (2 * row_threads)],
            [32 // row_threads, row_threads],
            [_GL_NUM_WARPS, 1],
            [1, 0],
        )
        load_rows = start_m + gl.arange(0, block_q, gl.SliceLayout(1, quantize_layout))
        load_features = gl.arange(0, head_dim, gl.SliceLayout(0, quantize_layout))
        # Rows past the length read zero Q, which keeps their recurrence finite.
        raw_query = gl.load(
            query_base + load_rows[:, None].to(gl.int64) * stride_qn + load_features[None, :],
            mask=(load_rows < query_length)[:, None],
            other=0.0,
        ).to(gl.float32)
    else:
        _copy_rows(query_base, start_m, query_shared, row_copy_layout)
    _copy_rows(key_base, 0, key_shared, row_copy_layout)
    _copy_key_metadata(multiplier_base, 0, key_length, multiplier_shared, metadata_copy_layout)
    async_copy.commit_group()
    _copy_value_tile(value_base, key_storage, 0, value_shared, value_copy_layout)
    async_copy.commit_group()

    if quantize_query:
        # Quantize while the first K and V tiles are in flight.
        quantized_query, quantized_scale = _quantize_query_rows(
            raw_query, softmax_scale, head_dim, final_layout
        )
        query_shared.store(quantized_query)
    else:
        # Q shares the first K group; only the first V group may remain in flight.
        async_copy.wait_group(1)
    gl.barrier()
    query = query_shared.load(query_layout)
    query_rows = start_m + gl.arange(0, block_q, row_layout)
    if quantize_query:
        query_scale = gl.convert_layout(quantized_scale, row_layout)
    else:
        # Padded query rows have zero scales and zero codes, so they stay finite.
        query_scale = gl.load(query_scale_ptr + batch_head * query_storage + query_rows)
    accumulator = gl.zeros([block_q, head_dim], gl.float32, mma_layout)
    denominator = gl.zeros([block_q], gl.float32, row_layout)
    running_max = gl.full([block_q], -float("inf"), gl.float32, row_layout)

    for tile in range(tile_count - 1):
        start_n = tile * _GL_BLOCK
        # The consumed K group is followed only by its V group.
        async_copy.wait_group(1)
        gl.barrier()
        if is_causal and block_q > _GL_BLOCK and tile >= mask_from:
            codes, denominator, running_max, old_weight, current_weight = _probability_tile(
                query,
                key_shared,
                query_scale,
                key_scale_base,
                multiplier_shared,
                denominator,
                running_max,
                start_n,
                query_rows,
                key_length,
                mma_layout,
                key_layout,
                probability_layout,
                True,
                is_causal,
            )
        else:
            codes, denominator, running_max, old_weight, current_weight = _probability_tile(
                query,
                key_shared,
                query_scale,
                key_scale_base,
                multiplier_shared,
                denominator,
                running_max,
                start_n,
                query_rows,
                key_length,
                mma_layout,
                key_layout,
                probability_layout,
                False,
                is_causal,
            )
        gl.barrier()
        _copy_rows(key_base, start_n + _GL_BLOCK, key_shared, row_copy_layout)
        _copy_key_metadata(
            multiplier_base,
            start_n + _GL_BLOCK,
            key_length,
            multiplier_shared,
            metadata_copy_layout,
        )
        async_copy.commit_group()
        # The consumed V group is followed only by the next K group.
        async_copy.wait_group(1)
        gl.barrier()
        accumulator = _pv_tile(
            codes,
            value_shared,
            accumulator,
            old_weight,
            current_weight,
            mma_layout,
            value_layout,
        )
        gl.barrier()
        _copy_value_tile(
            value_base, key_storage, start_n + _GL_BLOCK, value_shared, value_copy_layout
        )
        async_copy.commit_group()

    # The final tile holds the key tail and, for causal attention, the diagonal.
    async_copy.wait_group(1)
    gl.barrier()
    codes, denominator, running_max, old_weight, current_weight = _probability_tile(
        query,
        key_shared,
        query_scale,
        key_scale_base,
        multiplier_shared,
        denominator,
        running_max,
        (tile_count - 1) * _GL_BLOCK,
        query_rows,
        key_length,
        mma_layout,
        key_layout,
        probability_layout,
        True,
        is_causal,
    )
    async_copy.wait_group(0)
    gl.barrier()
    accumulator = _pv_tile(
        codes,
        value_shared,
        accumulator,
        old_weight,
        current_weight,
        mma_layout,
        value_layout,
    )

    output = accumulator / (gl.maximum(denominator, 1e-30) * 255.0)[:, None]
    features = gl.arange(0, head_dim, column_layout)
    if not is_causal:
        output += gl.load(value_mean_ptr + kv_batch_head * head_dim + features)[None, :]
    gl.store(
        output_ptr
        + (batch_head * query_length + query_rows[:, None]) * head_dim
        + features[None, :],
        output.to(output_ptr.dtype.element_ty),
        mask=(query_rows < query_length)[:, None],
    )


def launch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    output: torch.Tensor,
    *,
    softmax_scale: float,
    key_length: int,
    is_causal: bool,
    block_q: int,
    quantize_query: bool,
    max_registers: int | None,
) -> torch.Tensor:
    """Launch the recurrence on INT8 K/V stored in whole K64 tiles.

    With ``quantize_query``, ``query`` is the FP16/BF16 ``[batch, heads,
    query_length, head_dim]`` input and ``query_scale`` is unused. Otherwise it is
    prepared INT8 ``[batch, heads, query_storage, head_dim]`` storage with rows,
    and Q scales, padded to a multiple of ``block_q``. ``key`` is ``[batch,
    kv_heads, key_storage, head_dim]`` and ``value`` its transposed ``[batch,
    kv_heads, head_dim, key_storage]``, padded to a multiple of 64 keys. K scales
    and V multipliers cover the logical key length.
    """
    batch, heads, query_length, head_dim = output.shape
    # Raw Q is read through its strides; its length stays a runtime value, and the
    # storage argument keeps a length-independent specialization.
    query_storage = (
        triton.cdiv(query_length, block_q) * block_q if quantize_query else query.shape[2]
    )
    key_storage = key.shape[2]
    compile_options = {} if max_registers is None else {"maxnreg": max_registers}
    with device_context(output.device):
        install_uint8_int8_dot_hook()
        _dense_piper_attention_kernel[(triton.cdiv(query_length, block_q), heads, batch)](
            query,
            key,
            value,
            query_scale,
            key_scale,
            value_scale_multiplier,
            value_mean,
            output,
            softmax_scale,
            query_length,
            key_length,
            query_storage,
            key_storage,
            heads,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            heads // key.shape[1],
            head_dim,
            is_causal,
            block_q,
            quantize_query,
            num_warps=_NUM_WARPS,
            num_stages=1,
            **compile_options,
        )
    return output
