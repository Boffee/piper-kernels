"""Shared dense/sparse RDNA4 WMMA for transposed QK/PV fragments."""

# Gluon device function types are not represented by Python annotations.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportAssignmentType=false
# pyright: reportCallIssue=false

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_INIT_ASM = gl.constexpr(
    (
        "v_wmma_i32_16x16x16_iu8 v[0:7], $9, $8, 0 neg_lo:[1,0,0]",
        "v_wmma_i32_16x16x16_iu8 v[8:15], $9, $8, 0 neg_lo:[1,0,0]",
        "v_wmma_i32_16x16x16_iu8 v[0:7], $9, $8, 0x3e22f983 neg_lo:[1,1,0]",
        "v_wmma_i32_16x16x16_iu8 v[8:15], $9, $8, 0x3e22f983 neg_lo:[1,1,0]",
    )
)
_ACCUMULATE_ASM = gl.constexpr(
    (
        "v_wmma_i32_16x16x16_iu8 v[0:7], $9, $8, v[0:7] neg_lo:[1,0,0]",
        "v_wmma_i32_16x16x16_iu8 v[8:15], $9, $8, v[8:15] neg_lo:[1,0,0]",
        "v_wmma_i32_16x16x16_iu8 v[0:7], $9, $8, v[0:7] neg_lo:[1,1,0]",
        "v_wmma_i32_16x16x16_iu8 v[8:15], $9, $8, v[8:15] neg_lo:[1,1,0]",
    )
)
_INIT_CONSTRAINTS = gl.constexpr(
    (
        "={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},v,v",
        "={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},v,v",
    )
)
_ACCUMULATE_CONSTRAINTS = gl.constexpr(
    (
        "={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},v,v,0,1,2,3,4,5,6,7",
        "={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},v,v,0,1,2,3,4,5,6,7",
    )
)


@gluon.jit
def _wmma_iu8(lhs_bits, rhs_bits, acc, bank: gl.constexpr, signed_rhs: gl.constexpr):
    gl.static_assert(bank >= 0)
    gl.static_assert(bank <= 1)
    variant: gl.constexpr = signed_rhs * 2 + bank
    if acc is None:
        return gl.inline_asm_elementwise(
            _INIT_ASM[variant],
            constraints=_INIT_CONSTRAINTS[bank],
            args=(lhs_bits, rhs_bits),
            dtype=(gl.int32,) * 8,
            is_pure=True,
            pack=1,
        )
    else:
        return gl.inline_asm_elementwise(
            _ACCUMULATE_ASM[variant],
            constraints=_ACCUMULATE_CONSTRAINTS[bank],
            args=(
                lhs_bits,
                rhs_bits,
                acc[0],
                acc[1],
                acc[2],
                acc[3],
                acc[4],
                acc[5],
                acc[6],
                acc[7],
            ),
            dtype=(gl.int32,) * 8,
            is_pure=True,
            pack=1,
        )


@gluon.jit
def uint8_int8_wmma(lhs_bits, rhs_bits, acc, bank: gl.constexpr):
    """Unsigned P times signed V, physically transposed for wave-local fragments.

    Pass ``acc=None`` on the first K step to seed the accumulator inline with 0
    without materializing an eight-VGPR zero splat. Each explicit contiguous
    bank describes the eight-VGPR tuple to the compiler without hidden clobbers.
    """
    return _wmma_iu8(lhs_bits, rhs_bits, acc, bank, 0)


@gluon.jit
def int8_wmma(lhs_bits, rhs_bits, acc, bank: gl.constexpr):
    """Signed QK in the same transposed wave-local fragment contract as PV.

    Pass ``acc=None`` on the first K step to seed the accumulator inline with
    ``0x3e22f983`` so bounded D64/D128 INT8 dot products stay in one FP32
    binade for ``_recover_scores`` without an ``i32 -> f32`` conversion or an
    eight-VGPR bias splat.
    """
    return _wmma_iu8(lhs_bits, rhs_bits, acc, bank, 1)
