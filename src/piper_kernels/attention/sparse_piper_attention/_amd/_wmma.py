"""Native RDNA4 WMMA for the fixed transposed QK/PV fragment layout."""

# Gluon device function types are not represented by Python annotations.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportAssignmentType=false
# pyright: reportCallIssue=false

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def uint8_int8_wmma(lhs_bits, rhs_bits, acc, bank: gl.constexpr):
    """Unsigned P times signed V, physically transposed for wave-local fragments.

    Each output register is tied to its accumulator input. Explicit contiguous
    banks describe the eight-VGPR tuple to the compiler without hidden clobbers.
    Both banks are used by the production schedule; operand orientation is fixed.
    """
    gl.static_assert(bank >= 0)
    gl.static_assert(bank <= 1)
    if bank == 0:
        instruction: gl.constexpr = "v_wmma_i32_16x16x16_iu8 v[0:7], $9, $8, v[0:7] neg_lo:[1,0,0]"
        constraints: gl.constexpr = (
            "={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},v,v,0,1,2,3,4,5,6,7"
        )
    else:
        instruction: gl.constexpr = (
            "v_wmma_i32_16x16x16_iu8 v[8:15], $9, $8, v[8:15] neg_lo:[1,0,0]"
        )
        constraints: gl.constexpr = (
            "={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},v,v,0,1,2,3,4,5,6,7"
        )
    return gl.inline_asm_elementwise(
        instruction,
        constraints=constraints,
        args=(lhs_bits, rhs_bits, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]),
        dtype=(gl.int32,) * 8,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def int8_wmma(lhs_bits, rhs_bits, acc, bank: gl.constexpr):
    """Signed QK in the same transposed wave-local fragment contract as PV."""
    gl.static_assert(bank >= 0)
    gl.static_assert(bank <= 1)
    if bank == 0:
        instruction: gl.constexpr = "v_wmma_i32_16x16x16_iu8 v[0:7], $9, $8, v[0:7] neg_lo:[1,1,0]"
        constraints: gl.constexpr = (
            "={v0},={v1},={v2},={v3},={v4},={v5},={v6},={v7},v,v,0,1,2,3,4,5,6,7"
        )
    else:
        instruction: gl.constexpr = (
            "v_wmma_i32_16x16x16_iu8 v[8:15], $9, $8, v[8:15] neg_lo:[1,1,0]"
        )
        constraints: gl.constexpr = (
            "={v8},={v9},={v10},={v11},={v12},={v13},={v14},={v15},v,v,0,1,2,3,4,5,6,7"
        )
    return gl.inline_asm_elementwise(
        instruction,
        constraints=constraints,
        args=(lhs_bits, rhs_bits, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]),
        dtype=(gl.int32,) * 8,
        is_pure=True,
        pack=1,
    )
