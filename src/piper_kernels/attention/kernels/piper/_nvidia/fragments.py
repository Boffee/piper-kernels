"""Shared NVIDIA matrix products and packed arithmetic for dense and sparse Piper."""

# Gluon device values and inline-assembly signatures are not Python values.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportCallIssue=false

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

_RESCALE_ASM = gl.constexpr(
    "\n".join(
        [
            "{",
            ".reg .pred keep_a, keep_b, keep_pair;",
            ".reg .f32 product;",
            "setp.eq.f32 keep_a, $96, 0f3f800000;",
            "setp.eq.f32 keep_b, $98, 0f3f800000;",
            "and.pred keep_pair, keep_a, keep_b;",
            "@keep_pair bra PIPER_RESCALE_DONE;",
        ]
        + [f"mul.rn.f32 ${index}, ${index}, ${96 + index};" for index in range(32)]
        + ["PIPER_RESCALE_DONE:"]
        + [
            f"cvt.rn.f32.s32 product, ${32 + index};\n"
            f"fma.rn.f32 ${index}, product, ${128 + index}, ${index};"
            for index in range(32)
        ]
        + ["}"]
    )
)
_RESCALE_CONSTRAINTS = gl.constexpr(
    ",".join(
        # $0..31: FP32 outputs, written before all inputs are consumed.
        ["=&f"] * 32
        # $32..63: INT32 PV products.
        + ["r"] * 32
        # $64..95: FP32 accumulator inputs tied to the output registers.
        + [str(index) for index in range(32)]
        # $96..127: old row weights (elements 0 and 2 are $96 and $98).
        + ["f"] * 32
        # $128..159: current row weights.
        + ["f"] * 32
    )
)


@gluon.jit
def uint8_int8_mma(lhs, rhs, accumulator):
    """Mark the signed MMAv2 product for the mixed UINT8/INT8 lowering hook."""
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
def packed_float32_to_uint8(values):
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
def rescale_packed(partial, accumulator, old_weight, current_weight):
    """Update the FP32 numerator, skipping rescaling when both row weights are one.

    Validated for M64/D64 MMA[2,1] or [4,1], M128/D64 MMA[4,1], and
    M64/D128 MMA[4,1]: register order A,A,B,B repeats within each 32-element
    pack. Checking elements 0 and 2 covers both rows. No MMA instruction or
    collective synchronization occurs inside the branch.
    """
    return gl.inline_asm_elementwise(
        asm=_RESCALE_ASM,
        constraints=_RESCALE_CONSTRAINTS,
        args=[partial, accumulator, old_weight[:, None], current_weight[:, None]],
        dtype=gl.float32,
        is_pure=True,
        pack=32,
    )
