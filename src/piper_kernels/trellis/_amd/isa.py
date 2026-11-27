"""AMD GCN byte-sum ISA primitives for little-endian Trellis decoding."""

# Triton's JIT decorator and inline GCN intrinsic signatures are not fully typed.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportReturnType=false

from __future__ import annotations

import triton
import triton.language as tl

from .._layout import CODE_OFFSET

_NEG_CODE_OFFSET = tl.constexpr(-CODE_OFFSET)


@triton.jit
def sad_u8_lcg_numerator_i32(x: tl.tensor) -> tl.tensor:
    """Sum the four unsigned bytes of ``x`` minus ``CODE_OFFSET`` via AMD ``v_sad_u8``."""
    offset = tl.full(x.shape, _NEG_CODE_OFFSET, tl.int32)
    return tl.inline_asm_elementwise(
        asm="v_sad_u8 $0, $1, 0, $2;",
        constraints="=v,v,v",
        args=[x, offset],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
