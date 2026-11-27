"""NVIDIA PTX byte-sum ISA primitives for little-endian Trellis decoding."""

# Triton's JIT decorator and inline PTX intrinsic signatures are not fully typed.
# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportReturnType=false

from __future__ import annotations

import triton
import triton.language as tl

from .._layout import CODE_OFFSET

if CODE_OFFSET != 510:
    raise AssertionError(
        f"NVIDIA dp4a_lcg_numerator_i32 hardcodes -510 matching CODE_OFFSET, got {CODE_OFFSET}"
    )


@triton.jit
def dp4a_lcg_numerator_i32(x: tl.tensor) -> tl.tensor:
    """Sum the four unsigned bytes of ``x`` minus 510 via NVIDIA ``dp4a.s32.u32``."""
    return tl.inline_asm_elementwise(
        asm="dp4a.s32.u32 $0, 0x01010101, $1, -510;",
        constraints="=r,r",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
