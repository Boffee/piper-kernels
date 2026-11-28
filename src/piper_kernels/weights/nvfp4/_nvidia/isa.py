"""NVIDIA SM120 inline PTX NVFP4 packing primitives."""

# Triton's JIT DSL types are not fully represented in its Python typing surface.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def pack_e2m1_pairs(low, high):
    """Pack two FP32 values into one SM120 E2M1x2 byte via hardware conversion."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 packed;
            cvt.rn.satfinite.e2m1x2.f32 packed, $2, $1;
            cvt.u32.u8 $0, packed;
        }
        """,
        constraints="=r,f,f",
        args=[low, high],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )
