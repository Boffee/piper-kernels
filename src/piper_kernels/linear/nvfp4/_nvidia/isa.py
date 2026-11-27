"""NVIDIA SM120 inline PTX NVFP4 decode and block-scale primitives."""

# Triton's JIT DSL types are not fully represented in its Python typing surface.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def decode_sm120_e2m1x2_f32(packed_u8, high_first: tl.constexpr):
    """Decode packed E2M1 byte pairs via SM120 hardware cvt into exact FP32 pairs."""
    lo, hi = tl.inline_asm_elementwise(
        asm=(
            "{ .reg .b8 b0; .reg .b32 p; "
            "mov.b16 {b0, _}, $2; "
            "cvt.rn.f16x2.e2m1x2 p, b0; "
            "mov.b32 {$0, $1}, p; }"
        ),
        constraints="=h,=h,h",
        args=[packed_u8.to(tl.uint16)],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    if high_first:
        return hi.to(tl.float32), lo.to(tl.float32)
    return lo.to(tl.float32), hi.to(tl.float32)


@triton.jit
def decode_scale_sm120_e2m1x2(
    packed_u8,
    scales_16,
    high_first: tl.constexpr,
    is_fp16: tl.constexpr,
):
    """Decode and block-scale packed E2M1 byte pairs using SM120 hardware vector ops."""
    if is_fp16:
        lo, hi = tl.inline_asm_elementwise(
            asm=(
                "{ .reg .b8 b0; .reg .b32 p, s2; "
                "mov.b16 {b0, _}, $2; "
                "cvt.rn.f16x2.e2m1x2 p, b0; "
                "mov.b32 s2, {$3, $3}; "
                "mul.rn.f16x2 p, p, s2; "
                "mov.b32 {$0, $1}, p; }"
            ),
            constraints="=h,=h,h,h",
            args=[packed_u8.to(tl.uint16), scales_16],
            dtype=(tl.float16, tl.float16),
            is_pure=True,
            pack=1,
        )
    else:
        lo, hi = tl.inline_asm_elementwise(
            asm=(
                "{ .reg .b8 b0; .reg .b32 p, s2; "
                "mov.b16 {b0, _}, $2; "
                "cvt.rn.bf16x2.e2m1x2 p, b0; "
                "mov.b32 s2, {$3, $3}; "
                "mul.rn.bf16x2 p, p, s2; "
                "mov.b32 {$0, $1}, p; }"
            ),
            constraints="=h,=h,h,h",
            args=[packed_u8.to(tl.uint16), scales_16],
            dtype=(tl.bfloat16, tl.bfloat16),
            is_pure=True,
            pack=1,
        )
    if high_first:
        return hi, lo
    return lo, hi
