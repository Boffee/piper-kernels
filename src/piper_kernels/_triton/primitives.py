"""Shared format-independent Triton primitives and compiler defaults."""

# Triton's JIT DSL types are not fully represented in its Python typing surface.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

type TritonCompileFn = Callable[
    [ASTSource, GPUTarget, dict[str, object]],
    triton.compiler.CompiledKernel,
]

TRITON_FLOAT_POINTER_TYPES: dict[torch.dtype, str] = {
    torch.float16: "*fp16",
    torch.bfloat16: "*bf16",
    torch.float32: "*fp32",
}


def default_triton_compile(
    source: ASTSource,
    target: GPUTarget,
    options: dict[str, object],
) -> triton.compiler.CompiledKernel:
    """Invoke Triton's standard offline/online kernel compiler."""
    return triton.compile(source, target=target, options=options)


@triton.jit
def splitk_tile_bounds(total_tiles, pid_k, split_k):
    """Return balanced Split-K tile interval ``[tile_start, tile_end)`` for ``pid_k``."""
    base_tiles = total_tiles // split_k
    remainder_tiles = total_tiles % split_k
    tile_start = pid_k * base_tiles + tl.minimum(pid_k, remainder_tiles)
    tile_end = tile_start + base_tiles + tl.where(pid_k < remainder_tiles, 1, 0)
    return tile_start, tile_end


@triton.jit
def decode_e4m3(code):
    """Exact FP32 value of FP8 E4M3FN byte bits without backend FP8 lowering quirks."""
    sign = code >> 7
    exponent = (code >> 3) & 15
    mantissa = code & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    subnormal = mantissa.to(tl.float32) * 0.001953125
    magnitude = tl.where(exponent == 0, subnormal, normal)
    return tl.where(sign != 0, -magnitude, magnitude)


__all__ = [
    "TRITON_FLOAT_POINTER_TYPES",
    "TritonCompileFn",
    "decode_e4m3",
    "default_triton_compile",
    "splitk_tile_bounds",
]
