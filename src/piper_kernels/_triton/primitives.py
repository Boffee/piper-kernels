"""Shared format-independent Triton primitives and compiler defaults."""

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


__all__ = [
    "TRITON_FLOAT_POINTER_TYPES",
    "TritonCompileFn",
    "default_triton_compile",
    "splitk_tile_bounds",
]
