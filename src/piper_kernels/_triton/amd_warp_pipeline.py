"""Keep Piper's RDNA4 warp-pipelined loads schedulable under Triton 3.9.

Triton 3.9 (triton-lang/triton#10840) lowers every ``gl.amd.warp_pipeline_stage``
with a scheduling barrier after each memory operation, so the backend keeps a
stage's memory operations in program order. That helps GEMM prefetching on CDNA,
but it pins Piper's K/V loads beside their WMMAs: LLVM can no longer hoist or pair
them, and dense RDNA4 attention loses about 5%.

This module removes those per-memory-operation barriers from Piper's own
warp-pipelined kernels after Triton lowers them to LLVM IR. Stage-boundary
barriers stay, and every other kernel passes through unchanged.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable
from typing import Any, cast

import torch
import triton
from triton import knobs

# Kernels whose warp-pipeline stages Piper schedules itself.
_KERNELS = frozenset({"_dense_piper_kernel", "_sparse_piper_attention_kernel"})
_KERNEL_NAME = re.compile(r"define amdgpu_kernel void @([-a-zA-Z$._0-9]+)\(")
# Mask 1 lets only non-memory ALU work cross; Triton emits it after each memory op.
_MEMORY_ORDER_BARRIER = re.compile(
    r"^[ \t]*(?:tail )?call void @llvm\.amdgcn\.sched\.barrier\(i32 1\)[^\n]*\n", re.MULTILINE
)
_CACHE_KEY = "piper_amd_warp_pipeline_memory_order_v1"
_CACHE_HASH = hashlib.sha256(_CACHE_KEY.encode()).hexdigest()
_INSTALL_LOCK = threading.Lock()


def strip_memory_order_barriers(llvm_ir: str) -> str:
    """Remove per-memory-operation scheduling barriers from Piper's pipelined kernels."""
    name = _KERNEL_NAME.search(llvm_ir)
    if name is None or name.group(1) not in _KERNELS:
        return llvm_ir
    return _MEMORY_ORDER_BARRIER.sub("", llvm_ir)


class _StageHook:
    def __init__(self, previous: Callable[..., tuple[str, str]] | None) -> None:
        self.previous = previous

    def _cache_identity(self) -> tuple[str, str]:
        if self.previous is None:
            return _CACHE_KEY, _CACHE_HASH
        previous_key, previous_hash = self.previous()
        return (
            f"{previous_key}\0{_CACHE_KEY}",
            hashlib.sha256(f"{previous_hash}\0{_CACHE_HASH}".encode()).hexdigest(),
        )

    def __call__(
        self,
        backend: object | None = None,
        stages: dict[str, Callable[..., object]] | None = None,
        options: object | None = None,
        language: object | None = None,
        capability: object | None = None,
    ) -> tuple[str, str]:
        if all(item is None for item in (backend, stages, options, language, capability)):
            return self._cache_identity()
        if self.previous is not None:
            self.previous(backend, stages, options, language, capability)
        make_llir = None if stages is None else stages.get("llir")
        if stages is not None and make_llir is not None:
            stages["llir"] = lambda src, metadata: strip_memory_order_barriers(
                cast(str, make_llir(src, metadata))
            )
        return self._cache_identity()


def _triton_adds_memory_order_barriers() -> bool:
    major, minor = (int(part) for part in re.findall(r"\d+", triton.__version__)[:2])
    return (major, minor) >= (3, 9)


def _register() -> None:
    runtime_knobs = cast(Any, knobs.runtime)
    if not hasattr(runtime_knobs, "add_stages_inspection_hook"):
        return
    with _INSTALL_LOCK:
        current = runtime_knobs.add_stages_inspection_hook
        if isinstance(current, _StageHook):
            return
        runtime_knobs.add_stages_inspection_hook = _StageHook(
            cast(Callable[..., tuple[str, str]] | None, current)
        )


# Install at import so every kernel's cache identity is settled before the first
# compile; installing later would recompile kernels that were already cached.
if torch.version.hip is not None and _triton_adds_memory_order_barriers():
    _register()
