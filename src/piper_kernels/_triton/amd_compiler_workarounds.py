"""Workarounds for Triton 3.9 code generation in Piper's RDNA4 attention kernels.

Two Triton 3.9 compiler changes made dense Piper attention 12-17% slower on
RX 9070 XT, and sparse Piper attention about 3% slower. Each workaround below
applies to a range of Triton releases.

1. Triton 3.8.0 disabled LLVM's ``slp-copyable-elements`` for that release only.
   With it enabled, the SLP vectorizer bundles scalar FMAs into vector FMAs.
   RDNA4 has no packed FP32 FMA, so the vectors only add register pressure: dense
   D128 attention grows from 188 to 208 VGPRs and drops from eight waves to
   seven. Kernels that read :data:`USE_SCALAR_FMA` issue those FMAs as opaque
   scalar instructions instead. Remove this once Triton disables that
   vectorization for AMD again or offers a per-kernel opt-out. Uses:
   ``_SCALAR_FMA``, ``scalar_fma``, and ``qk_tiles(scalar_score_recovery=...)``
   in the shared RDNA4 fragments, and their dense Piper callers.
2. triton-lang/triton#10840 adds a scheduling barrier after every memory
   operation in a warp-pipeline stage. That keeps CDNA GEMM prefetches in order,
   but it pins Piper's K/V loads beside their WMMAs, so LLVM can no longer hoist
   or pair them. Kernels that pass :data:`UNPINNED_WARP_PIPELINE_ATTRS` in their
   ``llvm_fn_attrs`` have those barriers removed from their LLVM IR; stage
   boundaries remain. Remove this once Triton scopes the barriers or offers an
   opt-out. Uses: :data:`UNPINNED_WARP_PIPELINE_ATTRS` at the dense and sparse
   Piper launches, and the stage hook below.

To retire a workaround, set its ``fixed`` version below when a Triton release
addresses the cause; releases from then on compile without it. Once Piper's
minimum supported Triton reaches ``fixed``, delete the workaround and the uses
it lists. Tests in ``tests/attention/piper_attention/test_amd_compile.py`` fail
when an installed Triton no longer needs a workaround that still applies.
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

_TRITON_VERSION = tuple(int(part) for part in re.findall(r"\d+", triton.__version__)[:2])


def _applies_to_installed_triton(first: tuple[int, int], fixed: tuple[int, int] | None) -> bool:
    """Whether the installed Triton is in ``[first, fixed)``; ``fixed=None`` is unfixed."""
    return first <= _TRITON_VERSION and (fixed is None or fixed > _TRITON_VERSION)


USE_SCALAR_FMA = _applies_to_installed_triton(first=(3, 9), fixed=None)
"""Whether RDNA4 kernels issue FMAs that LLVM must not vectorize as scalar assembly.

On Triton 3.8 plain FMAs stay scalar, and inline assembly would cost 1-3% because
the compiler cannot dual-issue it.
"""

_UNPINNED_WARP_PIPELINE_MARKER = "piper-unpinned-warp-pipeline"
_PINS_WARP_PIPELINE_MEMORY = _applies_to_installed_triton(first=(3, 9), fixed=None)
UNPINNED_WARP_PIPELINE_ATTRS: tuple[tuple[str, str], ...] = (
    ((_UNPINNED_WARP_PIPELINE_MARKER, ""),) if _PINS_WARP_PIPELINE_MEMORY else ()
)
"""``llvm_fn_attrs`` entries that opt a warp-pipelined kernel out of pinned loads.

LLVM ignores the attribute itself. It is empty for Triton releases that add no
such barriers, so those kernels compile unchanged.
"""

# Mask 1 lets only non-memory ALU work cross; Triton emits it after each memory op.
_MEMORY_ORDER_BARRIER = re.compile(
    r"^[ \t]*(?:tail )?call void @llvm\.amdgcn\.sched\.barrier\(i32 1\)[^\n]*\n", re.MULTILINE
)
_CACHE_KEY = "piper_amd_unpinned_warp_pipeline_v1"
_CACHE_HASH = hashlib.sha256(_CACHE_KEY.encode()).hexdigest()
_INSTALL_LOCK = threading.Lock()


def strip_memory_order_barriers(llvm_ir: str) -> str:
    """Remove warp-pipeline barriers that pin each memory operation in place."""
    return _MEMORY_ORDER_BARRIER.sub("", llvm_ir)


def _requests_unpinned_warp_pipeline(options: object) -> bool:
    attributes = getattr(options, "llvm_fn_attrs", ())
    return any(name == _UNPINNED_WARP_PIPELINE_MARKER for name, _ in attributes)


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
        if (
            stages is not None
            and make_llir is not None
            and _requests_unpinned_warp_pipeline(options)
        ):
            stages["llir"] = lambda src, metadata: strip_memory_order_barriers(
                cast(str, make_llir(src, metadata))
            )
        return self._cache_identity()


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
if torch.version.hip is not None and _PINS_WARP_PIPELINE_MEMORY:
    _register()
