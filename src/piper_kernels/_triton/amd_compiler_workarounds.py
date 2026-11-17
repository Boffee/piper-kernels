"""Workarounds for Triton code generation in Piper's RDNA4 kernels.

Each workaround below applies to a range of Triton releases. The first two are
Triton 3.9 changes that made dense Piper attention 12-17% slower on RX 9070 XT,
and sparse Piper attention about 3% slower.

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
3. Triton fences every address space at the workgroup barriers that guard
   shared-memory (LDS) reuse. In RDNA's default WGP mode, LLVM lowers the
   release fence to a wait for all outstanding global loads and the acquire
   fence to an L0 cache invalidate (``global_inv``). A software-pipelined GEMM
   therefore drains its next tile's prefetch at every barrier and refetches
   global data after it. On RX 9070 XT this costs the ConvRot INT8 GEMM 2-4% and
   the fused sparse Piper Q/K/V projections about 8%. Kernels whose barriers order
   only LDS pass :data:`LDS_ONLY_FENCE_ATTRS`; their workgroup fences are tagged
   as LDS-only, which keeps the LDS waits and drops the global ones. Remove this
   once Triton emits LDS-only fences for its shared-memory barriers. Uses:
   :data:`LDS_ONLY_FENCE_ATTRS` at the RDNA4 ConvRot INT8 GEMM, dense Piper
   attention, and sparse Piper attention launches; :func:`lds_only_fence_options`
   at the shared Conv3D and dense/sparse Piper projection launches, which RDNA4
   policies enable through their plans' ``lds_only_fences``; and the stage hook
   below.

To retire a workaround, set its ``fixed`` version below when a Triton release
addresses the cause; releases from then on compile without it. Once Piper's
minimum supported Triton reaches ``fixed``, delete the workaround and the uses
it lists. Tests in ``tests/attention/piper_attention/test_amd_compile.py`` and
``tests/linear/convrot/int8/test_amd_compile.py`` fail when an installed Triton
no longer needs a workaround that still applies.
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

_LDS_ONLY_FENCE_MARKER = "piper-lds-only-fences"
# Triton 3.8 is the oldest release Piper's ROCm environments use, not where the fences began.
_FENCES_GLOBAL_MEMORY = _applies_to_installed_triton(first=(3, 8), fixed=None)
LDS_ONLY_FENCE_ATTRS: tuple[tuple[str, str], ...] = (
    ((_LDS_ONLY_FENCE_MARKER, ""),) if _FENCES_GLOBAL_MEMORY else ()
)
"""``llvm_fn_attrs`` entries that restrict a kernel's workgroup fences to LDS.

Pass them only for kernels whose barriers order shared memory alone: global
memory written by one wave is then not ordered for the rest of its workgroup.
"""


def lds_only_fence_options(requested: bool) -> dict[str, object]:
    """Launch options applying :data:`LDS_ONLY_FENCE_ATTRS` when an execution plan asks."""
    return {"llvm_fn_attrs": LDS_ONLY_FENCE_ATTRS} if requested and LDS_ONLY_FENCE_ATTRS else {}


# Mask 1 lets only non-memory ALU work cross; Triton emits it after each memory op.
_MEMORY_ORDER_BARRIER = re.compile(
    r"^[ \t]*(?:tail )?call void @llvm\.amdgcn\.sched\.barrier\(i32 1\)[^\n]*\n", re.MULTILINE
)
_WORKGROUP_FENCE = re.compile(
    r'^([ \t]*fence syncscope\("workgroup"\) (?:acquire|release|acq_rel))'
    r"((?:, ![\w.]+ ![\w.]+)*)[ \t]*$",
    re.MULTILINE,
)
_METADATA_ID = re.compile(r"^!(\d+) = ", re.MULTILINE)
_CACHE_KEY = "piper_amd_llir_workarounds_v2"
_CACHE_HASH = hashlib.sha256(_CACHE_KEY.encode()).hexdigest()
_INSTALL_LOCK = threading.Lock()


def strip_memory_order_barriers(llvm_ir: str) -> str:
    """Remove warp-pipeline barriers that pin each memory operation in place."""
    return _MEMORY_ORDER_BARRIER.sub("", llvm_ir)


def restrict_workgroup_fences_to_lds(llvm_ir: str) -> str:
    """Tag workgroup acquire/release fences as synchronizing only LDS.

    Fences that already carry memory-model metadata keep it. The tag is LLVM's
    ``amdgpu-synchronize-as`` memory-model relaxation annotation (MMRA).
    """
    metadata_id = 1 + max((int(match) for match in _METADATA_ID.findall(llvm_ir)), default=-1)
    tagged = _WORKGROUP_FENCE.sub(
        lambda fence: (
            fence.group(0)
            if "!mmra " in fence.group(2)
            else f"{fence.group(1)}{fence.group(2)}, !mmra !{metadata_id}"
        ),
        llvm_ir,
    )
    if tagged == llvm_ir:
        return llvm_ir
    return f'{tagged.rstrip()}\n!{metadata_id} = !{{!"amdgpu-synchronize-as", !"local"}}\n'


_LLIR_REWRITES: tuple[tuple[str, Callable[[str], str]], ...] = (
    (_UNPINNED_WARP_PIPELINE_MARKER, strip_memory_order_barriers),
    (_LDS_ONLY_FENCE_MARKER, restrict_workgroup_fences_to_lds),
)


def _requested_llir_rewrites(options: object) -> tuple[Callable[[str], str], ...]:
    names = {name for name, _ in getattr(options, "llvm_fn_attrs", ())}
    return tuple(rewrite for marker, rewrite in _LLIR_REWRITES if marker in names)


def _rewrite_llir(
    make_llir: Callable[..., object], rewrites: tuple[Callable[[str], str], ...]
) -> Callable[[object, object], str]:
    def make_rewritten_llir(src: object, metadata: object) -> str:
        llvm_ir = cast(str, make_llir(src, metadata))
        for rewrite in rewrites:
            llvm_ir = rewrite(llvm_ir)
        return llvm_ir

    return make_rewritten_llir


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
        rewrites = _requested_llir_rewrites(options)
        if stages is not None and make_llir is not None and rewrites:
            stages["llir"] = _rewrite_llir(make_llir, rewrites)
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
if torch.version.hip is not None and (_PINS_WARP_PIPELINE_MEMORY or _FENCES_GLOBAL_MEMORY):
    _register()
