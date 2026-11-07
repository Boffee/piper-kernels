"""Install Piper Kernels' compiler passes into Inductor compile options.

Each ``add_*`` helper copies an options mapping and returns it with its passes in
``post_grad_custom_pre_pass``, keeping passes already installed and never adding
one twice. The input mapping is never mutated. ``compute_compile_pass_uuid``
supplies every pass's Inductor cache key.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from torch._inductor.codecache import build_code_hash
from torch._inductor.custom_graph_pass import CustomPassBase

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


def compute_compile_pass_uuid(compiler_pass: CustomPassBase) -> bytes:
    """Return the Inductor cache key for one of this package's compiler passes.

    Inductor's FX graph cache keys a compiled graph on each custom pass's
    ``uuid()``. A pass can depend at compile time on any package module, such as
    rewrite patterns, backend policies, or accelerator targets, so the key hashes
    every module's source with the pass's class path: editing any package source
    invalidates cached graphs, and no two pass classes share a key. The key omits
    instance state, so a pass must hold no configuration that changes its rewrite.
    """
    pass_class = type(compiler_pass)
    hasher = hashlib.sha256(_compute_package_source_hash())
    hasher.update(f"{pass_class.__module__}.{pass_class.__qualname__}".encode())
    return hasher.digest()


@functools.cache
def _compute_package_source_hash() -> bytes:
    """Hash every package module's name and source once per process.

    This is how ``torch_key()`` hashes torch's own source. This module sits at the
    package root, so its directory holds every package module.
    """
    hasher = hashlib.sha256()
    build_code_hash([str(Path(__file__).parent)], "piper_kernels.", hasher)
    return hasher.digest()


def add_post_grad_pass(
    options: Mapping[str, object] | None,
    compiler_pass: object,
) -> dict[str, object]:
    """Copy options and append one post-grad pass without mutating or duplicating."""
    combined = dict(options) if options is not None else {}
    existing = combined.get(_POST_GRAD_PRE_PASS)
    if existing is None:
        combined[_POST_GRAD_PRE_PASS] = compiler_pass
    elif existing is compiler_pass:
        pass
    elif isinstance(existing, (list, tuple)):
        if not any(item is compiler_pass for item in existing):
            combined[_POST_GRAD_PRE_PASS] = (*existing, compiler_pass)
    else:
        combined[_POST_GRAD_PRE_PASS] = (existing, compiler_pass)
    return combined


def add_first_post_grad_pass(
    options: Mapping[str, object] | None,
    compiler_pass: object,
) -> dict[str, object]:
    """Copy options and move or add one post-grad pass ahead of every other."""
    combined = dict(options) if options is not None else {}
    existing = combined.get(_POST_GRAD_PRE_PASS)
    if existing is None:
        passes: tuple[object, ...] = ()
    elif isinstance(existing, (list, tuple)):
        passes = tuple(existing)
    else:
        passes = (existing,)
    combined[_POST_GRAD_PRE_PASS] = (
        compiler_pass,
        *(other for other in passes if other is not compiler_pass),
    )
    return combined


def add_ordered_post_grad_passes(
    options: Mapping[str, object] | None,
    compiler_passes: Sequence[object],
) -> dict[str, object]:
    """Install an ordered pass group where its first existing member was located."""
    if not compiler_passes:
        return dict(options) if options is not None else {}
    combined = dict(options) if options is not None else {}
    existing = combined.get(_POST_GRAD_PRE_PASS)
    if existing is None:
        passes: tuple[object, ...] = ()
    elif isinstance(existing, (list, tuple)):
        passes = tuple(existing)
    else:
        passes = (existing,)

    anchor_index = next(
        (
            index
            for index, compiler_pass in enumerate(passes)
            if any(compiler_pass is grouped for grouped in compiler_passes)
        ),
        len(passes),
    )
    unrelated = tuple(
        compiler_pass
        for compiler_pass in passes
        if not any(compiler_pass is grouped for grouped in compiler_passes)
    )
    insertion_index = sum(
        not any(compiler_pass is grouped for grouped in compiler_passes)
        for compiler_pass in passes[:anchor_index]
    )
    combined[_POST_GRAD_PRE_PASS] = (
        *unrelated[:insertion_index],
        *compiler_passes,
        *unrelated[insertion_index:],
    )
    return combined


__all__ = [
    "add_first_post_grad_pass",
    "add_ordered_post_grad_passes",
    "add_post_grad_pass",
    "compute_compile_pass_uuid",
]
