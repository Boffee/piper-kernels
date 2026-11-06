"""Install Piper Kernels' compiler passes into Inductor compile options.

Each helper copies an options mapping and returns it with its passes in
``post_grad_custom_pre_pass``, keeping passes already installed and never adding
one twice. The input mapping is never mutated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


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


__all__ = ["add_first_post_grad_pass", "add_ordered_post_grad_passes", "add_post_grad_pass"]
