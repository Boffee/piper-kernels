"""In-place residual updates for compiled blocks that overwrite their input."""

from __future__ import annotations

from collections.abc import Mapping


def residual_updates_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Make compiled blocks add their residual updates into an input they overwrite.

    Returns a copy of ``options`` with the rewrite installed ahead of every other
    post-grad pass, so it may be applied before or after the attention and FFN fusion
    helpers. A block is rewritten only when its compiled program copies its result
    back into a graph input on a device Triton supports, as
    ``hidden.copy_(block(hidden, ...))`` does, and builds that result from the input
    through gated sums ``base + gate[indices] * update``: that input is then updated
    in place, and the caller must not read it or any view of its storage afterward.
    ``gate`` is two-dimensional, ``indices`` selects one of its rows for each input row
    (``gate.index_select(0, indices)`` also works), and ``update`` is contiguous and
    matches the input. Other blocks are unchanged, including those whose gates
    broadcast rather than being selected per row, or with any sum that doesn't
    qualify.

    Each sum becomes an add into the input, whatever produced its update. When the
    update is an output of an FFN fusion, or of ConvRot INT8 Piper attention fused
    with its output projection (dense with ``fuse_output=True``, or sparse), and that
    fusion's helper is installed, the fusion computes its output inside the add
    without a full-size output. Inductor adds an update in place only when the block
    reads none of the input's original values afterward, through any view; otherwise
    it copies the input first, which forgoes that update's saving. Each update rounds
    the input to its dtype. The returned options also set ``allow_buffer_reuse`` to
    ``False``. See docs/attention.md#update-a-residual-stream-in-place for an example.
    """
    from ._compile import (  # noqa: PLC0415 - keep compiler integration lazy
        residual_updates_compile_options as compile_options,
    )

    return compile_options(options)


__all__ = ["residual_updates_compile_options"]
