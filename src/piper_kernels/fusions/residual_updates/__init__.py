"""In-place residual updates for compiled blocks that overwrite their input."""

from __future__ import annotations

from collections.abc import Mapping


def residual_updates_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Make compiled blocks add their residual updates into an input they overwrite.

    Returns a copy of ``options`` with the rewrite installed after the post-grad
    passes already present; apply it after the attention and FFN fusion helpers. A
    block is rewritten only when its compiled program copies its result back into a
    graph input, as ``hidden.copy_(block(hidden, ...))`` does: that input is then
    updated in place, and the caller must not read it or any view of its storage
    afterward. A gated ConvRot INT8 SwiGLU FFN update and the attention update before
    it add into the input; a fused dense ConvRot INT8 Piper attention output does so
    without a full-size output. Other graphs are unchanged. Each update rounds the
    input to its dtype. The returned options also set ``allow_buffer_reuse`` to
    ``False``. See docs/attention.md#update-a-residual-stream-in-place for an example.
    """
    from ._compile import (  # noqa: PLC0415 - keep compiler integration lazy
        residual_updates_compile_options as compile_options,
    )

    return compile_options(options)


__all__ = ["residual_updates_compile_options"]
