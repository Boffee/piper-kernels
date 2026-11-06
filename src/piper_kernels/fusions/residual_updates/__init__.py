"""In-place residual updates for compiled blocks that overwrite their input."""

from __future__ import annotations

from collections.abc import Mapping


def residual_updates_compile_options(
    options: Mapping[str, object] | None = None,
    *,
    attention_head_groups: int = 1,
) -> dict[str, object]:
    """Make compiled blocks add their residual updates into an input they overwrite.

    Returns a copy of ``options`` with the rewrite installed after the post-grad
    passes already present; apply it after the attention and FFN fusion helpers. A
    block is rewritten only when its compiled program copies its result back into a
    graph input, as ``hidden.copy_(block(hidden, ...))`` does: that input is then
    updated in place, and the caller must not read it or any view of its storage
    afterward. A gated ConvRot INT8 SwiGLU FFN update and the attention update before
    it add into the input; a fused ConvRot INT8 Piper attention output, dense or
    sparse with fused Q projection, does so without a full-size output. Inductor adds
    an update in place only when the block reads none of the input's original values
    afterward, through any view; otherwise it copies the input first, which forgoes
    that update's saving. Other graphs are unchanged. Each update rounds the input to
    its dtype. The returned options also set ``allow_buffer_reuse`` to ``False``. See
    docs/attention.md#update-a-residual-stream-in-place for an example.

    With ``attention_head_groups`` above 1, a dense fused attention whose K and V feed
    only it runs that many head groups in turn, each projecting and freeing its own K and
    V, so only one group's K and V are ever allocated. The split must divide the query
    and K/V heads and give each group a multiple of the output projection's ConvRot
    group size; an attention it doesn't fit stays ungrouped with a logged warning.
    Grouping changes outputs slightly: each group's output-projection input gets its own
    per-token scales, and the input rounds once per group.
    """
    from ._compile import (  # noqa: PLC0415 - keep compiler integration lazy
        residual_updates_compile_options as compile_options,
    )

    return compile_options(options, attention_head_groups=attention_head_groups)


__all__ = ["residual_updates_compile_options"]
