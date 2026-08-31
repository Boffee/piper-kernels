"""Chunked NVFP4 SwiGLU feed-forward fusion."""

from __future__ import annotations

from collections.abc import Mapping


def nvfp4_swiglu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily install chunked SwiGLU FFN and ordinary NVFP4 optimization."""
    from ._compile import nvfp4_swiglu_ffn_compile_options as compile_options  # noqa: PLC0415

    return compile_options(options)


__all__ = ["nvfp4_swiglu_ffn_compile_options"]
