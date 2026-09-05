"""Chunked ConvRot INT8 SwiGLU feed-forward fusion."""

from __future__ import annotations

from collections.abc import Mapping


def convrot_int8_swiglu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily install chunked SwiGLU FFN and ordinary ConvRot INT8 optimization."""
    from ._compile import (  # noqa: PLC0415 - keep compiler integration lazy
        convrot_int8_swiglu_ffn_compile_options as compile_options,
    )

    return compile_options(options)


__all__ = ["convrot_int8_swiglu_ffn_compile_options"]
