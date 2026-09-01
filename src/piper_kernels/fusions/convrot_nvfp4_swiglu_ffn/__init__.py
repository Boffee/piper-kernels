"""Chunked ConvRot NVFP4 SwiGLU feed-forward fusion."""

from __future__ import annotations

from collections.abc import Mapping


def convrot_nvfp4_swiglu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily install chunked FFN and ConvRot NVFP4 optimization."""
    from ._compile import (  # noqa: PLC0415
        convrot_nvfp4_swiglu_ffn_compile_options as compile_options,
    )

    return compile_options(options)


__all__ = ["convrot_nvfp4_swiglu_ffn_compile_options"]
