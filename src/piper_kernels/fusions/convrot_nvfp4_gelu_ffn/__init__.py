"""Chunked standard/ConvRot NVFP4 GELU feed-forward fusion."""

from __future__ import annotations

from collections.abc import Mapping


def convrot_nvfp4_gelu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily install mixed ConvRot NVFP4 GELU FFN optimization."""
    from ._compile import (  # noqa: PLC0415 - keep compiler integration lazy
        convrot_nvfp4_gelu_ffn_compile_options as compile_options,
    )

    return compile_options(options)


__all__ = ["convrot_nvfp4_gelu_ffn_compile_options"]
