"""Bounded-workspace feed-forward network fusions across quantized formats."""

from __future__ import annotations

from collections.abc import Mapping


def ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install FFN folding before ordinary quantized-linear normalization passes.

    Folds FP16/BF16 ConvRot INT8, NVFP4, and ConvRot NVFP4 SwiGLU and exclusive
    tanh-GELU FFNs (including indexed gated updates and in-place residual adds)
    while importing format backends lazily based on what appears in the graph.
    Existing post-grad passes are preserved and reapplication is idempotent; the
    input mapping is not mutated.
    """
    from ._compile import ffn_compile_options as compile_options  # noqa: PLC0415

    return compile_options(options)


__all__ = ["ffn_compile_options"]
