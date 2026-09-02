"""MiniMax-H3 video VAE kernel specialization."""

from __future__ import annotations

from collections.abc import Mapping


def minimax_h3_vae_convrot_int8_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install the H3 VAE INT8 ConvRot schedule specialization."""
    from ._compile import (  # noqa: PLC0415
        minimax_h3_vae_convrot_int8_compile_options as compile_options,
    )

    return compile_options(options)


__all__ = ["minimax_h3_vae_convrot_int8_compile_options"]
