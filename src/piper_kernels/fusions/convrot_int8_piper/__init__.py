"""ConvRot INT8 query projection fusion for dense Piper attention."""

from __future__ import annotations

from collections.abc import Mapping


def convrot_int8_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install dense query fusion before ordinary ConvRot INT8 compiler passes."""
    from ._compile import (  # noqa: PLC0415 - keep compiler integration lazy
        convrot_int8_piper_compile_options as compile_options,
    )

    return compile_options(options)


__all__ = ["convrot_int8_piper_compile_options"]
