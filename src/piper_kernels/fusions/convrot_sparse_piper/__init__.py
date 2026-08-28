"""ConvRot projection fusion for sparse Piper attention."""

from __future__ import annotations

from collections.abc import Mapping


def convrot_sparse_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily install the ConvRot-to-sparse-Piper and ConvRot compiler passes."""
    from ._compile import convrot_sparse_piper_compile_options as compile_options  # noqa: PLC0415

    return compile_options(options)


__all__ = ["convrot_sparse_piper_compile_options"]
