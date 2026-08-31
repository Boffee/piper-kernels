"""NVFP4 projection fusion for sparse Piper attention."""

from collections.abc import Mapping


def nvfp4_sparse_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily load Inductor options for NVFP4 sparse-Piper fusion."""
    try:
        from ._compile import (  # noqa: PLC0415
            nvfp4_sparse_piper_compile_options as compile_options,
        )
    except ImportError as error:
        raise RuntimeError(
            "NVFP4 sparse-Piper compiler integration requires compatible PyTorch "
            "Inductor graph-pass and Triton APIs"
        ) from error
    return compile_options(options)


__all__ = ["nvfp4_sparse_piper_compile_options"]
