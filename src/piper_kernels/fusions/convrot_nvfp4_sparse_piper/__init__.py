"""ConvRot NVFP4 projection fusion for sparse Piper attention."""

from collections.abc import Mapping


def convrot_nvfp4_sparse_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily install ConvRot NVFP4 sparse-Piper compiler passes."""
    try:
        from ._compile import (  # noqa: PLC0415
            convrot_nvfp4_sparse_piper_compile_options as compile_options,
        )
    except ImportError as error:
        raise RuntimeError(
            "ConvRot NVFP4 sparse-Piper compiler integration requires compatible "
            "PyTorch Inductor graph-pass and Triton APIs"
        ) from error
    return compile_options(options)


__all__ = ["convrot_nvfp4_sparse_piper_compile_options"]
