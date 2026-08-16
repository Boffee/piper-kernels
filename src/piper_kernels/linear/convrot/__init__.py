"""Rotated quantized weights and functional linear operators."""

from collections.abc import Mapping

from .int8 import ConvRotInt8Tensor
from .int8.tensor import convrot_linear


def convrot_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily load Inductor options for automatic preparation sharing."""
    try:
        from ._compile import convrot_compile_options as compile_options  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "ConvRot compiler integration requires a compatible PyTorch Inductor "
            "custom graph-pass API"
        ) from error

    return compile_options(options)


__all__ = [
    "ConvRotInt8Tensor",
    "convrot_compile_options",
    "convrot_linear",
]
