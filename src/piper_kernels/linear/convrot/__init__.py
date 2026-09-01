"""Rotated quantized weights and functional linear operators."""

from collections.abc import Mapping

from ._rotation import SUPPORTED_GROUP_SIZES
from .int8 import ConvRotInt8Tensor
from .int8.tensor import convrot_int8_linear


def convrot_int8_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily load Inductor options for ConvRot INT8 inference optimization."""
    try:
        from .int8._compile import convrot_int8_compile_options as compile_options  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "ConvRot compiler integration requires a compatible PyTorch Inductor "
            "custom graph-pass API"
        ) from error

    return compile_options(options)


__all__ = [
    "SUPPORTED_GROUP_SIZES",
    "ConvRotInt8Tensor",
    "convrot_int8_compile_options",
    "convrot_int8_linear",
]
