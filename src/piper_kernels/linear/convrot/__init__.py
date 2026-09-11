"""ConvRot linear operators and compiler integration."""

from collections.abc import Mapping

from .int8._functional import convrot_int8_linear


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
    "convrot_int8_compile_options",
    "convrot_int8_linear",
]
