"""NVFP4 tensors and inference graph optimizations."""

from collections.abc import Mapping

from .tensor import PiperNVFP4Tensor


def nvfp4_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily load Inductor options for NVFP4 preparation sharing."""
    try:
        from ._compile import nvfp4_compile_options as compile_options  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "NVFP4 compiler integration requires a compatible PyTorch Inductor "
            "custom graph-pass API"
        ) from error

    return compile_options(options)


__all__ = ["PiperNVFP4Tensor", "nvfp4_compile_options"]
