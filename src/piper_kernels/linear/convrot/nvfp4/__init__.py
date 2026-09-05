"""NVFP4 tensors and activation preparation in the ConvRot basis."""

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING

from .tensor import ConvRotNVFP4Tensor, convrot_nvfp4_linear

if TYPE_CHECKING:
    from .triton import dynamic_scale, prepare_dynamic, prepare_static, prepare_static_out


def __getattr__(name: str) -> object:
    """Load optimized preparation only when its public entry points are requested."""
    if name in ("dynamic_scale", "prepare_dynamic", "prepare_static", "prepare_static_out"):
        backend = import_module(".triton", __name__)
        value = getattr(backend, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def convrot_nvfp4_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily load Inductor options for ConvRot NVFP4 preparation sharing."""
    try:
        from ._compile import convrot_nvfp4_compile_options as compile_options  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "ConvRot NVFP4 compiler integration requires a compatible PyTorch "
            "Inductor custom graph-pass API"
        ) from error

    return compile_options(options)


__all__ = [
    "ConvRotNVFP4Tensor",
    "convrot_nvfp4_compile_options",
    "convrot_nvfp4_linear",
    "dynamic_scale",
    "prepare_dynamic",
    "prepare_static",
    "prepare_static_out",
]
