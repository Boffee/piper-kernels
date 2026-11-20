"""Linear operators and activation preparation in the ConvRot NVFP4 basis."""

from importlib import import_module
from typing import TYPE_CHECKING

from ._functional import convrot_nvfp4_linear

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


__all__ = [
    "convrot_nvfp4_linear",
    "dynamic_scale",
    "prepare_dynamic",
    "prepare_static",
    "prepare_static_out",
]
