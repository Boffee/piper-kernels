"""Static-scale ConvRot INT8 3x3x3 convolutions and causal convenience APIs."""

from collections.abc import Mapping

from ._functional import conv3d, group_norm_silu_conv3d
from ._padding import SpatialPadding
from .module import ConvRotInt8Conv3d


def convrot_int8_conv3d_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Lazily load Inductor options for ConvRot INT8 convolution padding fusion."""
    try:
        from ._compile import (  # noqa: PLC0415
            convrot_int8_conv3d_compile_options as compile_options,
        )
    except ImportError as error:
        raise RuntimeError(
            "ConvRot compiler integration requires a compatible PyTorch Inductor "
            "custom graph-pass API"
        ) from error

    return compile_options(options)


__all__ = [
    "ConvRotInt8Conv3d",
    "SpatialPadding",
    "conv3d",
    "convrot_int8_conv3d_compile_options",
    "group_norm_silu_conv3d",
]
