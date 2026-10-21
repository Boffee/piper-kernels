"""Explicit convolution padding, in PyTorch F.pad order (W, H, T).

Temporal padding always supplies zeros. Spatial padding supplies zeros or
reflection, with each reflection amount smaller than its input dimension.
These are execution parameters, independent of model identity.
"""

from typing import Literal

type Padding = tuple[int, int, int, int, int, int]
type SpatialPadding = Literal["none", "reflect", "reflect_right"]


def causal_padding(padding: SpatialPadding) -> tuple[Padding, bool]:
    """Resolve the legacy causal API to explicit spatial and temporal padding."""
    if padding == "none":
        return (0, 0, 0, 0, 2, 0), False
    if padding == "reflect":
        return (1, 1, 1, 1, 2, 0), True
    if padding == "reflect_right":
        return (0, 1, 0, 1, 2, 0), True
    raise ValueError(f"unsupported ConvRot INT8 Conv3D spatial padding mode {padding!r}")
