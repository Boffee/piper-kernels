"""Allocation of prepared NVFP4 activation storage."""

from typing import cast

import torch

from piper_kernels.weights.nvfp4._layout import has_scale_padding, qdata_shape, scale_shape


def prepare_activation_storage(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    rows: int,
    features: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate or view caller-owned canonical activation storage."""
    qdata_dimensions = cast(tuple[int, int], qdata_shape(rows, features))
    scale_dimensions = cast(tuple[int, int], scale_shape(rows, features))
    scale_elements = scale_dimensions[0] * scale_dimensions[1]
    if out is None:
        qdata = torch.empty(qdata_dimensions, device=input.device, dtype=torch.uint8)
        scale = torch.empty(scale_dimensions, device=input.device, dtype=torch.float8_e4m3fn)
    else:
        qdata_storage, scale_storage = out
        if (
            qdata_storage.ndim != 2
            or qdata_storage.shape[0] < rows
            or qdata_storage.shape[1] != qdata_dimensions[1]
            or qdata_storage.dtype is not torch.uint8
            or scale_storage.numel() < scale_elements
            or scale_storage.dtype is not torch.float8_e4m3fn
            or qdata_storage.device != input.device
            or scale_storage.device != input.device
            or not qdata_storage.is_contiguous()
            or not scale_storage.is_contiguous()
        ):
            raise ValueError("NVFP4 activation storage is incompatible")
        qdata = qdata_storage[:rows]
        scale = scale_storage.flatten()[:scale_elements].view(scale_dimensions)
    if has_scale_padding(rows, features):
        scale.zero_()
    return qdata, scale
