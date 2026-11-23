"""Allocation of prepared NVFP4 activation storage."""

from typing import cast

import torch

from piper_kernels._input_activations import input_activation_width
from piper_kernels.weights.nvfp4._layout import has_scale_padding, qdata_shape, scale_shape


def prepared_input_metadata(
    input: torch.Tensor,  # noqa: A002
    activation_per_tensor_scale: torch.Tensor | None,
    activation_fn: str | None,
    *,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe the packed data, block scales, and global scale after activation."""
    features = input.shape[-1] // input_activation_width(activation_fn)
    if group_size is not None and isinstance(features, int) and features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 input features {features} must be divisible by group size {group_size}"
        )
    rows = input.numel() // input.shape[-1]
    return (
        input.new_empty(qdata_shape(rows, features), dtype=torch.uint8),
        input.new_empty(scale_shape(rows, features), dtype=torch.float8_e4m3fn),
        activation_per_tensor_scale.new_empty(())
        if activation_per_tensor_scale is not None
        else input.new_empty((), dtype=torch.float32),
    )


def prepare_activation_storage(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    rows: int,
    features: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    initialize_padding: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate or view canonical storage; callers may initialize padding in their kernel."""
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
    if initialize_padding and has_scale_padding(rows, features):
        scale.zero_()
    return qdata, scale
