"""Portable reference implementation for INT8 ConvRot linear layers."""

import torch

from .._rotation import rotate_groups, validate_group_size


def validate_storage(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> None:
    """Validate INT8 ConvRot storage and its logical floating-point dtype."""
    validate_group_size(group_size)
    if qdata.dtype is not torch.int8 or qdata.ndim != 2:
        raise ValueError(
            f"ConvRot INT8 qdata must be a 2-D int8 tensor, got {qdata.dtype} {qdata.shape}"
        )
    if qdata.shape[1] % group_size:
        raise ValueError(
            f"ConvRot in_features {qdata.shape[1]} is not divisible by group size {group_size}"
        )
    if scale.dtype is not torch.float32 or scale.numel() != qdata.shape[0]:
        raise ValueError(
            "ConvRot INT8 scale must be float32 with one element per output channel, "
            f"got {scale.dtype} {tuple(scale.shape)} for qdata {tuple(qdata.shape)}"
        )
    if scale.device != qdata.device:
        raise ValueError(
            "ConvRot INT8 qdata and scale must share a device, "
            f"got {qdata.device}/{scale.device}"
        )
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"ConvRot logical dtype must be floating point, got {dtype}")


def dynamic_quantize_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize each row to signed INT8 with a float32 scale."""
    scale = (value.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    qdata = (value / scale.to(value.dtype)).round().clamp(-128, 127).to(torch.int8)
    return qdata, scale


def reference_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the portable PyTorch ConvRot W8A8 linear implementation."""
    original_shape = activation.shape
    activation_2d = activation.reshape(-1, original_shape[-1])
    rotated = rotate_groups(activation_2d, group_size)
    activation_qdata, activation_scale = dynamic_quantize_rows(rotated)
    if activation.device.type == "cpu":
        accumulated = activation_qdata.to(torch.int32) @ qdata.T.to(torch.int32)
    else:
        # Float32 represents each INT8 product exactly. Only very long reductions
        # can round the integer sum, which is preferable to rejecting the shape.
        accumulated = activation_qdata.float() @ qdata.T.float()
    result = (
        accumulated.to(torch.float32)
        * activation_scale.to(torch.float32)
        * scale.reshape(1, -1).to(torch.float32)
    )
    if bias is not None:
        result += bias.to(torch.float32)
    return result.to(activation.dtype).reshape(*original_shape[:-1], qdata.shape[0])
