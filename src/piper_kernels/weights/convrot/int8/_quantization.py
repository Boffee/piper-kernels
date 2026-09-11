"""Portable ConvRot INT8 weight quantization and storage validation."""

import torch

from piper_kernels._stochastic_quantization import stochastic_round_to_int
from piper_kernels.weights.convrot._rotation import rotate_groups, validate_group_size

_SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _rotate_weight_groups(value: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate FP32 convolution weights without a dense CPU matmul."""
    if value.device.type != "cpu":
        return rotate_groups(value, group_size)

    features = value.shape[-1]
    rotated = value.reshape(-1, features // group_size, group_size)
    stride = 1
    while stride < group_size:
        stage = rotated.reshape(
            *rotated.shape[:-1],
            group_size // (4 * stride),
            4,
            stride,
        )
        first, second, third, fourth = stage.unbind(-2)
        rotated = torch.stack(
            (
                first + second + third - fourth,
                first + second - third + fourth,
                first - second + third + fourth,
                -first + second + third + fourth,
            ),
            dim=-2,
        ).reshape(rotated.shape)
        stride *= 4
    return rotated.mul_(group_size**-0.5).reshape(value.shape)


def quantize_weight(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a linear or OI-DHW convolution weight per output channel."""
    validate_group_size(group_size)
    if weight.ndim not in (2, 5):
        raise ValueError(
            "ConvRot INT8 high-precision weight must be 2-D or 5-D, "
            f"got shape {tuple(weight.shape)}"
        )
    if weight.dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            "ConvRot INT8 high-precision weight must use float16, bfloat16, or float32, "
            f"got {weight.dtype}"
        )
    if weight.layout is not torch.strided:
        raise ValueError("ConvRot INT8 high-precision weight must use strided layout")
    if weight.device.type == "meta":
        raise ValueError("ConvRot INT8 cannot quantize a meta tensor without values")
    if weight.shape[1] % group_size:
        raise ValueError(
            f"ConvRot in_features {weight.shape[1]} is not divisible by group size {group_size}"
        )
    if weight.ndim == 5:
        if tuple(weight.shape[2:]) != (3, 3, 3) or weight.shape[0] <= 0:
            raise ValueError("ConvRot INT8 Conv3D weight must have shape [out, in, 3, 3, 3]")
        _validate_conv3d_channels(weight.shape[1])
        channels_last = weight.permute(0, 2, 3, 4, 1).to(
            dtype=torch.float32, memory_format=torch.contiguous_format
        )
        rotated = _rotate_weight_groups(channels_last, group_size)
        qdata, scale = dynamic_quantize_rows(rotated.flatten(1))
        return qdata.view_as(rotated), scale
    return dynamic_quantize_rows(rotate_groups(weight.float(), group_size))


def validate_storage(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> None:
    """Validate INT8 ConvRot storage and its logical floating-point dtype."""
    validate_group_size(group_size)
    if qdata.dtype is not torch.int8 or qdata.ndim not in (2, 5):
        raise ValueError(
            f"ConvRot INT8 qdata must be a 2-D or 5-D int8 tensor, got {qdata.dtype} {qdata.shape}"
        )
    if qdata.ndim == 5:
        if tuple(qdata.shape[1:4]) != (3, 3, 3) or qdata.shape[0] <= 0:
            raise ValueError("ConvRot INT8 Conv3D qdata must have shape [out, 3, 3, 3, in]")
        _validate_conv3d_channels(qdata.shape[-1])
    if qdata.shape[-1] % group_size:
        raise ValueError(
            f"ConvRot in_features {qdata.shape[-1]} is not divisible by group size {group_size}"
        )
    expected_scale_shape = (qdata.shape[0], 1)
    if scale.dtype is not torch.float32 or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"ConvRot INT8 scale must be float32 with shape {expected_scale_shape}, "
            f"got {scale.dtype} {tuple(scale.shape)} for qdata {tuple(qdata.shape)}"
        )
    if scale.device != qdata.device:
        raise ValueError(
            f"ConvRot INT8 qdata and scale must share a device, got {qdata.device}/{scale.device}"
        )
    if qdata.layout is not torch.strided or scale.layout is not torch.strided:
        raise ValueError("ConvRot INT8 weight storage must use strided layout")
    if not qdata.is_contiguous() or not scale.is_contiguous():
        raise ValueError(
            "ConvRot INT8 qdata and scale must be contiguous; "
            "use from_quantized to canonicalize storage"
        )
    if dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            f"ConvRot logical dtype must be float16, bfloat16, or float32, got {dtype}"
        )


def _validate_conv3d_channels(channels: int) -> None:
    # Power-of-two preparation tiles and a bounded 27*C INT32 reduction.
    if channels < 64 or channels > 4096 or channels & (channels - 1):
        raise ValueError("ConvRot INT8 Conv3D requires power-of-two input channels in [64, 4096]")


def validate_activation_scale(scale: torch.Tensor | None, device: torch.device) -> None:
    """Validate optional static activation storage without reading device values."""
    if scale is None:
        return
    if scale.ndim != 0 or scale.dtype is not torch.float32 or scale.layout is not torch.strided:
        raise ValueError("ConvRot INT8 activation scale must be a strided FP32 scalar")
    if scale.device != device:
        raise ValueError("ConvRot INT8 activation scale must share the weight device")
    if scale.requires_grad:
        raise ValueError("ConvRot INT8 activation scale must not require gradients")


def dequantize_weight(
    qdata: torch.Tensor, scale: torch.Tensor, group_size: int, output_dtype: torch.dtype
) -> torch.Tensor:
    """Restore the logical layout using FP32 rescaling and inverse rotation."""
    if output_dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError("ConvRot INT8 dequantization output dtype must be floating point")
    rotated = qdata.float() * scale.reshape(-1, *((1,) * (qdata.ndim - 1)))
    if qdata.ndim == 5:
        result = _rotate_weight_groups(rotated, group_size).permute(0, 4, 1, 2, 3)
    else:
        result = rotate_groups(rotated, group_size)
    return result.to(dtype=output_dtype, memory_format=torch.contiguous_format)


def dynamic_quantize_rows(
    value: torch.Tensor,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize each row to signed INT8 with a float32 scale."""
    scale = (value.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    scaled = value.float() / scale
    qdata = scaled.round().clamp(-128, 127).to(torch.int8)
    if rounding_seed is not None:
        stochastic_scaled = value.to(torch.float32) / scale
        qdata = stochastic_round_to_int(
            stochastic_scaled,
            seed=rounding_seed,
            quant_min=-128,
            quant_max=127,
            deterministic=qdata,
        ).to(torch.int8)
    return qdata, scale
