"""Portable ConvRot INT8 weight quantization and storage validation."""

import torch

from piper_kernels._stochastic_quantization import stochastic_round_to_int
from piper_kernels.weights.convrot._rotation import rotate_groups, validate_group_size

_SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def quantize_weight(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and quantize a dense two-dimensional weight per output row."""
    validate_group_size(group_size)
    if weight.ndim != 2:
        raise ValueError(
            f"ConvRot INT8 high-precision weight must be 2-D, got shape {tuple(weight.shape)}"
        )
    if weight.dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            "ConvRot INT8 high-precision weight must use float16, bfloat16, or float32, "
            f"got {weight.dtype}"
        )
    if weight.device.type == "meta":
        raise ValueError("ConvRot INT8 cannot quantize a meta tensor without values")
    if weight.shape[1] % group_size:
        raise ValueError(
            f"ConvRot in_features {weight.shape[1]} is not divisible by group size {group_size}"
        )
    return dynamic_quantize_rows(rotate_groups(weight.float(), group_size))


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
    if not qdata.is_contiguous() or not scale.is_contiguous():
        raise ValueError(
            "ConvRot INT8 qdata and scale must be contiguous; "
            "use from_quantized to canonicalize storage"
        )
    if dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            f"ConvRot logical dtype must be float16, bfloat16, or float32, got {dtype}"
        )


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
