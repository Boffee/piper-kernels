"""Convert packed GGUF weights directly into ConvRot NVFP4 storage."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.gguf._tensor import prepare_packed_matrix
from piper_kernels.linear.convrot._rotation import validate_group_size
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4.tensor import _MIN_PER_TENSOR_SCALE

type _Conversion = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


def convert(
    data: torch.Tensor,
    *,
    quant_type: int | None,
    logical_dtype: torch.dtype,
    group_size: int,
    per_tensor_scale: torch.Tensor | None,
    compute_per_tensor_scale: bool,
    is_swizzled_scales: bool,
    high_first: bool,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    per_tensor_scale_out: torch.Tensor | None = None,
) -> _Conversion:
    """Decode, rotate, and quantize one packed GGUF matrix."""
    validate_group_size(group_size)
    if logical_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("ConvRot NVFP4 GGUF logical dtype must be FP16 or BF16")
    if compute_per_tensor_scale and per_tensor_scale is not None:
        raise ValueError("NVFP4 from_gguf cannot both compute and receive a per-tensor scale")
    raw, normalized, rows, features = prepare_packed_matrix(data, quant_type)
    if features % group_size or features % nvfp4_layout.BLOCK_SIZE:
        raise ValueError(
            "GGUF in_features must be divisible by the ConvRot group and NVFP4 block sizes"
        )

    if raw.device.type != "cuda" or not AcceleratorTarget.from_device(
        raw.device
    ).is_cuda_capability(12, 0):
        raise ValueError("ConvRot NVFP4 GGUF conversion requires exact NVIDIA SM120")

    qdata_shape = (rows, features // 2)
    scale_shape = (
        tuple(nvfp4_layout.scale_shape(rows, features))
        if is_swizzled_scales
        else (rows, features // nvfp4_layout.BLOCK_SIZE)
    )
    if out is None:
        qdata = torch.empty(qdata_shape, device=raw.device, dtype=torch.uint8)
        scale = torch.empty(scale_shape, device=raw.device, dtype=torch.float8_e4m3fn)
    else:
        qdata, scale = out
        if (
            tuple(qdata.shape) != qdata_shape
            or qdata.dtype is not torch.uint8
            or qdata.device != raw.device
            or not qdata.is_contiguous()
            or tuple(scale.shape) != scale_shape
            or scale.dtype is not torch.float8_e4m3fn
            or scale.device != raw.device
            or not scale.is_contiguous()
        ):
            raise ValueError("ConvRot NVFP4 GGUF output storage is incompatible")
    if is_swizzled_scales and nvfp4_layout.has_scale_padding(rows, features):
        scale.zero_()

    from . import triton as convrot_nvfp4  # noqa: PLC0415

    if compute_per_tensor_scale:
        if per_tensor_scale_out is not None and (
            per_tensor_scale_out.shape != ()
            or per_tensor_scale_out.dtype is not torch.float32
            or per_tensor_scale_out.device != raw.device
            or not per_tensor_scale_out.is_contiguous()
        ):
            raise ValueError("NVFP4 per-tensor scale output must be a contiguous FP32 scalar")
        computed_scale = convrot_nvfp4._gguf_dynamic_scale(
            raw,
            int(normalized),
            group_size,
            logical_dtype,
            rows,
            features,
            out=per_tensor_scale_out,
        )
        computed_scale.clamp_min_(_MIN_PER_TENSOR_SCALE)
        per_tensor_scale = computed_scale
    elif per_tensor_scale is not None and (
        per_tensor_scale.shape != ()
        or per_tensor_scale.dtype is not torch.float32
        or per_tensor_scale.device != raw.device
        or not per_tensor_scale.is_contiguous()
    ):
        raise ValueError(
            "NVFP4 per-tensor scale must be a contiguous FP32 scalar on the input device"
        )

    convrot_nvfp4._gguf_prepare_out(
        raw,
        int(normalized),
        group_size,
        logical_dtype,
        per_tensor_scale,
        qdata,
        scale,
        is_swizzled_scales=is_swizzled_scales,
        high_first=high_first,
    )
    return qdata, scale, per_tensor_scale


__all__ = ["convert"]
