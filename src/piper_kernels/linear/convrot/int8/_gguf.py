"""Convert packed GGUF weights directly into ConvRot INT8 storage."""

from __future__ import annotations

import torch

from piper_kernels.gguf._tensor import prepare_packed_matrix
from piper_kernels.linear.convrot._rotation import validate_group_size

from . import _backend

_SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def convert(
    data: torch.Tensor,
    *,
    quant_type: int | None,
    group_size: int,
    logical_dtype: torch.dtype,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode, rotate, and quantize one packed GGUF matrix."""
    validate_group_size(group_size)
    if logical_dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            f"ConvRot INT8 GGUF logical dtype must be FP16, BF16, or FP32, got {logical_dtype}"
        )
    raw, normalized, rows, features = prepare_packed_matrix(data, quant_type)
    convert_out = _backend.select_gguf_converter(raw)
    if convert_out is None:
        raise ValueError(
            "ConvRot INT8 GGUF conversion requires a supported Triton accelerator; "
            f"conversion is unavailable on {raw.device}"
        )
    if features % group_size:
        raise ValueError(
            f"GGUF in_features {features} is not divisible by ConvRot group size {group_size}"
        )

    expected_qdata = (rows, features)
    expected_scale = (rows, 1)
    if out is None:
        qdata = torch.empty(expected_qdata, device=raw.device, dtype=torch.int8)
        scale = torch.empty(expected_scale, device=raw.device, dtype=torch.float32)
    else:
        qdata, scale = out
        if (
            tuple(qdata.shape) != expected_qdata
            or qdata.dtype is not torch.int8
            or qdata.device != raw.device
            or not qdata.is_contiguous()
            or tuple(scale.shape) != expected_scale
            or scale.dtype is not torch.float32
            or scale.device != raw.device
            or not scale.is_contiguous()
        ):
            raise ValueError("ConvRot INT8 GGUF output storage is incompatible")

    if rows == 0 or features == 0:
        scale.fill_(1e-30)
    else:
        convert_out(
            raw,
            int(normalized),
            group_size,
            qdata,
            scale,
        )
    return qdata, scale


__all__ = ["convert"]
