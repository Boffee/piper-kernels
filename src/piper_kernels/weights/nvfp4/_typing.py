"""Structural typing for TorchAO's dynamically attributed NVFP4 wrapper."""

from __future__ import annotations

from typing import Protocol

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs


class NVFP4Storage(Protocol):
    """Runtime attributes supplied by TorchAO's NVFP4Tensor constructor."""

    qdata: torch.Tensor
    scale: torch.Tensor
    block_size: int
    orig_dtype: torch.dtype
    per_tensor_scale: torch.Tensor | None
    act_per_tensor_scale: torch.Tensor | None
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None


__all__ = ["NVFP4Storage"]
