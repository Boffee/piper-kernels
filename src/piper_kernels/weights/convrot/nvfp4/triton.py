"""Direct GGUF conversion into ConvRot NVFP4 weight storage."""

# pyright: reportCallIssue=false
from __future__ import annotations

import torch

from piper_kernels._triton import nvfp4 as nvfp4_backend
from piper_kernels._triton.convrot_nvfp4 import (
    _preparation_num_warps,
    _rotate_quantize_nvfp4_kernel,
    _rotated_row_amax_kernel,
    _rotation_chunk_sizes,
)
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout


def _gguf_dynamic_scale(
    data: torch.Tensor,
    quant_type: int,
    group_size: int,
    rows: int,
    row_width: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode GGUF rows through the existing post-rotation amax kernel."""
    chunk_sizes = _rotation_chunk_sizes(row_width, group_size)
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    chunk_size0, chunk_size1, chunk_size2 = chunk_sizes
    amax_num_warps, _ = _preparation_num_warps(chunk_sizes, group_size)
    row_amax = torch.empty(rows, device=data.device, dtype=torch.float32)
    target = AcceleratorTarget.from_device(data.device)
    with device_context(data.device):
        _rotated_row_amax_kernel[(rows,)](
            data,
            row_amax,
            row_width,
            chunk_count=chunk_count,
            chunk_size0=chunk_size0,
            chunk_size1=chunk_size1,
            chunk_size2=chunk_size2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=None,
            accelerator_backend=target.backend,
            gguf_quant_type=quant_type,
            num_warps=amax_num_warps,
        )
        return nvfp4_backend.dynamic_scale(row_amax, out=out)


def _gguf_prepare_out(
    data: torch.Tensor,
    quant_type: int,
    group_size: int,
    per_tensor_scale: torch.Tensor | None,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    is_swizzled_scales: bool,
    high_first: bool,
) -> None:
    """Decode GGUF rows through the existing ConvRot NVFP4 packing kernel."""
    rows, packed_width = qdata.shape
    row_width = packed_width * 2
    chunk_sizes = _rotation_chunk_sizes(row_width, group_size)
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    chunk_size0, chunk_size1, chunk_size2 = chunk_sizes
    _, packing_num_warps = _preparation_num_warps(chunk_sizes, group_size)
    target = AcceleratorTarget.from_device(data.device)
    with device_context(data.device):
        _rotate_quantize_nvfp4_kernel[(rows,)](
            data,
            per_tensor_scale if per_tensor_scale is not None else data,
            qdata,
            scale,
            row_width,
            chunk_count=chunk_count,
            chunk_size0=chunk_size0,
            chunk_size1=chunk_size1,
            chunk_size2=chunk_size2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            scale_column_blocks=(row_width + nvfp4_layout.SCALE_COLUMN_TILE - 1)
            // nvfp4_layout.SCALE_COLUMN_TILE,
            activation_fn=None,
            accelerator_backend=target.backend,
            gguf_quant_type=quant_type,
            has_per_tensor_scale=per_tensor_scale is not None,
            swizzled_scales=is_swizzled_scales,
            high_first=high_first,
            num_warps=packing_num_warps,
        )
