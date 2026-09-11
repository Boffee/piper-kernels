"""Exact static and dynamic ConvRot NVFP4 activation preparation."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch

from piper_kernels._input_activations import input_activation_width
from piper_kernels._triton.convrot_nvfp4 import (
    _preparation_num_warps,
    _rotate_quantize_nvfp4_kernel,
    _rotated_row_amax_kernel,
    _rotation_chunk_sizes,
)
from piper_kernels._triton.nvfp4 import dynamic_scale as nvfp4_dynamic_scale
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.nvfp4._storage import prepare_activation_storage
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

_NVFP4_BLOCK_SIZE = nvfp4_layout.BLOCK_SIZE


type _ValidatedInput = tuple[
    torch.Tensor,
    int,
    int,
    tuple[int, int, int],
    AcceleratorTarget,
]


def _validate_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
) -> _ValidatedInput:
    validate_group_size(group_size)
    if input.ndim == 0 or input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("ConvRot NVFP4 input must be a non-scalar FP16, BF16, or FP32 tensor")
    source_features = int(input.shape[-1])
    if source_features < 1:
        raise ValueError("ConvRot NVFP4 requires a nonempty feature dimension")
    activation_width = input_activation_width(activation_fn)
    if source_features % activation_width:
        raise ValueError("ConvRot NVFP4 input activation requires equal feature partitions")
    input_features = source_features // activation_width
    rows = int(input.numel() // source_features)
    if rows < 1 or input_features % group_size or input_features % _NVFP4_BLOCK_SIZE:
        raise ValueError(
            "ConvRot NVFP4 requires nonempty rows divisible by the rotation and FP4 blocks"
        )
    if input.device.type != "cuda":
        raise ValueError("ConvRot NVFP4 currently requires CUDA")
    target = AcceleratorTarget.from_device(input.device)
    if not target.is_cuda_capability(12, 0):
        raise ValueError("ConvRot NVFP4 requires exact NVIDIA SM120")
    return (
        input.contiguous(),
        rows,
        input_features,
        _rotation_chunk_sizes(input_features, group_size),
        target,
    )


def _prepare_dynamic_scale(
    validated_input: _ValidatedInput,
    group_size: int,
    out: torch.Tensor | None = None,
    *,
    activation_fn: str | None = None,
) -> torch.Tensor:
    contiguous_input, rows, input_features, chunk_sizes, target = validated_input
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    chunk_size0, chunk_size1, chunk_size2 = chunk_sizes
    amax_num_warps, _ = _preparation_num_warps(chunk_sizes, group_size)
    row_amax = torch.empty(rows, device=contiguous_input.device, dtype=torch.float32)
    with device_context(contiguous_input.device):
        _rotated_row_amax_kernel[(rows,)](
            contiguous_input,
            row_amax,
            input_features,
            chunk_count=chunk_count,
            chunk_size0=chunk_size0,
            chunk_size1=chunk_size1,
            chunk_size2=chunk_size2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=activation_fn,
            accelerator_backend=target.backend,
            gguf_quant_type=-1,
            num_warps=amax_num_warps,
        )
        return nvfp4_dynamic_scale(row_amax, out=out)


def _prepare_static_storage(
    validated_input: _ValidatedInput,
    per_tensor_scale: torch.Tensor,
    group_size: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    contiguous_input, rows, input_features, chunk_sizes, target = validated_input
    if (
        per_tensor_scale.shape != ()
        or per_tensor_scale.dtype is not torch.float32
        or per_tensor_scale.device != contiguous_input.device
        or not per_tensor_scale.is_contiguous()
    ):
        raise ValueError("ConvRot NVFP4 scale must be a contiguous FP32 scalar on the input device")
    qdata, scale = prepare_activation_storage(
        contiguous_input,
        rows,
        input_features,
        out,
    )
    chunk_count = sum(chunk_size > 0 for chunk_size in chunk_sizes)
    chunk_size0, chunk_size1, chunk_size2 = chunk_sizes
    _, packing_num_warps = _preparation_num_warps(chunk_sizes, group_size)
    with device_context(contiguous_input.device):
        _rotate_quantize_nvfp4_kernel[(rows,)](
            contiguous_input,
            per_tensor_scale,
            qdata,
            scale,
            input_features,
            chunk_count=chunk_count,
            chunk_size0=chunk_size0,
            chunk_size1=chunk_size1,
            chunk_size2=chunk_size2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            scale_column_blocks=(input_features + nvfp4_layout.SCALE_COLUMN_TILE - 1)
            // nvfp4_layout.SCALE_COLUMN_TILE,
            activation_fn=activation_fn,
            accelerator_backend=target.backend,
            gguf_quant_type=-1,
            has_per_tensor_scale=True,
            swizzled_scales=True,
            high_first=high_first,
            num_warps=packing_num_warps,
        )
        return qdata, scale


def dynamic_scale(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate the exact dynamic scale after grouped rotation."""
    validated_input = _validate_input(input, group_size)
    return _prepare_dynamic_scale(validated_input, group_size, out)


def prepare_static(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    group_size: int,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then rotate and pack using a supplied scale."""
    validated_input = _validate_input(input, group_size, activation_fn)
    qdata, scale = _prepare_static_storage(
        validated_input,
        per_tensor_scale,
        group_size,
        activation_fn=activation_fn,
        high_first=high_first,
    )
    return qdata, scale, per_tensor_scale.clone()


def prepare_static_out(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    group_size: int,
    out: tuple[torch.Tensor, torch.Tensor],
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and pack into reusable caller-owned NVFP4 storage."""
    validated_input = _validate_input(input, group_size)
    return _prepare_static_storage(
        validated_input,
        per_tensor_scale,
        group_size,
        out,
        high_first=high_first,
    )


def prepare_dynamic(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then prepare exact dynamic ConvRot NVFP4 storage."""
    validated_input = _validate_input(input, group_size, activation_fn)
    per_tensor_scale = _prepare_dynamic_scale(
        validated_input,
        group_size,
        activation_fn=activation_fn,
    )
    qdata, scale = _prepare_static_storage(
        validated_input,
        per_tensor_scale,
        group_size,
        out,
        activation_fn=activation_fn,
        high_first=high_first,
    )
    return qdata, scale, per_tensor_scale


__all__ = [
    "dynamic_scale",
    "prepare_dynamic",
    "prepare_static",
    "prepare_static_out",
]
