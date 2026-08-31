"""Validation shared by NVFP4 sparse-Piper projection operators."""

from __future__ import annotations

import math

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import HEAD_DIM, TILE_ROWS


def validate_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    chunk_rows: int,
    name: str,
) -> tuple[int, int]:
    """Validate one batch-one prepared NVFP4 projection and return length/heads."""
    if input_qdata.ndim != 2 or input_qdata.dtype is not torch.uint8:
        raise ValueError(f"{name} input must be a two-dimensional packed UINT8 tensor")
    sequence_length, packed_input_features = input_qdata.shape
    input_features = 2 * packed_input_features
    if sequence_length < TILE_ROWS or input_features % 16:
        raise ValueError(f"{name} input must contain K64 rows and block-16 features")
    expected_input_scale = (
        (sequence_length + 127) // 128 * 32,
        (input_features + 63) // 64 * 16,
    )
    if input_scale.shape != expected_input_scale or input_scale.dtype is not torch.float8_e4m3fn:
        raise ValueError(f"{name} input scale has an incompatible swizzled layout")
    if input_per_tensor_scale.shape != () or input_per_tensor_scale.dtype is not torch.float32:
        raise ValueError(f"{name} input per-tensor scale must be an FP32 scalar")
    if (
        weight_qdata.ndim != 2
        or weight_qdata.dtype is not torch.uint8
        or weight_qdata.shape[1] != packed_input_features
        or weight_qdata.shape[0] % HEAD_DIM
    ):
        raise ValueError(f"{name} weight must map to complete packed D128 heads")
    output_features = weight_qdata.shape[0]
    expected_weight_scale = (
        (output_features + 127) // 128 * 32,
        (input_features + 63) // 64 * 16,
    )
    if weight_scale.shape != expected_weight_scale or weight_scale.dtype is not torch.float8_e4m3fn:
        raise ValueError(f"{name} weight scale has an incompatible swizzled layout")
    if weight_per_tensor_scale is not None and (
        weight_per_tensor_scale.shape != () or weight_per_tensor_scale.dtype is not torch.float32
    ):
        raise ValueError(f"{name} weight per-tensor scale must be an FP32 scalar")
    if bias is not None and (bias.shape != (output_features,) or bias.dtype is not torch.bfloat16):
        raise ValueError(f"{name} bias must be one BF16 value per output feature")
    if chunk_rows < 128 or chunk_rows % 128:
        raise ValueError(f"{name} chunk rows must be a positive multiple of 128")
    operands = [
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
    ]
    operands.extend(operand for operand in (weight_per_tensor_scale, bias) if operand is not None)
    if any(operand.device != input_qdata.device for operand in operands):
        raise ValueError(f"{name} operands must share a device")
    if any(not operand.is_contiguous() for operand in operands):
        raise ValueError(f"{name} operands must be contiguous")
    if not AcceleratorTarget.from_device(input_qdata.device).is_cuda_capability(12, 0):
        raise ValueError(f"{name} fusion requires exact NVIDIA SM120")
    return sequence_length, output_features // HEAD_DIM


def validate_qk_epilogue(
    input_qdata: torch.Tensor,
    sequence_length: int,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    name: str,
) -> None:
    """Validate the RMSNorm/RoPE inputs shared by Q and K epilogues."""
    rotary_dim = cos.shape[1] if cos.ndim == 2 else 0
    operands = (norm_weight, cos, sin)
    if (
        norm_weight.shape != (HEAD_DIM,)
        or norm_weight.dtype is not torch.bfloat16
        or cos.ndim != 2
        or sin.shape != cos.shape
        or cos.shape[0] != sequence_length
        or cos.dtype is not torch.float32
        or sin.dtype is not torch.float32
        or not 2 <= rotary_dim <= HEAD_DIM
        or rotary_dim % 2
        or any(operand.device != input_qdata.device for operand in operands)
        or any(not operand.is_contiguous() for operand in operands)
    ):
        raise ValueError(f"{name} requires contiguous BF16 norm and FP32 split-half RoPE")
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError(f"{name} norm epsilon must be finite and positive")


__all__ = ["validate_projection", "validate_qk_epilogue"]
