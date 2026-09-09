"""Tensor validation for ConvRot INT8 Q/K projection, independent of Triton."""

import math

import torch

_SUPPORTED_HEAD_DIMS = (64, 128)
_SUPPORTED_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_qk_projection_inputs(  # noqa: PLR0912
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
    name: str,
) -> tuple[int, int, int]:
    """Validate inputs to a fused ConvRot INT8 Q/K projection kernel."""
    head_dim = norm_weight.shape[0] if norm_weight.ndim == 1 else 0
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"{name} projection RMSNorm weight must be a D64/D128 vector")
    if input_qdata.ndim != 3 or input_qdata.dtype is not torch.int8:
        raise ValueError(f"{name} projection input must be [batch,sequence,features] INT8")
    batch, sequence_length, input_features = input_qdata.shape
    if input_scale.shape != (batch, sequence_length) or input_scale.dtype is not torch.float32:
        raise ValueError(f"{name} projection input scale must be a batch/sequence FP32 matrix")
    if weight_qdata.ndim != 2 or weight_qdata.dtype is not torch.int8:
        raise ValueError(f"{name} projection weight must be a two-dimensional INT8 tensor")
    if weight_qdata.shape[1] != input_features or weight_qdata.shape[0] % head_dim:
        raise ValueError(f"{name} projection weight must map the input to complete D64/D128 heads")
    if weight_scale.shape != (weight_qdata.shape[0], 1) or weight_scale.dtype is not torch.float32:
        raise ValueError(
            f"{name} projection weight scale must be one FP32 value per output feature"
        )
    heads = weight_qdata.shape[0] // head_dim
    if norm_weight.dtype not in _SUPPORTED_NORM_DTYPES:
        raise ValueError(
            f"{name} projection RMSNorm weight must be an FP16/BF16/FP32 D64/D128 vector"
        )
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != sequence_length:
        raise ValueError(f"{name} projection RoPE cos/sin must match the sequence")
    rotary_dim = cos.shape[1]
    if rotary_dim < 2 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"{name} projection rotary dimension must be even and fit D64/D128")
    if cos.dtype is not torch.float32 or sin.dtype is not cos.dtype:
        raise ValueError(f"{name} projection RoPE cos/sin must use FP32")
    operands = input_qdata, input_scale, weight_qdata, weight_scale, norm_weight, cos, sin
    if any(operand.device != input_qdata.device for operand in operands):
        raise ValueError(f"{name} projection operands must share a device")
    if any(
        operand.layout is not torch.strided or not operand.is_contiguous() for operand in operands
    ):
        raise ValueError(f"{name} projection operands must be contiguous strided tensors")
    if sequence_length < 1:
        raise ValueError(f"{name} projection sequence must contain at least one row")
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError(f"{name} projection RMSNorm epsilon must be finite and positive")
    return batch, sequence_length, heads
