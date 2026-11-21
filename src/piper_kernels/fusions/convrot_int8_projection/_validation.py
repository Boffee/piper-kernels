"""Metadata checks for prepared ConvRot INT8 projections into attention heads."""

import torch

from piper_kernels._validation import validate_positive_finite_float
from piper_kernels.fusions.attention._validation import resolve_head_dim
from piper_kernels.linear import _bias

_SUPPORTED_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_projection_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    head_dim: int,
    name: str,
) -> tuple[int, int, int]:
    """Check common storage; callers own head support, row bounds, and transforms."""
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
    if bias is not None:
        _bias.validate_dtype(bias, f"{name} projection")
        if bias.shape != (weight_qdata.shape[0],):
            raise ValueError(f"{name} projection bias must have one value per output feature")
    for operand in (input_qdata, input_scale, weight_qdata, weight_scale, bias):
        if operand is None:
            continue
        if operand.device != input_qdata.device:
            raise ValueError(f"{name} projection operands must share a device")
        if operand.layout is not torch.strided or not operand.is_contiguous():
            raise ValueError(f"{name} projection operands must be contiguous strided tensors")
    return batch, sequence_length, weight_qdata.shape[0] // head_dim


def validate_qk_transform_inputs(
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequence_length: int,
    head_dim: int | None = None,
    device: torch.device,
    norm_epsilon: float,
    name: str,
) -> int:
    """Validate RMSNorm and RoPE transform operands and return ``head_dim``."""
    resolved_head_dim = resolve_head_dim(norm_weight, head_dim)
    if norm_weight is not None and norm_weight.dtype not in _SUPPORTED_NORM_DTYPES:
        raise ValueError(
            f"{name} projection RMSNorm weight must be an FP16/BF16/FP32 D64/D128 vector"
        )
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != sequence_length:
        raise ValueError(f"{name} projection RoPE cos/sin must match the sequence")
    rotary_dim = cos.shape[1]
    if rotary_dim < 2 or rotary_dim > resolved_head_dim or rotary_dim % 2:
        raise ValueError(f"{name} projection rotary dimension must be even and fit D64/D128")
    if cos.dtype is not torch.float32 or sin.dtype is not cos.dtype:
        raise ValueError(f"{name} projection RoPE cos/sin must use FP32")
    operands = tuple(operand for operand in (norm_weight, cos, sin) if operand is not None)
    if any(operand.device != device for operand in operands):
        raise ValueError(f"{name} projection operands must share a device")
    if any(
        operand.layout is not torch.strided or not operand.is_contiguous() for operand in operands
    ):
        raise ValueError(f"{name} projection operands must be contiguous strided tensors")
    if sequence_length < 1:
        raise ValueError(f"{name} projection sequence must contain at least one row")
    validate_positive_finite_float(
        norm_epsilon,
        f"{name} projection RMSNorm epsilon must be finite and positive",
    )
    return resolved_head_dim


def validate_qk_projection_inputs(  # noqa: PLR0913
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
    name: str,
    head_dim: int | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    """Validate inputs to a fused ConvRot INT8 Q/K projection kernel."""
    head_dim = resolve_head_dim(norm_weight, head_dim)
    batch, sequence_length, heads = validate_projection_inputs(
        input_qdata, input_scale, weight_qdata, weight_scale, bias, head_dim=head_dim, name=name
    )
    validate_qk_transform_inputs(
        norm_weight,
        cos,
        sin,
        sequence_length=sequence_length,
        head_dim=head_dim,
        device=input_qdata.device,
        norm_epsilon=norm_epsilon,
        name=name,
    )
    return batch, sequence_length, heads
