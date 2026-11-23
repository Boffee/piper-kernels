"""Validation shared by projected Q/K operators."""

import torch

from piper_kernels._validation import validate_positive_finite_float

_SUPPORTED_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def resolve_head_dim(norm_weight: torch.Tensor | None, head_dim: int | None) -> int:
    """Infer the head width from an affine norm, or require it for a weightless norm."""
    if head_dim is None:
        if norm_weight is None:
            raise ValueError("Weightless RMSNorm requires an explicit head_dim")
        head_dim = norm_weight.shape[0] if norm_weight.ndim == 1 else 0
    if (
        isinstance(head_dim, bool)
        or not isinstance(head_dim, (int, torch.SymInt))
        or head_dim not in (64, 128)
    ):
        raise ValueError("Q/K projection requires head_dim=64 or 128")
    if norm_weight is not None and norm_weight.shape != (head_dim,):
        raise ValueError("Q/K projection RMSNorm weight must match head_dim")
    return head_dim


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
        raise ValueError(f"{name} RMSNorm weight must be an FP16/BF16/FP32 D64/D128 vector")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != sequence_length:
        raise ValueError(f"{name} split-half RoPE cos/sin must match the sequence")
    rotary_dim = cos.shape[1]
    if rotary_dim < 2 or rotary_dim > resolved_head_dim or rotary_dim % 2:
        raise ValueError(f"{name} split-half RoPE rotary dimension must be even and fit D64/D128")
    if cos.dtype is not torch.float32 or sin.dtype is not cos.dtype:
        raise ValueError(f"{name} split-half RoPE cos/sin must use FP32")
    operands = tuple(operand for operand in (norm_weight, cos, sin) if operand is not None)
    if any(operand.device != device for operand in operands):
        raise ValueError(f"{name} operands must share a device")
    if any(
        operand.layout is not torch.strided or not operand.is_contiguous() for operand in operands
    ):
        raise ValueError(f"{name} operands must be contiguous strided tensors")
    if sequence_length < 1:
        raise ValueError(f"{name} sequence must contain at least one row")
    validate_positive_finite_float(
        norm_epsilon,
        f"{name} RMSNorm epsilon must be finite and positive",
    )
    return resolved_head_dim
