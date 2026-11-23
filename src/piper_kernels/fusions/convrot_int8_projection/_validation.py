"""Metadata checks for prepared ConvRot INT8 projections into attention heads."""

import torch

from piper_kernels.fusions.attention._validation import (
    resolve_head_dim,
    validate_qk_transform_inputs,
)
from piper_kernels.linear import _bias


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
        name=f"{name} projection",
    )
    return batch, sequence_length, heads
