"""Semantic ConvRot NVFP4 W4A4 operators."""

from __future__ import annotations

import torch

from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation
from piper_kernels.weights.convrot._rotation import validate_group_size


@torch.library.custom_op("piper_kernels::convrot_nvfp4_linear", mutates_args=())
def linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_activation_scale: bool,
    group_size: int,
    high_first: bool = False,
) -> torch.Tensor:
    """Apply a standard NVFP4 weight and activation in the same ConvRot basis."""
    validate_group_size(group_size)
    shape = nvfp4_validation.validate_semantic_linear(
        input,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        dynamic_activation_scale,
        "ConvRot NVFP4 linear",
        allow_empty=True,
    )
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("ConvRot NVFP4 linear input must be FP16, BF16, or FP32")
    if isinstance(shape.input_features, int) and shape.input_features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 input features {shape.input_features} must be divisible "
            f"by group size {group_size}"
        )

    if shape.rows == 0:
        return input.new_empty((*input.shape[:-1], shape.output_features))

    input_qdata, input_scale, input_per_tensor_scale = nvfp4_ops._prepare_compiled(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
        high_first=high_first,
        group_size=group_size,
    )
    result = nvfp4_ops._execute_prepared(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        input.dtype,
    )
    return result.reshape(*input.shape[:-1], shape.output_features)


@linear.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _linear_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _activation_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _dynamic_activation_scale: bool,
    group_size: int,
    _high_first: bool = False,
) -> torch.Tensor:
    validate_group_size(group_size)
    input_features = input.shape[-1]
    if isinstance(input_features, int) and input_features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 input features {input_features} must be divisible "
            f"by group size {group_size}"
        )
    return input.new_empty((*input.shape[:-1], weight_qdata.shape[0]))


__all__ = ["linear"]
