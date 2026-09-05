"""Semantic ConvRot NVFP4 W4A4 operators."""

from __future__ import annotations

import torch

from piper_kernels.linear._input_activations import input_activation_width
from piper_kernels.linear.convrot._rotation import validate_group_size
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation
from piper_kernels.linear.nvfp4 import reference as nvfp4_reference

from . import triton as convrot_nvfp4


def _prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    group_size: int,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input.numel() == 0:
        return nvfp4_reference.prepare_input(
            input,
            activation_per_tensor_scale,
            dynamic_activation_scale,
            activation_fn,
            high_first,
            group_size=group_size,
        )
    if dynamic_activation_scale:
        return convrot_nvfp4.prepare_dynamic(
            input,
            group_size,
            activation_fn,
            high_first=high_first,
        )
    if activation_per_tensor_scale is None:
        raise ValueError("static ConvRot NVFP4 preparation requires a per-tensor scale")
    return convrot_nvfp4.prepare_static(
        input,
        activation_per_tensor_scale,
        group_size,
        activation_fn,
        high_first=high_first,
    )


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
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("ConvRot NVFP4 linear input must be FP16 or BF16")
    if isinstance(shape.input_features, int) and shape.input_features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 input features {shape.input_features} must be divisible "
            f"by group size {group_size}"
        )

    if shape.rows == 0:
        return input.new_empty((*input.shape[:-1], shape.output_features))

    input_qdata, input_scale, input_per_tensor_scale = _prepare_input(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
        group_size,
        high_first=high_first,
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


@torch.library.custom_op("piper_kernels::convrot_nvfp4_prepare_input", mutates_args=())
def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    group_size: int,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional activation and prepare it in the ConvRot NVFP4 basis."""
    validate_group_size(group_size)
    return _prepare_input(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
        group_size,
        activation_fn,
        high_first,
    )


@prepare_input.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _prepare_input_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_per_tensor_scale: torch.Tensor | None,
    _dynamic_activation_scale: bool,
    group_size: int,
    activation_fn: str | None = None,
    _high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_group_size(group_size)
    if input.ndim == 0:
        raise ValueError("ConvRot NVFP4 input must be non-scalar")
    input_features = input.shape[-1] // input_activation_width(activation_fn)
    if isinstance(input_features, int) and input_features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 input features {input_features} must be divisible "
            f"by group size {group_size}"
        )
    rows = input.numel() // input.shape[-1]
    per_tensor_scale = (
        activation_per_tensor_scale.new_empty(())
        if activation_per_tensor_scale is not None
        else input.new_empty((), dtype=torch.float32)
    )
    return (
        input.new_empty(nvfp4_layout.qdata_shape(rows, input_features), dtype=torch.uint8),
        input.new_empty(
            nvfp4_layout.scale_shape(rows, input_features),
            dtype=torch.float8_e4m3fn,
        ),
        per_tensor_scale,
    )


__all__ = ["linear", "prepare_input"]
