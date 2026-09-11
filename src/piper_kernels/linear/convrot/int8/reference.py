"""Portable INT8 ConvRot reference implementation."""

import math

import torch

from piper_kernels._input_activations import apply_input_activation
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8._quantization import dynamic_quantize_rows


def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and dynamically quantize a linear input for one or more weights."""
    input_2d = input.reshape(math.prod(input.shape[:-1]), input.shape[-1])
    input_qdata, input_scale = dynamic_quantize_rows(rotate_groups(input_2d.float(), group_size))
    return (
        input_qdata.reshape(input.shape),
        input_scale.reshape(input.shape[:-1]),
    )


def linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a ConvRot weight to an already rotated and quantized linear input."""
    leading_shape = input_qdata.shape[:-1]
    in_features = input_qdata.shape[-1]
    rows = math.prod(leading_shape)
    input_2d = input_qdata.reshape(rows, in_features)
    if input_qdata.device.type == "cpu":
        accumulated = input_2d.to(torch.int32) @ weight_qdata.T.to(torch.int32)
    else:
        # Float32 represents each INT8 product exactly. Only very long reductions
        # can round the integer sum, which is preferable to rejecting the shape.
        accumulated = input_2d.float() @ weight_qdata.T.float()

    # Reuse the FP32 accumulator for the epilogue. Out-of-place broadcasts retain
    # multiple full [rows, out_features] temporaries, which is prohibitive for
    # large reference workloads even though the final BF16 output itself fits.
    result = accumulated.to(torch.float32)
    result.mul_(input_scale.reshape(-1, 1).to(torch.float32))
    result.mul_(weight_scale.reshape(1, -1).to(torch.float32))
    if bias is not None:
        result.add_(bias.to(torch.float32))
    return result.to(logical_dtype).reshape(*leading_shape, weight_qdata.shape[0])


def linear(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
    *,
    activation_fn: str | None = None,
) -> torch.Tensor:
    """Run the portable PyTorch ConvRot W8A8 linear implementation."""
    prepared_input = apply_input_activation(input, activation_fn)
    input_qdata, input_scale = prepare_input(
        prepared_input,
        group_size,
    )
    return linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        input.dtype,
        bias,
    )
