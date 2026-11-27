"""Portable INT8 ConvRot reference implementation."""

import math

import torch

from piper_kernels._input_activations import apply_input_activation
from piper_kernels.linear._dispatch import empty_linear_output
from piper_kernels.trellis._layout import TrellisLayout
from piper_kernels.trellis._reference import decode_reference, decode_variable_rows_reference
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8._quantization import dynamic_quantize_rows


def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and quantize a linear input using dynamic or static input scaling."""
    input_2d = input.reshape(math.prod(input.shape[:-1]), input.shape[-1])
    rotated = rotate_groups(input_2d.float(), group_size)
    if input_scale is None:
        input_qdata, row_scales = dynamic_quantize_rows(rotated)
    else:
        input_qdata = (rotated / input_scale).round().clamp(-128, 127).to(torch.int8)
        row_scales = input_scale.expand(input_2d.shape[0]).clone()
    return input_qdata.reshape(input.shape), row_scales.reshape(input.shape[:-1])


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
        # Keep operands in INT8 while accumulating in INT32; ordinary INT8 @ overflows.
        accumulated = torch._int_mm(input_2d, weight_qdata.T)
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
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the portable PyTorch ConvRot W8A8 linear implementation."""
    prepared_input = apply_input_activation(input, activation_fn)
    input_qdata, row_scales = prepare_input(prepared_input, group_size, input_scale)
    return linear_prepared(
        input_qdata,
        row_scales,
        weight_qdata,
        weight_scale,
        input.dtype,
        bias,
    )


def _decode_trellis_qdata(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor | None,
    block_size: int,
    *,
    round_int8: bool = True,
) -> torch.Tensor:
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    if row_widths is None:
        layout = TrellisLayout.from_storage(block_size, bitstreams.shape[2])
        return decode_reference(bitstreams, block_scales, layout, round_int8=round_int8).reshape(
            out_features, in_features
        )
    return decode_variable_rows_reference(
        bitstreams, block_scales, row_widths, block_size, round_int8=round_int8
    )


def linear_trellis_a16(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Portable FP32-rotated A16 reference for ConvRot Trellis linear."""
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    empty = empty_linear_output(input, out_features, in_features, bias)
    if empty is not None:
        return empty

    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)
    with torch.amp.autocast(input.device.type, enabled=False):
        qdata = _decode_trellis_qdata(
            bitstreams, block_scales, row_widths, block_size, round_int8=False
        )
        rotated = rotate_groups(input.reshape(rows, in_features).float(), group_size)
        accumulated = torch.matmul(rotated, qdata.float().T)
        accumulated.mul_(scale.reshape(1, out_features).to(torch.float32))
        if bias is not None:
            accumulated.add_(bias.to(torch.float32))
    return accumulated.to(input.dtype).reshape(*leading_shape, out_features)


def linear_trellis_w8a8(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Portable W8A8 reference matching decoded ``ConvRotInt8Tensor`` linear."""
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    empty = empty_linear_output(input, out_features, in_features, bias)
    if empty is not None:
        return empty
    with torch.amp.autocast(input.device.type, enabled=False):
        qdata = _decode_trellis_qdata(
            bitstreams, block_scales, row_widths, block_size, round_int8=True
        )
        return linear(input, qdata, scale, group_size, bias)
