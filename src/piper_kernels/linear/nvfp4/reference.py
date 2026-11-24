"""Portable PyTorch NVFP4 activation preparation and projection."""

from __future__ import annotations

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels._input_activations import apply_input_activation
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.nvfp4 import _layout
from piper_kernels.weights.nvfp4._quantization import quantize_nvfp4


def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
    high_first: bool = False,
    *,
    group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare activations in PyTorch; static None means unit global scale."""
    activated = apply_input_activation(input, activation_fn)
    if group_size:
        activated = rotate_groups(activated.float(), group_size)
    flattened = activated.reshape(-1, activated.shape[-1]).contiguous()
    if flattened.shape[0] == 0:
        if dynamic_activation_scale:
            global_scale = input.new_zeros((), dtype=torch.float32)
        elif activation_per_tensor_scale is not None:
            global_scale = activation_per_tensor_scale.clone()
        else:
            global_scale = input.new_ones((), dtype=torch.float32)
        return (
            input.new_empty(_layout.qdata_shape(0, flattened.shape[1]), dtype=torch.uint8),
            input.new_empty(_layout.scale_shape(0, flattened.shape[1]), dtype=torch.float8_e4m3fn),
            global_scale,
        )
    global_scale = (
        per_tensor_amax_to_scale(flattened.abs().amax())
        if dynamic_activation_scale
        else activation_per_tensor_scale
    )
    if global_scale is None:
        global_scale = input.new_ones((), dtype=torch.float32)
    qdata, scale = quantize_nvfp4(flattened, global_scale, swizzled=True)
    if high_first:
        qdata = _layout.swap_packed_pairs(qdata)
    return qdata, scale, global_scale


def linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize block scales, accumulate and apply affine terms in FP32, then cast."""
    if input_qdata.shape[0] == 0:
        return input_qdata.new_empty((0, weight_qdata.shape[0]), dtype=logical_dtype)

    def dequantize(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return TorchAONVFP4Tensor(
            qdata,
            scale,
            _layout.BLOCK_SIZE,
            torch.float32,
            is_swizzled_scales=True,
        ).dequantize(torch.float32)

    # Both operands have the same nibble ordering, so their dot product also
    # agrees when each adjacent pair is swapped in both dequantized tensors.
    result = dequantize(input_qdata, input_scale) @ dequantize(weight_qdata, weight_scale).T
    global_scale = input_per_tensor_scale
    if weight_per_tensor_scale is not None:
        global_scale = global_scale * weight_per_tensor_scale
    result.mul_(global_scale)
    if bias is not None:
        result.add_(bias.float())
    return result.to(logical_dtype)


def linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_activation_scale: bool,
    high_first: bool = False,
    *,
    group_size: int = 0,
) -> torch.Tensor:
    """Run an independent PyTorch W4A4 projection, optionally in the ConvRot basis."""
    prepared = prepare_input(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
        high_first=high_first,
        group_size=group_size,
    )
    return linear_prepared(
        *prepared,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        input.dtype,
    ).reshape(*input.shape[:-1], weight_qdata.shape[0])


def linear_w4a16(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    high_first: bool = False,
    *,
    group_size: int = 0,
) -> torch.Tensor:
    """Dequantize swizzled NVFP4 weights and apply a floating activation, optionally rotated."""
    if input.numel() == 0 or weight_qdata.shape[0] == 0:
        return input.new_empty((*input.shape[:-1], weight_qdata.shape[0]))
    active = rotate_groups(input.float(), group_size) if group_size else input.float()
    flattened = active.reshape(-1, active.shape[-1])
    packed_weight = _layout.swap_packed_pairs(weight_qdata) if high_first else weight_qdata
    weight = TorchAONVFP4Tensor(
        packed_weight,
        weight_scale,
        _layout.BLOCK_SIZE,
        torch.float32,
        is_swizzled_scales=True,
    ).dequantize(torch.float32)
    if weight_per_tensor_scale is not None:
        gs = weight_per_tensor_scale.to(device=weight.device, dtype=torch.float32)
        apply_early = gs.abs() < 1.0
        pre_scale = torch.where(apply_early, gs, torch.ones_like(gs))
        post_scale = torch.where(apply_early, torch.ones_like(gs), gs)
        result = ((flattened * pre_scale) @ weight.T) * post_scale
    else:
        result = flattened @ weight.T
    if bias is not None:
        result.add_(bias.float())
    return result.to(input.dtype).reshape(*input.shape[:-1], weight_qdata.shape[0])


__all__ = ["linear", "linear_prepared", "linear_w4a16", "prepare_input"]
