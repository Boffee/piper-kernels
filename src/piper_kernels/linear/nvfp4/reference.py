"""Portable PyTorch NVFP4 activation preparation and projection."""

from __future__ import annotations

from typing import cast

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels._input_activations import apply_input_activation
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.nvfp4 import _layout
from piper_kernels.weights.nvfp4._typing import NVFP4Storage


def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
    high_first: bool = False,
    *,
    group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare activations using only PyTorch rotation and quantization operations."""
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
            raise ValueError("static NVFP4 preparation requires a per-tensor scale")
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
        raise ValueError("static NVFP4 preparation requires a per-tensor scale")
    encoding_scale = torch.where(global_scale == 0, torch.ones_like(global_scale), global_scale)
    encoded = cast(
        NVFP4Storage,
        TorchAONVFP4Tensor.to_nvfp4(
            flattened.float(),
            block_size=_layout.BLOCK_SIZE,
            per_tensor_scale=encoding_scale,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        ),
    )
    qdata = _layout.swap_packed_pairs(encoded.qdata) if high_first else encoded.qdata
    return qdata, encoded.scale, global_scale


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


__all__ = ["linear", "linear_prepared", "prepare_input"]
