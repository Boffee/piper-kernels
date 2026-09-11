"""Functional NVFP4 linear dispatch and operand eligibility."""

from collections.abc import Callable
from typing import Any, cast

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor as TorchAONVFP4Tensor
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_linear as torchao_nvfp4_linear

from piper_kernels.linear._dispatch import apply_linear_autocast, bind_linear_arguments
from piper_kernels.weights._views import require_untransposed
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor, _layout


def supports_semantic_linear(input: object, weight: PiperNVFP4Tensor) -> bool:  # noqa: A002
    if not isinstance(input, torch.Tensor):
        return False
    tensor_input = cast(torch.Tensor, input)
    if isinstance(input, TorchAONVFP4Tensor):
        return False
    quantization = weight.act_quant_kwargs
    return (
        tensor_input.ndim > 0
        and tensor_input.dtype is weight.orig_dtype
        and tensor_input.device.type in ("cuda", "meta")
        and weight.block_size == _layout.BLOCK_SIZE
        and weight.is_swizzled_scales
        and not weight.use_triton_kernel
        and quantization is not None
        and quantization.block_size == _layout.BLOCK_SIZE
        and quantization.is_swizzled_scales
        and not quantization.use_triton_kernel
        and (quantization.use_dynamic_per_tensor_scale or weight.act_per_tensor_scale is not None)
        and (weight.per_tensor_scale is None or weight.per_tensor_scale.ndim == 0)
    )


def linear_dispatch(
    func: Callable[..., torch.Tensor],
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    input, weight, bias = bind_linear_arguments(args, kwargs)  # noqa: A001
    if not isinstance(input, torch.Tensor) or not isinstance(weight, PiperNVFP4Tensor):
        return torchao_nvfp4_linear(func, types, args, kwargs)
    require_untransposed(weight, "linear")
    if bias is not None and not isinstance(bias, torch.Tensor):
        return torchao_nvfp4_linear(func, types, args, kwargs)

    converted_input, converted_weight, bias = apply_linear_autocast(input, weight, bias)
    assert isinstance(converted_weight, PiperNVFP4Tensor)
    weight = converted_weight
    normalized_args = (converted_input, weight, bias)
    supported = supports_semantic_linear(converted_input, weight)
    if not supported:
        if weight.high_first:
            return torch.nn.functional.linear(converted_input, weight.dequantize(), bias)
        return torchao_nvfp4_linear(func, types, normalized_args, {})
    quantization = weight.act_quant_kwargs
    assert quantization is not None
    from . import _ops  # noqa: PLC0415

    return _ops.linear(
        converted_input,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
        bias,
        quantization.use_dynamic_per_tensor_scale,
        weight.high_first,
    )
