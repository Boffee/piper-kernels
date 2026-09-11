"""Functional linear execution using ConvRot NVFP4 weights."""

from collections.abc import Callable
from typing import Any

import torch

from piper_kernels.linear._dispatch import apply_linear_autocast, bind_linear_arguments
from piper_kernels.linear.nvfp4._functional import supports_semantic_linear
from piper_kernels.weights._views import require_untransposed
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor


def _supports_convrot_linear(input: object, weight: ConvRotNVFP4Tensor) -> bool:  # noqa: A002
    return (
        supports_semantic_linear(input, weight)
        and isinstance(input, torch.Tensor)
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and input.ndim > 0
        and input.shape[-1] % weight.group_size == 0
    )


def convrot_nvfp4_linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight: ConvRotNVFP4Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a canonical NVFP4 weight and activation in the same ConvRot basis."""
    if not isinstance(input, torch.Tensor) or not isinstance(weight, ConvRotNVFP4Tensor):
        raise TypeError(
            "ConvRot NVFP4 linear requires a tensor input and ConvRotNVFP4Tensor weight"
        )
    require_untransposed(weight, "linear")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"ConvRot NVFP4 linear bias must be a tensor or None, got {type(bias).__name__}"
        )
    converted_input, converted_weight, bias = apply_linear_autocast(input, weight, bias)
    assert isinstance(converted_weight, ConvRotNVFP4Tensor)
    weight = converted_weight
    if not _supports_convrot_linear(converted_input, weight):
        raise ValueError("ConvRot NVFP4 linear requires canonical SM120 NVFP4 operands")
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
        weight.group_size,
        weight.high_first,
    )


def linear_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    input, weight, bias = bind_linear_arguments(args, kwargs)  # noqa: A001
    if not isinstance(input, torch.Tensor) or not isinstance(weight, ConvRotNVFP4Tensor):
        raise TypeError(
            "ConvRot NVFP4 linear dispatch requires a tensor input and ConvRotNVFP4Tensor weight"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"ConvRot NVFP4 linear bias must be a tensor or None, got {type(bias).__name__}"
        )
    return convrot_nvfp4_linear(input, weight, bias)
