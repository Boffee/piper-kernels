"""Functional execution and ``F.linear`` dispatch for ConvRot Trellis weights."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from piper_kernels.linear import _bias
from piper_kernels.linear._dispatch import apply_linear_autocast, bind_linear_arguments
from piper_kernels.weights._views import require_untransposed
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from piper_kernels.weights.convrot.trellis.tensor import validate_storage

from . import _ops


def _validate_linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    group_size: int,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    out_features, in_features, _ = validate_storage(
        bitstreams,
        block_scales,
        scale,
        block_size,
        group_size,
        dtype,
        row_widths=row_widths,
        require_contiguous=True,
    )
    if input.ndim == 0 or input.shape[-1] != in_features:
        actual = 0 if input.ndim == 0 else input.shape[-1]
        raise ValueError(
            f"ConvRot Trellis linear input has {actual} features, expected {in_features}"
        )
    if input.device != bitstreams.device:
        raise ValueError(
            "ConvRot Trellis input and weight must share a device, "
            f"got {input.device}/{bitstreams.device}"
        )
    if input.dtype is not dtype:
        raise ValueError(
            "ConvRot Trellis input must match the weight's logical dtype, "
            f"got {input.dtype}/{dtype}"
        )
    if input.layout is not torch.strided:
        raise ValueError("ConvRot Trellis input must use strided layout")

    if bias is not None:
        expected_bias_shape = (out_features,)
        if tuple(bias.shape) != expected_bias_shape:
            raise ValueError(
                f"ConvRot Trellis linear bias must have shape {expected_bias_shape}, "
                f"got {tuple(bias.shape)}"
            )
        if bias.device != input.device:
            raise ValueError(
                "ConvRot Trellis input, weight, and bias must share a device, "
                f"got {input.device}/{bitstreams.device}/{bias.device}"
            )
        _bias.validate_dtype(bias, "ConvRot Trellis linear")
        if bias.layout is not torch.strided:
            raise ValueError("ConvRot Trellis linear bias must use strided layout")

    if torch.is_grad_enabled() and (
        input.requires_grad
        or block_scales.requires_grad
        or scale.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError(
            "ConvRot Trellis linear does not support autograd; detach its inputs or use no_grad"
        )
    return None if bias is None else bias.contiguous()


def convrot_trellis_linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight: ConvRotTrellisTensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a canonical ConvRot Trellis weight to a floating-point activation."""
    if not isinstance(input, torch.Tensor) or not isinstance(weight, ConvRotTrellisTensor):
        raise TypeError(
            "ConvRot Trellis linear requires a tensor input and ConvRotTrellisTensor weight"
        )
    require_untransposed(weight, "linear")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"ConvRot Trellis linear bias must be a tensor or None, got {type(bias).__name__}"
        )
    converted_input, converted_weight, bias = apply_linear_autocast(input, weight, bias)
    assert isinstance(converted_weight, ConvRotTrellisTensor)
    canonical_bias = _validate_linear(
        converted_input,
        converted_weight.bitstreams,
        converted_weight.block_scales,
        converted_weight.scale,
        converted_weight.dtype,
        converted_weight.block_size,
        converted_weight.group_size,
        converted_weight.row_widths,
        bias,
    )
    return _ops.linear(
        converted_input,
        converted_weight.bitstreams,
        converted_weight.block_scales,
        converted_weight.scale,
        converted_weight.row_widths,
        canonical_bias,
        converted_weight.block_size,
        converted_weight.group_size,
    )


def linear_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    linear_input, weight, bias = bind_linear_arguments(args, kwargs)
    if not isinstance(linear_input, torch.Tensor) or not isinstance(weight, ConvRotTrellisTensor):
        raise TypeError(
            "ConvRot Trellis linear dispatch requires a tensor input and "
            "ConvRotTrellisTensor weight"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"ConvRot Trellis linear bias must be a tensor or None, got {type(bias).__name__}"
        )
    return convrot_trellis_linear(linear_input, weight, bias)


__all__ = ["convrot_trellis_linear", "linear_dispatch"]
