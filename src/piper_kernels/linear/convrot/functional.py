"""Format-neutral functional ConvRot operators."""

from typing import Literal

import torch

from .int8.tensor import ConvRotInt8Tensor, _convrot_int8_input_activation_linear

__all__ = ["convrot_linear"]


def convrot_linear(
    input: torch.Tensor,  # noqa: A002 - match torch.nn.functional.linear terminology
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_activation: Literal["swiglu"],
) -> torch.Tensor:
    """Apply an explicit input activation followed by a ConvRot linear.

    ``input`` has shape ``[..., 2 * in_features]`` for ``"swiglu"`` and
    uses the raw ``[up | gate]`` layout. ``weight`` has logical shape
    ``[out_features, in_features]`` and ``bias``, when present, has shape
    ``[out_features]``. Input, logical weight, and bias dtypes and devices
    must match. The operation computes ``up * silu(gate)`` before the linear,
    and the result has shape ``[..., out_features]``.

    ``input_activation`` is required and keyword-only so activation fusion is
    always explicit. Unsupported devices and shapes materialize the activation
    and then use the ordinary ConvRot linear path. This is an inference-only
    operation and does not support autograd.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"ConvRot input must be a tensor, got {type(input).__name__}")
    if not isinstance(weight, ConvRotInt8Tensor):
        raise TypeError(f"ConvRot weight must be ConvRotInt8Tensor, got {type(weight).__name__}")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(f"ConvRot linear bias must be a tensor or None, got {type(bias).__name__}")
    return _convrot_int8_input_activation_linear(input, weight, bias, input_activation)
