"""Functional ConvRot INT8 linear execution."""

from collections.abc import Callable
from typing import Any

import torch

from piper_kernels._input_activations import InputActivation
from piper_kernels.linear._dispatch import apply_linear_autocast, bind_linear_arguments
from piper_kernels.weights._views import require_untransposed
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

from . import dispatch


def convrot_int8_linear(
    input: torch.Tensor,  # noqa: A002
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation_fn: InputActivation | None = None,
) -> torch.Tensor:
    """Apply an optional input activation followed by a ConvRot INT8 linear."""
    weight._require_matrix("linear")
    require_untransposed(weight, "linear")
    converted_input, converted_weight, bias = apply_linear_autocast(input, weight, bias)
    assert isinstance(converted_weight, ConvRotInt8Tensor)
    return dispatch.linear(
        converted_input,
        converted_weight.qdata,
        converted_weight.scale,
        converted_weight.dtype,
        converted_weight.group_size,
        bias,
        activation_fn=activation_fn,
        input_scale=converted_weight.act_per_tensor_scale,
    )


def linear_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    linear_input, weight, bias = bind_linear_arguments(args, kwargs)
    if not isinstance(linear_input, torch.Tensor) or not isinstance(weight, ConvRotInt8Tensor):
        raise TypeError(
            "ConvRot linear dispatch requires a tensor input and ConvRotInt8Tensor weight"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(f"ConvRot linear bias must be a tensor or None, got {type(bias).__name__}")
    return convrot_int8_linear(linear_input, weight, bias)
