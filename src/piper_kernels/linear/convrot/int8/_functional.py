"""Functional and storage-level ConvRot INT8 linear entrypoints."""

from collections.abc import Callable
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor

from piper_kernels._input_activations import InputActivation, input_activation_width
from piper_kernels.linear import _bias
from piper_kernels.linear._dispatch import apply_linear_autocast, bind_linear_arguments
from piper_kernels.weights._views import require_untransposed
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.int8._quantization import (
    validate_activation_scale,
    validate_storage,
)

from . import _backend, _ops, reference


def _validate_linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None,
    expected_features: int,
) -> torch.Tensor | None:
    """Validate one linear and canonicalize its optional bias."""
    validate_storage(qdata, scale, group_size, dtype)
    if input.ndim == 0 or input.shape[-1] != expected_features:
        actual = 0 if input.ndim == 0 else input.shape[-1]
        raise ValueError(
            f"ConvRot linear input has {actual} features, expected {expected_features}"
        )
    if input.device != qdata.device:
        raise ValueError(
            f"ConvRot input and weight must share a device, got {input.device}/{qdata.device}"
        )
    if input.dtype is not dtype:
        raise ValueError(
            f"ConvRot input must match the weight's logical dtype, got {input.dtype}/{dtype}"
        )
    if input.layout is not torch.strided:
        raise ValueError("ConvRot input must use strided layout")

    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError(
                f"ConvRot linear bias must be a tensor or None, got {type(bias).__name__}"
            )
        expected_bias_shape = (qdata.shape[0],)
        if tuple(bias.shape) != expected_bias_shape:
            raise ValueError(
                f"ConvRot linear bias must have shape {expected_bias_shape}, "
                f"got {tuple(bias.shape)}"
            )
        if bias.device != input.device:
            raise ValueError(
                "ConvRot input, weight, and bias must share a device, "
                f"got {input.device}/{qdata.device}/{bias.device}"
            )
        _bias.validate_dtype(bias, "ConvRot linear")
        if bias.layout is not torch.strided:
            raise ValueError("ConvRot linear bias must use strided layout")

    if torch.is_grad_enabled() and (
        input.requires_grad or scale.requires_grad or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError(
            "ConvRot INT8 linear does not support autograd; detach its inputs or use no_grad"
        )
    return None if bias is None else bias.contiguous()


def linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None = None,
    *,
    activation_fn: str | None = None,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply raw ConvRot INT8 storage to a floating-point activation.

    ``qdata`` stores the two-dimensional INT8 weight in its rotated basis,
    and ``scale`` contains one float32 value per output channel. This is the
    internal storage-level ABI. Consumers should call
    :func:`torch.nn.functional.linear` with a ``ConvRotInt8Tensor`` weight.
    """
    if qdata.ndim != 2:
        raise ValueError("ConvRot INT8 linear requires 2-D qdata")
    validate_activation_scale(input_scale, qdata.device)
    in_features = qdata.shape[1]
    bias = _validate_linear(
        input,
        qdata,
        scale,
        dtype,
        group_size,
        bias,
        in_features * input_activation_width(activation_fn),
    )
    if input.device.type == "meta":
        return input.new_empty((*input.shape[:-1], qdata.shape[0]))
    backend = _backend.select_linear_backend(input)
    if backend is None:
        return reference.linear(
            input,
            qdata,
            scale,
            group_size,
            bias,
            activation_fn=activation_fn,
            input_scale=input_scale,
        )
    if torch.compiler.is_compiling() or isinstance(input, FakeTensor):
        return _ops.linear(
            input,
            qdata,
            scale,
            bias,
            group_size,
            activation_fn,
            input_scale,
        )
    return backend.linear(
        input,
        qdata,
        scale,
        bias,
        group_size,
        activation_fn,
        input_scale,
    )


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
    return linear(
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
