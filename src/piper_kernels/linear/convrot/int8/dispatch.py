"""Validated storage-level entrypoints for ConvRot INT8."""

import torch

from piper_kernels.linear import _bias
from piper_kernels.linear._input_activations import input_activation_width

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
    reference.validate_storage(qdata, scale, group_size, dtype)
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


def _run_linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None,
    activation_fn: str | None,
) -> torch.Tensor:
    """Run a validated ConvRot linear through the selected backend."""
    if _backend.select_linear_backend(input) is not None:
        return _ops.linear(
            input,
            qdata,
            scale,
            bias,
            group_size,
            activation_fn,
        )
    return reference.linear(
        input,
        qdata,
        scale,
        group_size,
        bias,
        activation_fn=activation_fn,
    )


def linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None = None,
    *,
    activation_fn: str | None = None,
) -> torch.Tensor:
    """Apply raw ConvRot INT8 storage to a floating-point activation.

    ``qdata`` stores the two-dimensional INT8 weight in its rotated basis,
    and ``scale`` contains one float32 value per output channel. This is the
    internal storage-level ABI. Consumers should call
    :func:`torch.nn.functional.linear` with a ``ConvRotInt8Tensor`` weight.
    """
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
    return _run_linear(input, qdata, scale, group_size, bias, activation_fn)
