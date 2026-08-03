"""Backend-independent functional API for ConvRot operators."""

import torch

from ._reference import reference_linear, validate_storage

try:
    from ._triton import triton_int8_convrot_linear as _triton_linear
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_linear = None


def _can_use_triton(activation: torch.Tensor, qdata: torch.Tensor) -> bool:
    return (
        _triton_linear is not None
        and activation.device.type == "cuda"
        and activation.device == qdata.device
        and activation.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and torch.cuda.get_device_capability(activation.device) >= (7, 5)
    )


def int8_convrot_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a ConvRot INT8 weight to a floating-point activation.

    ``qdata`` stores the two-dimensional INT8 weight in its rotated basis.
    ``scale`` contains one float32 value per output channel. The function
    automatically selects the Triton backend when supported and otherwise
    executes the portable PyTorch reference implementation.
    """
    validate_storage(qdata, scale, group_size, activation.dtype)
    if activation.ndim == 0 or activation.shape[-1] != qdata.shape[1]:
        actual = 0 if activation.ndim == 0 else activation.shape[-1]
        raise ValueError(
            f"ConvRot linear input has {actual} features, expected {qdata.shape[1]}"
        )
    if activation.device != qdata.device:
        raise ValueError(
            "ConvRot activation and weight must share a device, "
            f"got {activation.device}/{qdata.device}"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(f"ConvRot linear bias must be a tensor or None, got {type(bias).__name__}")
    if _can_use_triton(activation, qdata):
        assert _triton_linear is not None
        return _triton_linear(activation, qdata, scale, bias, group_size)
    return reference_linear(activation, qdata, scale, group_size, bias)
