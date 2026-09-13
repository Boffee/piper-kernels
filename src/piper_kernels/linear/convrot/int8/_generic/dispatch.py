"""Dispatch generic activation preparation between Triton and PyTorch."""

import torch

from piper_kernels._input_activations import apply_input_activation
from piper_kernels._triton import runtime
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.convrot.int8._quantization import validate_activation_scale

from .. import reference
from .._interfaces import PreparedInput

try:
    from . import triton as _triton_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _triton_backend = None


def _use_triton(value: torch.Tensor) -> bool:
    # Bound live row storage; wider rows still work through ordinary PyTorch.
    return (
        _triton_backend is not None
        and value.numel() != 0
        and value.shape[-1] <= 16384
        and runtime.supports_device(value.device)
    )


def prepare_input(
    input: torch.Tensor,  # noqa: A002
    group_size: int,
    activation_fn: str | None = None,
    input_scale: torch.Tensor | None = None,
    *,
    out: PreparedInput | None = None,
) -> PreparedInput:
    """Rotate and quantize without requiring a tuned INT8 matrix backend."""
    validate_group_size(group_size)
    validate_activation_scale(input_scale, input.device)
    if input.ndim == 0 or input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("ConvRot preparation requires floating-point input with a feature axis")
    value = apply_input_activation(input, activation_fn).contiguous()
    if value.shape[-1] % group_size:
        raise ValueError("ConvRot preparation width must be divisible by group size")
    shapes = (value.shape, value.shape[:-1])
    dtypes = (torch.int8, torch.float32)
    if out is None:
        out = (
            torch.empty(shapes[0], dtype=dtypes[0], device=value.device),
            torch.empty(shapes[1], dtype=dtypes[1], device=value.device),
        )
    elif any(
        output.shape != shape
        or output.dtype != dtype
        or output.device != value.device
        or not output.is_contiguous()
        for output, shape, dtype in zip(out, shapes, dtypes, strict=True)
    ):
        raise ValueError("ConvRot preparation output storage is incompatible")
    if value.numel() == 0:
        if input_scale is None:
            out[1].fill_(1e-30)
        else:
            out[1].copy_(input_scale)
    elif _use_triton(value):
        assert _triton_backend is not None
        _triton_backend.prepare_input(value, group_size, input_scale, out=out)
    else:
        prepared = reference.prepare_input(value, group_size, input_scale)
        for output, prepared_value in zip(out, prepared, strict=True):
            output.copy_(prepared_value)
    return out
