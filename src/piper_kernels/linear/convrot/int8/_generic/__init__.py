"""Generic preparation and weight updates; tuned linear backends are optional."""

import torch

from piper_kernels.linear._input_activations import apply_input_activation
from piper_kernels.linear.convrot._rotation import validate_group_size

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
        and _triton_backend.supports_device(value.device)
    )


def prepare_input(
    input: torch.Tensor,  # noqa: A002
    group_size: int,
    activation_fn: str | None = None,
    *,
    out: PreparedInput | None = None,
) -> PreparedInput:
    """Rotate and quantize without requiring a tuned INT8 matrix backend."""
    validate_group_size(group_size)
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
        out[1].fill_(1e-30)
    elif _use_triton(value):
        assert _triton_backend is not None
        _triton_backend.prepare_input(value, group_size, out=out)
    else:
        prepared = reference.prepare_input(value, group_size)
        for output, prepared_value in zip(out, prepared, strict=True):
            output.copy_(prepared_value)
    return out


def add_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    update: torch.Tensor,
    group_size: int,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Update via shared Triton primitives where available, otherwise PyTorch."""
    if alpha == 0 or qdata.numel() == 0:
        return
    if _use_triton(qdata):
        assert _triton_backend is not None
        _triton_backend.add_(qdata, scale, update, group_size, alpha, rounding_seed)
    else:
        reference.add_(qdata, scale, update, group_size, alpha, rounding_seed)


def addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Share update orchestration independently of INT8 GEMM target support."""
    if (beta == 1 and alpha == 0) or qdata.numel() == 0:
        return
    if _use_triton(qdata):
        assert _triton_backend is not None
        _triton_backend.addmm_(qdata, scale, mat1, mat2, group_size, beta, alpha, rounding_seed)
    else:
        reference.addmm_(qdata, scale, mat1, mat2, group_size, beta, alpha, rounding_seed)


__all__ = ["add_", "addmm_", "prepare_input"]
