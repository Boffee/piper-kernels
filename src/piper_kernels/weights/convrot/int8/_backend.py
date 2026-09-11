"""Portable backend selection for ConvRot INT8 weight updates."""

from collections.abc import Callable

import torch

from piper_kernels._triton import runtime

from . import _update_reference as reference

type Add = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, float, int | None], None]
type Addmm = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, float, int | None], None
]
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


def select_add(input: torch.Tensor) -> Add | None:  # noqa: A002
    """Use shared accelerator updates; CPU keeps its directly traced reference."""
    return add_ if input.device.type not in ("cpu", "meta") else None


def select_addmm(input: torch.Tensor) -> Addmm | None:  # noqa: A002
    """Do not require an INT8 matrix policy for a floating-point update product."""
    return addmm_ if input.device.type not in ("cpu", "meta") else None
