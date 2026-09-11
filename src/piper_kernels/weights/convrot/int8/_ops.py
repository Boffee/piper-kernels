"""Stable custom operations for in-place ConvRot INT8 weight updates."""

import torch

from . import _backend


@torch.library.custom_op(
    "piper_kernels::convrot_int8_addmm_",
    mutates_args=("qdata", "scale"),
)
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
    """Dispatch the stable addmm_ operation to a supported implementation."""
    execute = _backend.select_addmm(qdata)
    if execute is None:
        raise ValueError(f"ConvRot INT8 optimized addmm_ is unavailable on {qdata.device}")
    execute(qdata, scale, mat1, mat2, group_size, beta, alpha, rounding_seed)


@addmm_.register_fake
def _addmm_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _mat1: torch.Tensor,
    _mat2: torch.Tensor,
    _group_size: int,
    _beta: float,
    _alpha: float,
    _rounding_seed: int | None = None,
) -> None:
    return None


@torch.library.custom_op(
    "piper_kernels::convrot_int8_add_",
    mutates_args=("qdata", "scale"),
)
def add_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    update: torch.Tensor,
    group_size: int,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Dispatch the stable add_ operation to a supported implementation."""
    execute = _backend.select_add(qdata)
    if execute is None:
        raise ValueError(f"ConvRot INT8 optimized add_ is unavailable on {qdata.device}")
    execute(qdata, scale, update, group_size, alpha, rounding_seed)


@add_.register_fake
def _add_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _update: torch.Tensor,
    _group_size: int,
    _alpha: float,
    _rounding_seed: int | None = None,
) -> None:
    return None
