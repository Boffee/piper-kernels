"""Portable updates for ConvRot INT8 weight storage."""

import torch

from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8._quantization import dynamic_quantize_rows


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
    """Add a matrix product to a logical ConvRot weight and requantize it in place."""
    if beta == 0:
        rotated_weight = torch.zeros(qdata.shape, device=qdata.device, dtype=torch.float32)
    else:
        rotated_weight = qdata.float() * scale
    rotated_mat2 = rotate_groups(mat2.float(), group_size)
    merged = torch.addmm(rotated_weight, mat1.float(), rotated_mat2, beta=beta, alpha=alpha)
    _requantize_(qdata, scale, merged, rounding_seed)


def add_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    update: torch.Tensor,
    group_size: int,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Add a dense logical update to a ConvRot weight and requantize it in place."""
    rotated_weight = qdata.float() * scale
    rotated_update = rotate_groups(update.float(), group_size)
    merged = torch.add(rotated_weight, rotated_update, alpha=alpha)
    _requantize_(qdata, scale, merged, rounding_seed)


def _requantize_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    merged: torch.Tensor,
    rounding_seed: int | None,
) -> None:
    """Refill existing rowwise INT8 storage from one merged rotated weight."""
    merged_qdata, merged_scale = dynamic_quantize_rows(
        merged,
        rounding_seed=rounding_seed,
    )
    qdata.copy_(merged_qdata)
    scale.copy_(merged_scale)
