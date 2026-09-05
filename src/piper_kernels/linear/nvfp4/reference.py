"""Portable in-place updates shared by plain and ConvRot NVFP4 weights."""

from __future__ import annotations

import torch
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    pack_uint4,
    unpack_uint4,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels._stochastic_quantization import stochastic_codebook_indices
from piper_kernels.linear.convrot._rotation import rotate_groups

from . import _layout
from .tensor import _MIN_PER_TENSOR_SCALE, PiperNVFP4Tensor


def addmm_(
    weight: PiperNVFP4Tensor,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Merge a matrix product in the stored basis and refill NVFP4 storage."""
    if beta == 0:
        rotated_weight = torch.zeros(
            weight.shape,
            device=weight.device,
            dtype=weight.orig_dtype,
        )
    else:
        # Bypass ConvRot's logical dequantize override and remain in the
        # physical rotated basis used by the packed storage.
        rotated_weight = PiperNVFP4Tensor.dequantize(weight, weight.orig_dtype)
    rotated_update = rotate_groups(mat2, group_size) if group_size else mat2
    merged = torch.addmm(
        rotated_weight,
        mat1,
        rotated_update,
        beta=beta,
        alpha=alpha,
    )

    _refill_(weight, merged, rounding_seed)


def add_(
    weight: PiperNVFP4Tensor,
    group_size: int,
    update: torch.Tensor,
    *,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Merge a dense logical update in the stored basis and refill NVFP4 storage."""
    # Stay in the physical rotated basis used by the packed storage. Rotation
    # is linear, so this equals rotating the logical sum before requantization.
    rotated_weight = PiperNVFP4Tensor.dequantize(weight, weight.orig_dtype)
    rotated_update = rotate_groups(update, group_size) if group_size else update
    merged = torch.add(rotated_weight, rotated_update, alpha=alpha)
    _refill_(weight, merged, rounding_seed)


def _refill_(
    weight: PiperNVFP4Tensor,
    merged: torch.Tensor,
    rounding_seed: int | None,
) -> None:
    """Encode one merged rotated weight into the existing NVFP4 storage."""

    per_tensor_scale = None
    if weight.per_tensor_scale is not None:
        per_tensor_scale = per_tensor_amax_to_scale(merged.detach().abs().amax()).clamp_min(
            _MIN_PER_TENSOR_SCALE
        )
    encoded = PiperNVFP4Tensor.from_torchao(
        TorchAONVFP4Tensor.to_nvfp4(
            merged.float() if merged.dtype is torch.float16 else merged,
            block_size=weight.block_size,
            per_tensor_scale=per_tensor_scale,
            act_per_tensor_scale=weight.act_per_tensor_scale,
            is_swizzled_scales=weight.is_swizzled_scales,
            use_triton_kernel=False,
            act_quant_kwargs=weight.act_quant_kwargs,
        )
    )
    if rounding_seed is not None:
        _stochastic_recode_(
            encoded,
            merged,
            rounding_seed=rounding_seed,
        )

    encoded_qdata = encoded.qdata
    if weight.high_first:
        encoded_qdata = _layout.swap_packed_pairs(encoded_qdata)
    weight.qdata.copy_(encoded_qdata)
    weight.scale.copy_(encoded.scale)
    if weight.per_tensor_scale is not None:
        assert encoded.per_tensor_scale is not None
        weight.per_tensor_scale.copy_(encoded.per_tensor_scale)


def _stochastic_recode_(
    encoded: PiperNVFP4Tensor,
    source: torch.Tensor,
    *,
    rounding_seed: int,
) -> None:
    element_scale = encoded.get_hp_scales().repeat_interleave(
        encoded.block_size,
        dim=-1,
    )
    valid_scale = torch.isfinite(element_scale) & (element_scale > 0)
    normalized = torch.where(
        valid_scale,
        source.to(torch.float32) / element_scale.to(torch.float32),
        torch.zeros_like(source, dtype=torch.float32),
    )
    deterministic = unpack_uint4(encoded.qdata.contiguous().view(torch.uint8))
    codebook = f4_unpacked_to_f32(torch.arange(16, device=source.device, dtype=torch.uint8))
    codes = stochastic_codebook_indices(
        normalized,
        codebook,
        seed=rounding_seed,
        deterministic=deterministic,
    )
    codes = torch.where(valid_scale, codes, deterministic.to(torch.int64))
    encoded.qdata.view(torch.uint8).copy_(pack_uint4(codes.to(torch.uint8)))


__all__ = ["add_", "addmm_"]
