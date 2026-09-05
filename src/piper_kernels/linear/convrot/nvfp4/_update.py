"""Rotation-aware in-place updates for ConvRot NVFP4 weights."""

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
from piper_kernels.linear.convrot._update import (
    validate_real_scalar,
    validate_rounding_seed,
    validate_update_operands,
)
from piper_kernels.linear.nvfp4 import _layout
from piper_kernels.linear.nvfp4.tensor import (
    _MIN_PER_TENSOR_SCALE,
    PiperNVFP4Tensor,
)


def _validate_storage(weight: PiperNVFP4Tensor, *, operation: str) -> None:
    if weight.device.type == "meta":
        raise ValueError(f"{operation} cannot update a meta tensor without values")
    rows, features = weight.shape
    expected_qdata_shape = _layout.qdata_shape(rows, features)
    if weight.qdata.dtype is not torch.uint8 or tuple(weight.qdata.shape) != expected_qdata_shape:
        raise ValueError(
            f"{operation} requires packed uint8 qdata with shape "
            f"{expected_qdata_shape}, got {weight.qdata.dtype} {tuple(weight.qdata.shape)}"
        )
    if weight.block_size != _layout.BLOCK_SIZE:
        raise ValueError(
            f"{operation} requires block size {_layout.BLOCK_SIZE}, got {weight.block_size}"
        )
    expected_scale_shape = (
        _layout.scale_shape(rows, features)
        if weight.is_swizzled_scales
        else (rows, features // weight.block_size)
    )
    if (
        weight.scale.dtype is not torch.float8_e4m3fn
        or tuple(weight.scale.shape) != expected_scale_shape
    ):
        raise ValueError(
            f"{operation} requires canonical block-16 FP8 scales with shape "
            f"{expected_scale_shape}, got {weight.scale.dtype} {tuple(weight.scale.shape)}"
        )
    if not weight.qdata.is_contiguous() or not weight.scale.is_contiguous():
        raise ValueError(
            f"{operation} requires contiguous packed storage; "
            "a transposed weight cannot be updated in place"
        )
    if weight.per_tensor_scale is not None and weight.per_tensor_scale.numel() != 1:
        raise ValueError(f"{operation} requires a scalar per-tensor weight scale")
    storage = (
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
    )
    if any(value is not None and value.device != weight.device for value in storage):
        raise ValueError(f"{operation} storage tensors must share a device")


def _validate_addmm(
    weight: PiperNVFP4Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    operation = "ConvRot NVFP4 addmm_"
    _validate_storage(weight, operation=operation)
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError(
            "ConvRot NVFP4 addmm_ matrices must be 2-D, "
            f"got shapes {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    expected_mat1 = (weight.shape[0], mat2.shape[0])
    expected_mat2 = (mat1.shape[1], weight.shape[1])
    if tuple(mat1.shape) != expected_mat1 or tuple(mat2.shape) != expected_mat2:
        raise ValueError(
            "ConvRot NVFP4 addmm_ shape mismatch: expected "
            f"mat1 {expected_mat1} and mat2 {expected_mat2} for weight "
            f"{tuple(weight.shape)}, got {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    validate_update_operands(
        (mat1, mat2),
        device=weight.device,
        dtype=weight.orig_dtype,
        differentiable_storage=(weight.scale, weight.per_tensor_scale),
        operation=operation,
    )


def _validate_add(
    weight: PiperNVFP4Tensor,
    update: torch.Tensor,
) -> None:
    operation = "ConvRot NVFP4 add_"
    _validate_storage(weight, operation=operation)
    if tuple(update.shape) != tuple(weight.shape):
        raise ValueError(
            f"{operation} shape mismatch: expected update {tuple(weight.shape)}, "
            f"got {tuple(update.shape)}"
        )
    validate_update_operands(
        (update,),
        device=weight.device,
        dtype=weight.orig_dtype,
        differentiable_storage=(weight.scale, weight.per_tensor_scale),
        operation=operation,
    )


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


def addmm_(
    weight: PiperNVFP4Tensor,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: int | float | complex = 1,
    alpha: int | float | complex = 1,
    rounding_seed: int | None = None,
) -> None:
    """Apply a logical addmm update and refill the existing packed storage."""
    _validate_addmm(weight, mat1, mat2)
    operation = "ConvRot NVFP4 addmm_"
    beta_float = validate_real_scalar(beta, "beta", operation=operation)
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    validate_rounding_seed(rounding_seed, operation=operation)
    if beta_float == 1 and alpha_float == 0:
        return

    if beta_float == 0:
        rotated_weight = torch.zeros(
            weight.shape,
            device=weight.device,
            dtype=weight.orig_dtype,
        )
    else:
        # Bypass ConvRot's logical dequantize override and remain in the
        # physical rotated basis used by the packed storage.
        rotated_weight = PiperNVFP4Tensor.dequantize(weight, weight.orig_dtype)
    rotated_update = rotate_groups(mat2, group_size)
    merged = torch.addmm(
        rotated_weight,
        mat1,
        rotated_update,
        beta=beta_float,
        alpha=alpha_float,
    )

    _refill_(weight, merged, rounding_seed)


def add_(
    weight: PiperNVFP4Tensor,
    group_size: int,
    update: torch.Tensor,
    *,
    alpha: int | float | complex = 1,
    rounding_seed: int | None = None,
) -> None:
    """Apply a logical dense update and refill the existing packed storage."""
    _validate_add(weight, update)
    operation = "ConvRot NVFP4 add_"
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    validate_rounding_seed(rounding_seed, operation=operation)
    if alpha_float == 0:
        return

    # Stay in the physical rotated basis used by the packed storage. Rotation
    # is linear, so this equals rotating the logical sum before requantization.
    rotated_weight = PiperNVFP4Tensor.dequantize(weight, weight.orig_dtype)
    rotated_update = rotate_groups(update, group_size)
    merged = torch.add(rotated_weight, rotated_update, alpha=alpha_float)
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
            merged,
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


__all__ = ["add_", "addmm_"]
