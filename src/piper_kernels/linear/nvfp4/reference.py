"""Portable PyTorch projections and updates for plain and ConvRot NVFP4."""

from __future__ import annotations

from typing import cast

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
from piper_kernels.linear._input_activations import apply_input_activation
from piper_kernels.linear.convrot._rotation import rotate_groups

from . import _layout
from ._typing import NVFP4Storage
from .tensor import _MIN_PER_TENSOR_SCALE, PiperNVFP4Tensor


def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
    high_first: bool = False,
    *,
    group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare activations using only PyTorch rotation and quantization operations."""
    activated = apply_input_activation(input, activation_fn)
    if group_size:
        activated = rotate_groups(activated.float(), group_size)
    flattened = activated.reshape(-1, activated.shape[-1]).contiguous()
    if flattened.shape[0] == 0:
        if dynamic_activation_scale:
            global_scale = input.new_zeros((), dtype=torch.float32)
        elif activation_per_tensor_scale is not None:
            global_scale = activation_per_tensor_scale.clone()
        else:
            raise ValueError("static NVFP4 preparation requires a per-tensor scale")
        return (
            input.new_empty(_layout.qdata_shape(0, flattened.shape[1]), dtype=torch.uint8),
            input.new_empty(_layout.scale_shape(0, flattened.shape[1]), dtype=torch.float8_e4m3fn),
            global_scale,
        )
    global_scale = (
        per_tensor_amax_to_scale(flattened.abs().amax())
        if dynamic_activation_scale
        else activation_per_tensor_scale
    )
    if global_scale is None:
        raise ValueError("static NVFP4 preparation requires a per-tensor scale")
    encoding_scale = torch.where(global_scale == 0, torch.ones_like(global_scale), global_scale)
    encoded = cast(
        NVFP4Storage,
        TorchAONVFP4Tensor.to_nvfp4(
            flattened.float(),
            block_size=_layout.BLOCK_SIZE,
            per_tensor_scale=encoding_scale,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        ),
    )
    qdata = _layout.swap_packed_pairs(encoded.qdata) if high_first else encoded.qdata
    return qdata, encoded.scale, global_scale


def linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize block scales, accumulate and apply affine terms in FP32, then cast."""
    if input_qdata.shape[0] == 0:
        return input_qdata.new_empty((0, weight_qdata.shape[0]), dtype=logical_dtype)

    def dequantize(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return TorchAONVFP4Tensor(
            qdata,
            scale,
            _layout.BLOCK_SIZE,
            torch.float32,
            is_swizzled_scales=True,
        ).dequantize(torch.float32)

    # Both operands have the same nibble ordering, so their dot product also
    # agrees when each adjacent pair is swapped in both dequantized tensors.
    result = dequantize(input_qdata, input_scale) @ dequantize(weight_qdata, weight_scale).T
    global_scale = input_per_tensor_scale
    if weight_per_tensor_scale is not None:
        global_scale = global_scale * weight_per_tensor_scale
    result.mul_(global_scale)
    if bias is not None:
        result.add_(bias.float())
    return result.to(logical_dtype)


def linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_activation_scale: bool,
    high_first: bool = False,
    *,
    group_size: int = 0,
) -> torch.Tensor:
    """Run an independent PyTorch W4A4 projection, optionally in the ConvRot basis."""
    prepared = prepare_input(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
        high_first=high_first,
        group_size=group_size,
    )
    return linear_prepared(
        *prepared,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        input.dtype,
    ).reshape(*input.shape[:-1], weight_qdata.shape[0])


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
            dtype=torch.float32,
        )
    else:
        # Bypass ConvRot's logical dequantize override and remain in the
        # physical rotated basis used by the packed storage.
        rotated_weight = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    rotated_update = rotate_groups(mat2.float(), group_size) if group_size else mat2.float()
    merged = torch.addmm(
        rotated_weight,
        mat1.float(),
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
    rotated_weight = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    rotated_update = rotate_groups(update.float(), group_size) if group_size else update.float()
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


__all__ = ["add_", "addmm_", "linear", "linear_prepared", "prepare_input"]
