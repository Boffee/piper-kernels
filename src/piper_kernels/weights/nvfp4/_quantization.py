"""Portable NVFP4 encoding and decoding with E4M3 subnormal block scales.

Global scales are decoding multipliers, the reciprocal of FlashInfer's encoding
scale. Each block scale is round-to-nearest E4M3 of ``amax / (6 * global_scale)``;
there is no smallest-normal floor. A nonzero block whose scale rounds to zero
uses saturated codes, matching FlashInfer, and reconstructs to zero. An all-zero
block uses zero codes. Both cases avoid NaNs during packing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from piper_kernels.weights.nvfp4 import _layout


def f32_to_e2m1_unpacked(values: torch.Tensor) -> torch.Tensor:
    """Round FP32 values to E2M1 nibbles (stored in low 4 bits of ``uint8``)."""
    fp32 = values.to(torch.float32)
    magnitude = fp32.abs()
    sign = torch.signbit(fp32).to(torch.uint8) << 3
    code = torch.zeros(fp32.shape, dtype=torch.uint8, device=fp32.device)
    code = torch.where(magnitude > 0.25, 1, code)
    code = torch.where(magnitude >= 0.75, 2, code)
    code = torch.where(magnitude > 1.25, 3, code)
    code = torch.where(magnitude >= 1.75, 4, code)
    code = torch.where(magnitude > 2.5, 5, code)
    code = torch.where(magnitude >= 3.5, 6, code)
    code = torch.where(magnitude > 5.0, 7, code)
    return sign | code


def e2m1_unpacked_to_f32(codes: torch.Tensor) -> torch.Tensor:
    """Decode unpacked E2M1 codes (0..15 in ``uint8``) to IEEE FP32 values."""
    code_i32 = codes.to(torch.int32)
    sign = (code_i32 & 8) << 28
    magnitude = code_i32 & 7
    bits = sign | torch.where(
        magnitude != 0,
        (magnitude + 252 - (magnitude == 1).to(torch.int32)) << 22,
        0,
    )
    return bits.view(torch.float32)


def pack_uint4(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack pairs of low-4-bit ``uint8`` nibbles (even=low, odd=high) along the last axis."""
    shape = unpacked.shape
    if shape[-1] % 2:
        raise ValueError(f"pack_uint4 requires an even trailing dimension, got {shape[-1]}")
    flat = unpacked.contiguous().view(-1)
    return (flat[::2] | (flat[1::2] << 4)).view(*shape[:-1], shape[-1] // 2)


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack low-first packed ``uint8`` bytes into low-4-bit ``uint8`` nibbles."""
    contiguous = packed.contiguous()
    first = (contiguous & 0x0F).to(torch.uint8)
    second = (contiguous >> 4).to(torch.uint8)
    return torch.stack((first, second), dim=-1).view(
        *contiguous.shape[:-1], contiguous.shape[-1] * 2
    )


def quantize_nvfp4(
    source: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    *,
    swizzled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed low-first E2M1 data and FP8 scales for a matrix or matrix batch."""
    if source.ndim not in (2, 3) or source.shape[-1] % _layout.BLOCK_SIZE:
        raise ValueError("NVFP4 requires matrices with a feature width divisible by 16")
    values = source.float().reshape(*source.shape[:-1], -1, _layout.BLOCK_SIZE)
    block_amax = values.abs().amax(dim=-1)
    global_scale = per_tensor_scale
    if global_scale is None:
        global_scale = source.new_ones((), dtype=torch.float32)
    effective_global_scale = torch.where(global_scale == 0, 1.0, global_scale)
    scales = (
        (block_amax * (1.0 / 6.0) / effective_global_scale).clamp(max=448).to(torch.float8_e4m3fn)
    )
    reciprocal = (1.0 / effective_global_scale) / scales.float()
    normalized = values * reciprocal.unsqueeze(-1)
    # A zero block scale produces signed infinities for nonzero values and NaN
    # for zeros. Saturate these in place; the FP4 converter saturates finite values.
    normalized.nan_to_num_(nan=6.0, posinf=6.0, neginf=-6.0)
    qdata = pack_uint4(f32_to_e2m1_unpacked(normalized.reshape(source.shape)))
    # All-zero blocks use zero codes, including signed-zero inputs. Correct the
    # packed bytes instead of selecting another source-sized FP32 tensor.
    qdata.reshape(*block_amax.shape, _layout.QDATA_BLOCK_SIZE).masked_fill_(
        (block_amax == 0).unsqueeze(-1), 0
    )
    if swizzled:
        rows, features = source.shape[-2:]
        # Each expert owns complete 128-row tiles, including its own padding.
        scales = (
            _layout.to_blocked(scales)
            if source.ndim == 2
            else torch.stack([_layout.to_blocked(expert) for expert in scales])
        )
        scales = scales.reshape(*source.shape[:-2], *_layout.scale_shape(rows, features))
    return qdata, scales


def unswizzle_nvfp4_scales(
    scale: torch.Tensor,
    logical_shape: Sequence[int | torch.SymInt],
    *,
    swizzled: bool,
    block_size: int = _layout.BLOCK_SIZE,
) -> torch.Tensor:
    """Recover canonical FP32 block scales aligned with logical matrix axes."""
    leading_dims = logical_shape[:-2]
    rows = cast(int, logical_shape[-2])
    features = cast(int, logical_shape[-1])
    scale_cols = features // block_size
    if swizzled:
        if len(logical_shape) == 3:
            scale_e4m3 = torch.stack(
                [_layout.from_blocked(expert, rows, scale_cols) for expert in scale]
            )
        else:
            scale_e4m3 = _layout.from_blocked(scale, rows, scale_cols)
    else:
        scale_e4m3 = scale.view(*leading_dims, rows, scale_cols)
    return scale_e4m3.to(torch.float32)


def dequantize_nvfp4(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    per_tensor_scale: torch.Tensor | None = None,
    swizzled: bool = False,
    high_first: bool = False,
    block_size: int = _layout.BLOCK_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode packed NVFP4 storage into a high-precision floating tensor."""
    logical_shape = _layout.tensor_size_fp4x2_to_hp(qdata.shape, transposed=False)
    leading_dims = logical_shape[:-2]
    rows = cast(int, logical_shape[-2])
    features = cast(int, logical_shape[-1])
    packed = _layout.swap_packed_pairs(qdata) if high_first else qdata
    unpacked = unpack_uint4(packed.contiguous().view(torch.uint8))
    values = e2m1_unpacked_to_f32(unpacked).view(
        *leading_dims, rows, features // block_size, block_size
    )
    scales = unswizzle_nvfp4_scales(
        scale,
        logical_shape,
        swizzled=swizzled,
        block_size=block_size,
    )
    if per_tensor_scale is not None:
        scales = per_tensor_scale * scales
    return (values * scales.unsqueeze(-1)).view(*leading_dims, rows, features).to(output_dtype)


__all__ = [
    "dequantize_nvfp4",
    "e2m1_unpacked_to_f32",
    "f32_to_e2m1_unpacked",
    "pack_uint4",
    "quantize_nvfp4",
    "unpack_uint4",
    "unswizzle_nvfp4_scales",
]
