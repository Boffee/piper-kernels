"""NVFP4 encoding, decoding, and global scale primitives."""

# pyright: reportCallIssue=false, reportIndexIssue=false
from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.weights.nvfp4 import _layout

_NVFP4_BLOCK_SIZE = _layout.BLOCK_SIZE
_NVFP4_QDATA_BLOCK_SIZE = _layout.QDATA_BLOCK_SIZE
_NVFP4_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_BLOCK_SIZE)
_NVFP4_QDATA_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_QDATA_BLOCK_SIZE)
_AMAX_REDUCTION_BLOCK_SIZE = 1_024
# Preserve the existing one-CTA final reduction for small inputs.
_AMAX_FINAL_REDUCTION_SIZE = 8_192


@triton.jit
def _decode_fp4_code(code):
    magnitude = code & 0x7
    value = tl.where(
        magnitude <= 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
    )
    return tl.where(code & 0x8 == 0, value, -value)


@triton.jit
def _decode_fp4(packed, logical_offsets):
    code = tl.where(logical_offsets % 2 == 0, packed & 0xF, packed >> 4)
    return _decode_fp4_code(code)


@triton.jit
def swizzled_scale_offsets(rows, scale_columns, column_blocks: tl.constexpr):
    row_block = rows // 128
    row_inner = rows % 128
    column_block = scale_columns // 4
    return (
        ((row_block * column_blocks + column_block) * 32 + row_inner % 32) * 16
        + (row_inner // 32) * 4
        + scale_columns % 4
    )


@triton.jit
def pack_e2m1_pairs(low, high):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 packed;
            cvt.rn.satfinite.e2m1x2.f32 packed, $2, $1;
            cvt.u32.u8 $0, packed;
        }
        """,
        constraints="=r,f,f",
        args=[low, high],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _normalize_nvfp4_blocks(values, per_tensor_scale):
    """Select canonical FP8 block scales and normalize values for E2M1 packing."""
    values = values.to(tl.float32)
    block_amax = tl.max(tl.abs(values), axis=1)
    encoded_scale = tl.clamp(
        block_amax * (1.0 / 6.0) / per_tensor_scale,  # pyright: ignore[reportOperatorIssue]
        0.015625,
        448.0,
    ).to(tl.float8e4nv)
    reciprocal_scale = (1.0 / per_tensor_scale) / encoded_scale.to(tl.float32)
    scaled = tl.clamp(values * reciprocal_scale[:, None], -6.0, 6.0)
    return scaled, encoded_scale


@triton.jit
def encode_nvfp4_blocks(
    values,
    per_tensor_scale,
    block_count: tl.constexpr,
    high_first: tl.constexpr,
):
    """Encode FP32 values into E2M1 pairs and FP8 block scales."""
    scaled, encoded_scale = _normalize_nvfp4_blocks(  # pyright: ignore[reportGeneralTypeIssues]
        values, per_tensor_scale
    )
    paired = tl.reshape(
        scaled,
        (block_count, _NVFP4_QDATA_BLOCK_SIZE_TL, 2),
    )
    low, high = tl.split(paired)
    if high_first:
        return pack_e2m1_pairs(high, low), encoded_scale
    return pack_e2m1_pairs(low, high), encoded_scale


@triton.jit
def _amax_partial_kernel(
    input_ptr,
    partial_ptr,
    elements,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    values = tl.load(input_ptr + offsets, mask=offsets < elements, other=0.0)
    tl.store(partial_ptr + tl.program_id(0), tl.max(tl.abs(values).to(tl.float32), axis=0))


@triton.jit
def _amax_scale_kernel(
    input_ptr,
    per_tensor_scale_ptr,
    elements,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    values = tl.load(input_ptr + offsets, mask=offsets < elements, other=0.0)
    amax = tl.max(tl.abs(values).to(tl.float32), axis=0)
    tl.store(per_tensor_scale_ptr, amax * (1.0 / (448.0 * 6.0)))  # pyright: ignore[reportOperatorIssue]


def dynamic_scale(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce a logical tensor to its exact NVFP4 global scale."""
    if input.numel() < 1:
        raise ValueError("dynamic NVFP4 scale requires a nonempty tensor")
    if not input.is_floating_point():
        raise ValueError("dynamic NVFP4 scale requires a floating tensor")
    if out is None:
        per_tensor_scale = torch.empty((), device=input.device, dtype=torch.float32)
    else:
        if (
            out.shape != ()
            or out.dtype is not torch.float32
            or out.device != input.device
            or not out.is_contiguous()
        ):
            raise ValueError("dynamic NVFP4 scale output must be a contiguous FP32 scalar")
        per_tensor_scale = out

    values = input.contiguous().view(-1)
    with device_context(input.device):
        while values.numel() > _AMAX_FINAL_REDUCTION_SIZE:
            partial_count = (
                values.numel() + _AMAX_REDUCTION_BLOCK_SIZE - 1
            ) // _AMAX_REDUCTION_BLOCK_SIZE
            partial = torch.empty(partial_count, device=input.device, dtype=torch.float32)
            _amax_partial_kernel[(partial_count,)](
                values,
                partial,
                values.numel(),
                block_size=_AMAX_REDUCTION_BLOCK_SIZE,
                num_warps=8,
            )
            values = partial
        _amax_scale_kernel[(1,)](
            values,
            per_tensor_scale,
            values.numel(),
            block_size=triton.next_power_of_2(values.numel()),
            num_warps=8,
        )
        return per_tensor_scale
