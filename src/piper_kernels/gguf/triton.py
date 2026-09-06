"""Register-only GGUF decoding primitives for fused Triton consumers."""

# Triton's JIT launcher and constexpr branches are not fully represented by its
# Python typing surface.
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import triton
import triton.language as tl

from piper_kernels.linear.convrot import triton as convrot_backend

from ._types import GGUFQuantizationType

_F32 = tl.constexpr(int(GGUFQuantizationType.F32))
_F16 = tl.constexpr(int(GGUFQuantizationType.F16))
_Q4_0 = tl.constexpr(int(GGUFQuantizationType.Q4_0))
_Q4_1 = tl.constexpr(int(GGUFQuantizationType.Q4_1))
_Q5_0 = tl.constexpr(int(GGUFQuantizationType.Q5_0))
_Q5_1 = tl.constexpr(int(GGUFQuantizationType.Q5_1))
_Q8_0 = tl.constexpr(int(GGUFQuantizationType.Q8_0))
_Q2_K = tl.constexpr(int(GGUFQuantizationType.Q2_K))
_Q3_K = tl.constexpr(int(GGUFQuantizationType.Q3_K))
_Q4_K = tl.constexpr(int(GGUFQuantizationType.Q4_K))
_Q5_K = tl.constexpr(int(GGUFQuantizationType.Q5_K))
_Q6_K = tl.constexpr(int(GGUFQuantizationType.Q6_K))
_IQ4_NL = tl.constexpr(int(GGUFQuantizationType.IQ4_NL))
_IQ4_XS = tl.constexpr(int(GGUFQuantizationType.IQ4_XS))
_BF16 = tl.constexpr(int(GGUFQuantizationType.BF16))


@triton.jit
def _block_size(quant_type: tl.constexpr):
    if quant_type == _F32 or quant_type == _F16 or quant_type == _BF16:
        return 1
    if (
        quant_type == _Q4_0
        or quant_type == _Q4_1
        or quant_type == _Q5_0
        or quant_type == _Q5_1
        or quant_type == _Q8_0
        or quant_type == _IQ4_NL
    ):
        return 32
    return 256


@triton.jit
def _type_size(quant_type: tl.constexpr):
    if quant_type == _F32:
        return 4
    if quant_type == _F16 or quant_type == _BF16:
        return 2
    if quant_type == _Q4_0 or quant_type == _IQ4_NL:
        return 18
    if quant_type == _Q4_1:
        return 20
    if quant_type == _Q5_0:
        return 22
    if quant_type == _Q5_1:
        return 24
    if quant_type == _Q8_0:
        return 34
    if quant_type == _Q2_K:
        return 84
    if quant_type == _Q3_K:
        return 110
    if quant_type == _Q4_K:
        return 144
    if quant_type == _Q5_K:
        return 176
    if quant_type == _Q6_K:
        return 210
    return 136


@triton.jit
def packed_row_size(logical_width, quant_type: tl.constexpr):
    """Return the byte width of one GGUF row."""
    return logical_width // _block_size(quant_type) * _type_size(quant_type)


@triton.jit
def _load_u8(data_ptr, offsets, mask):
    return tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.uint32)


@triton.jit
def _load_u16(data_ptr, offsets, mask):
    low = _load_u8(data_ptr, offsets, mask)
    high = _load_u8(data_ptr, offsets + 1, mask)
    return (low | high << 8).to(tl.uint16)


@triton.jit
def _load_f16(data_ptr, offsets, mask):
    return _load_u16(data_ptr, offsets, mask).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _load_f32(data_ptr, offsets, mask):
    bits = _load_u8(data_ptr, offsets, mask)
    bits |= _load_u8(data_ptr, offsets + 1, mask) << 8
    bits |= _load_u8(data_ptr, offsets + 2, mask) << 16
    bits |= _load_u8(data_ptr, offsets + 3, mask) << 24
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _load_i8(data_ptr, offsets, mask):
    return tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.int8).to(tl.float32)


@triton.jit
def _scale_min(data_ptr, block_base, group, minimum: tl.constexpr, mask):
    """Decode one of the eight six-bit Q4_K/Q5_K scales or minima."""
    low_base = 8 if minimum else 4
    low_index = group % 4
    low = _load_u8(data_ptr, block_base + low_base + low_index, mask)
    high = _load_u8(data_ptr, block_base + 12 + low_index, mask)
    first_half = low & 0x3F
    if minimum:
        second_half = (high >> 4) | ((low >> 2) & 0x30)
    else:
        second_half = (high & 0x0F) | ((low >> 2) & 0x30)
    return tl.where(group < 4, first_half, second_half).to(tl.float32)


@triton.jit
def _iq4_value(code):
    """Decode the nonlinear IQ4 codebook without a global lookup table."""
    result = tl.where(code == 0, -127.0, -104.0)
    result = tl.where(code == 2, -83.0, result)
    result = tl.where(code == 3, -65.0, result)
    result = tl.where(code == 4, -49.0, result)
    result = tl.where(code == 5, -35.0, result)
    result = tl.where(code == 6, -22.0, result)
    result = tl.where(code == 7, -10.0, result)
    result = tl.where(code == 8, 1.0, result)
    result = tl.where(code == 9, 13.0, result)
    result = tl.where(code == 10, 25.0, result)
    result = tl.where(code == 11, 38.0, result)
    result = tl.where(code == 12, 53.0, result)
    result = tl.where(code == 13, 69.0, result)
    result = tl.where(code == 14, 89.0, result)
    return tl.where(code == 15, 113.0, result)


@triton.jit
def load_values(
    data_ptr,
    row_byte_offset,
    logical_offsets,
    mask,
    quant_type: tl.constexpr,
):
    """Decode arbitrary logical offsets from one packed GGUF row."""
    block_size: tl.constexpr = _block_size(quant_type)
    type_size: tl.constexpr = _type_size(quant_type)
    block = logical_offsets // block_size
    index = logical_offsets % block_size
    block_base = row_byte_offset + block * type_size

    if quant_type == _F32:
        return _load_f32(data_ptr, block_base, mask)
    if quant_type == _F16:
        return _load_f16(data_ptr, block_base, mask)
    if quant_type == _BF16:
        bits = _load_u16(data_ptr, block_base, mask).to(tl.uint32) << 16
        return bits.to(tl.float32, bitcast=True)

    scale = _load_f16(data_ptr, block_base, mask)
    if quant_type == _Q8_0:
        return scale * _load_i8(data_ptr, block_base + 2 + index, mask)

    if (
        quant_type == _Q4_0
        or quant_type == _Q4_1
        or quant_type == _Q5_0
        or quant_type == _Q5_1
        or quant_type == _IQ4_NL
    ):
        minimum_bytes: tl.constexpr = 2 if quant_type == _Q4_1 or quant_type == _Q5_1 else 0
        high_bytes: tl.constexpr = 4 if quant_type == _Q5_0 or quant_type == _Q5_1 else 0
        quantized_base: tl.constexpr = 2 + minimum_bytes + high_bytes
        packed_index = index % 16
        packed = _load_u8(data_ptr, block_base + quantized_base + packed_index, mask)
        low = tl.where(index < 16, packed & 0x0F, packed >> 4)
        if quant_type == _Q5_0 or quant_type == _Q5_1:
            high_base: tl.constexpr = 2 + minimum_bytes
            high = _load_u8(data_ptr, block_base + high_base + index // 8, mask)
            low |= ((high >> (index % 8)) & 1) << 4
        if quant_type == _Q4_0:
            return scale * (low.to(tl.float32) - 8.0)
        if quant_type == _Q5_0:
            return scale * (low.to(tl.float32) - 16.0)
        if quant_type == _Q4_1 or quant_type == _Q5_1:
            minimum = _load_f16(data_ptr, block_base + 2, mask)
            return scale * low.to(tl.float32) + minimum
        return scale * _iq4_value(low)

    if quant_type == _Q6_K:
        chunk = index // 32
        position = index % 32
        low_index = (chunk // 4) * 64 + (chunk % 2) * 32 + position
        low = _load_u8(data_ptr, block_base + low_index, mask)
        low = (low >> tl.where(chunk % 4 >= 2, 4, 0)) & 0x0F
        high_index = (chunk // 4) * 32 + position
        high = _load_u8(data_ptr, block_base + 128 + high_index, mask)
        high = (high >> ((chunk % 4) * 2)) & 0x03
        quantized = (low | high << 4).to(tl.float32) - 32.0
        subscale = _load_i8(data_ptr, block_base + 192 + index // 16, mask)
        super_scale = _load_f16(data_ptr, block_base + 208, mask)
        return super_scale * subscale * quantized

    if quant_type == _Q4_K or quant_type == _Q5_K:
        group = index // 32
        position = index % 32
        quantized_base: tl.constexpr = 48 if quant_type == _Q5_K else 16
        packed = _load_u8(
            data_ptr,
            block_base + quantized_base + (group // 2) * 32 + position,
            mask,
        )
        quantized = (packed >> ((group % 2) * 4)) & 0x0F
        if quant_type == _Q5_K:
            high = _load_u8(data_ptr, block_base + 16 + position, mask)
            quantized |= ((high >> group) & 1) << 4
        block_scale = _scale_min(data_ptr, block_base, group, False, mask)
        block_minimum = _scale_min(data_ptr, block_base, group, True, mask)
        minimum_scale = _load_f16(data_ptr, block_base + 2, mask)
        return scale * block_scale * quantized.to(tl.float32) - minimum_scale * block_minimum

    if quant_type == _Q3_K:
        chunk = index // 32
        position = index % 32
        low = _load_u8(
            data_ptr,
            block_base + 32 + (chunk // 4) * 32 + position,
            mask,
        )
        low = (low >> ((chunk % 4) * 2)) & 3
        high = _load_u8(data_ptr, block_base + position, mask)
        high = (((high >> chunk) & 1) ^ 1) << 2
        group = index // 16
        low_scale = _load_u8(data_ptr, block_base + 96 + group % 8, mask)
        low_scale = (low_scale >> tl.where(group < 8, 0, 4)) & 0x0F
        high_scale = _load_u8(data_ptr, block_base + 104 + group % 4, mask)
        high_scale = (high_scale >> ((group // 4) * 2)) & 0x03
        subscale = (low_scale | high_scale << 4).to(tl.float32) - 32.0
        super_scale = _load_f16(data_ptr, block_base + 108, mask)
        return super_scale * subscale * (low.to(tl.float32) - high.to(tl.float32))

    if quant_type == _Q2_K:
        group = index // 16
        chunk = index // 32
        position = index % 32
        packed = _load_u8(
            data_ptr,
            block_base + 16 + (chunk // 4) * 32 + position,
            mask,
        )
        quantized = (packed >> ((chunk % 4) * 2)) & 3
        subscales = _load_u8(data_ptr, block_base + group, mask)
        super_scale = _load_f16(data_ptr, block_base + 80, mask)
        minimum_scale = _load_f16(data_ptr, block_base + 82, mask)
        return super_scale * (subscales & 0x0F).to(tl.float32) * quantized.to(
            tl.float32
        ) - minimum_scale * (subscales >> 4).to(tl.float32)

    # IQ4_XS is the only remaining supported type.
    group = index // 32
    position = index % 32
    packed = _load_u8(data_ptr, block_base + 8 + group * 16 + position % 16, mask)
    code = tl.where(position < 16, packed & 0x0F, packed >> 4)
    low_scale = _load_u8(data_ptr, block_base + 4 + group // 2, mask)
    low_scale = (low_scale >> ((group % 2) * 4)) & 0x0F
    high_scale_bits = _load_u16(data_ptr, block_base + 2, mask).to(tl.uint32)
    high_scale = (high_scale_bits >> (group * 2)) & 0x03
    subscale = (low_scale | high_scale << 4).to(tl.float32) - 32.0
    return scale * subscale * _iq4_value(code)


@triton.jit
def load_rotated_chunk(
    data_ptr,
    row_byte_offset,
    row_width,
    chunk_start: tl.constexpr,
    chunk_offsets,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    quant_type: tl.constexpr,
):
    """Decode and rotate one group-aligned slice without global dense storage."""
    offsets = chunk_start + chunk_offsets
    mask = offsets < row_width
    values = load_values(
        data_ptr,
        row_byte_offset,
        offsets,
        mask,
        quant_type,
    )
    values = convrot_backend.rotate_hadamard_groups(values, chunk_size, group_size)
    values *= inverse_sqrt_group
    return values


__all__ = ["load_rotated_chunk", "load_values", "packed_row_size"]
