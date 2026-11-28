"""Unified little-endian Triton decode of trellis-coded blocks into INT8 or FP32.

Every weight decodes independently: its state is a linear 16-bit little-endian
window inside a block whose trailing ``WRAP_GUARD_BYTES`` (2) duplicate its first
two bytes, eliminating wrap-around logic and byte swaps across all targets.
"""

# Triton's JIT launcher and constexpr arguments are not fully represented by its
# Python typing surface.
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from ._amd.isa import sad_u8_lcg_numerator_i32
from ._layout import (
    CODE_OFFSET,
    CODE_RECIPROCAL,
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    WRAP_GUARD_BYTES,
    TrellisLayout,
)
from ._nvidia.isa import dp4a_lcg_numerator_i32

_BLOCKS_PER_PROGRAM = 4
_DEFAULT_DECODE_WARPS = 2
_WRAP_GUARD_BYTES = tl.constexpr(WRAP_GUARD_BYTES)
_LCG_MULTIPLIER = tl.constexpr(LCG_MULTIPLIER)
_LCG_INCREMENT = tl.constexpr(LCG_INCREMENT)
_CODE_OFFSET = tl.constexpr(CODE_OFFSET)
_CODE_RECIPROCAL = tl.constexpr(CODE_RECIPROCAL)


@triton.jit
def _decode_e4m3(code):
    """Exact FP32 value of FP8 E4M3 bits, without relying on backend FP8 support."""
    sign = code >> 7
    exponent = (code >> 3) & 15
    mantissa = code & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    subnormal = mantissa.to(tl.float32) * 0.001953125  # mantissa · 2^-9
    magnitude = tl.where(exponent == 0, subnormal, normal)
    return tl.where(sign != 0, -magnitude, magnitude)


@triton.jit
def _decode_lcg_numerator_i32(
    states,
    use_dp4a: tl.constexpr,
    use_sad_u8: tl.constexpr = False,
):
    """Compute signed LCG byte-sum numerators from 16-bit trellis states."""
    x = states * _LCG_MULTIPLIER + _LCG_INCREMENT
    if use_dp4a:
        return dp4a_lcg_numerator_i32(x)
    if use_sad_u8:
        return sad_u8_lcg_numerator_i32(x)
    return ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)).to(
        tl.int32
    ) - _CODE_OFFSET


@triton.jit
def _load_u32_word(
    u32_ptr,
    byte_ptr,
    idx,
    full_words,
    limit_bytes,
    valid,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    evict_first: tl.constexpr = False,
    clamped_load: tl.constexpr = False,
):
    """Load an LE uint32 word within ``limit_bytes`` (clamping assumes ``full_words >= 1``)."""
    if clamped_load and ptr_aligned and not has_tail_bytes:
        safe_idx = tl.minimum(idx, full_words - 1)
        if evict_first:
            w = tl.load(u32_ptr + safe_idx, eviction_policy="evict_first")
        else:
            w = tl.load(u32_ptr + safe_idx)
    elif ptr_aligned:
        if evict_first:
            w = tl.load(
                u32_ptr + idx,
                mask=valid & (idx < full_words),
                other=0,
                eviction_policy="evict_first",
            )
        else:
            w = tl.load(u32_ptr + idx, mask=valid & (idx < full_words), other=0)
        if has_tail_bytes:
            is_tail = valid & (idx == full_words)
            base = idx << 2
            b0 = tl.load(byte_ptr + base, mask=is_tail & (base < limit_bytes), other=0).to(
                tl.uint32
            )
            b1 = tl.load(
                byte_ptr + base + 1, mask=is_tail & ((base + 1) < limit_bytes), other=0
            ).to(tl.uint32)
            b2 = tl.load(
                byte_ptr + base + 2, mask=is_tail & ((base + 2) < limit_bytes), other=0
            ).to(tl.uint32)
            w = tl.where(is_tail, b0 | (b1 << 8) | (b2 << 16), w)
    else:
        base = idx << 2
        b0 = tl.load(byte_ptr + base, mask=valid & (base < limit_bytes), other=0).to(tl.uint32)
        b1 = tl.load(byte_ptr + base + 1, mask=valid & ((base + 1) < limit_bytes), other=0).to(
            tl.uint32
        )
        b2 = tl.load(byte_ptr + base + 2, mask=valid & ((base + 2) < limit_bytes), other=0).to(
            tl.uint32
        )
        b3 = tl.load(byte_ptr + base + 3, mask=valid & ((base + 3) < limit_bytes), other=0).to(
            tl.uint32
        )
        w = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    return w


@triton.jit
def _extract_16_le_u64(bits64, rel_bit):
    """Extract 16 little-endian bits when the state lies inside ``bits64 = (w1 << 32) | w0``."""
    return (bits64 >> rel_bit.to(tl.uint64)).to(tl.uint32) & 0xFFFF


@triton.jit
def _extract_16_le_3word(w0, w1, w2, rel_bit):
    """Extract 16 little-endian bits across three uint32 words ``w0, w1, w2``."""
    lo = tl.where(rel_bit < 32, w0, w1)
    hi = tl.where(rel_bit < 32, w1, w2)
    sh = (rel_bit & 31).to(tl.uint64)
    return (((hi.to(tl.uint64) << 32) | lo.to(tl.uint64)) >> sh).to(tl.uint32) & 0xFFFF


@triton.jit
def _le_step_bit(step, step_bits, num_wide_steps, denom, block_size, exact_pow2: tl.constexpr):
    """Compute bit start offset for one step."""
    c = step if exact_pow2 else tl.minimum(step, block_size - 1)
    return step_bits * c + (c + 1) * num_wide_steps // denom


@triton.jit
def _le_step_bits_4(
    first_step,
    step_bits,
    num_wide_steps,
    denom,
    block_size,
    exact_pow2: tl.constexpr,
):
    """Compute bit start offsets for a 4-step group."""
    return (
        _le_step_bit(first_step, step_bits, num_wide_steps, denom, block_size, exact_pow2),
        _le_step_bit(first_step + 1, step_bits, num_wide_steps, denom, block_size, exact_pow2),
        _le_step_bit(first_step + 2, step_bits, num_wide_steps, denom, block_size, exact_pow2),
        _le_step_bit(first_step + 3, step_bits, num_wide_steps, denom, block_size, exact_pow2),
    )


@triton.jit
def _decode_words_4(
    u32_ptr,
    byte_ptr,
    base_byte,
    first_byte,
    full_words,
    limit_bytes,
    valid,
    needs_w2: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    evict_first: tl.constexpr = False,
    clamped_load: tl.constexpr = False,
):
    """Load LE uint32 words covering a 4-step group and return ``(w0, w1, w2, bit_bias)``."""
    blk_word_base = base_byte >> 2
    blk_align = (base_byte & 3).to(tl.int32)
    blk_align_bits = blk_align << 3
    first_word = (blk_align + first_byte) >> 2
    w0_idx = blk_word_base + first_word
    bit_bias = blk_align_bits - (first_word << 5)
    w0 = _load_u32_word(
        u32_ptr, byte_ptr, w0_idx, full_words, limit_bytes, valid, has_tail_bytes, ptr_aligned,
        evict_first=evict_first, clamped_load=clamped_load,
    )  # fmt: skip
    w1 = _load_u32_word(
        u32_ptr, byte_ptr, w0_idx + 1, full_words, limit_bytes, valid, has_tail_bytes, ptr_aligned,
        evict_first=evict_first, clamped_load=clamped_load,
    )  # fmt: skip
    if needs_w2:
        w2 = _load_u32_word(
            u32_ptr, byte_ptr, w0_idx + 2, full_words, limit_bytes, valid,
            has_tail_bytes, ptr_aligned, evict_first=evict_first, clamped_load=clamped_load,
        )  # fmt: skip
    else:
        w2 = tl.zeros_like(w0)
    return w0, w1, w2, bit_bias


@triton.jit
def _decode_nums_from_words(
    w0,
    w1,
    w2,
    rel0,
    rel1,
    rel2,
    rel3,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    use_sad_u8: tl.constexpr = False,
):
    """Extract 4 16-bit LE states from ``w0, w1[, w2]`` and return float32 numerators."""
    bits64 = (w1.to(tl.uint64) << 32) | w0.to(tl.uint64)
    # Steps 0..2 always lie within [0, 58] bits of w0|w1 (rel0 <= 31, gap <= 12, window 16).
    st0 = _extract_16_le_u64(bits64, rel0)
    st1 = _extract_16_le_u64(bits64, rel1)
    st2 = _extract_16_le_u64(bits64, rel2)
    st3 = _extract_16_le_3word(w0, w1, w2, rel3) if needs_w2 else _extract_16_le_u64(bits64, rel3)
    return (
        _decode_lcg_numerator_i32(st0, use_dp4a, use_sad_u8).to(tl.float32),
        _decode_lcg_numerator_i32(st1, use_dp4a, use_sad_u8).to(tl.float32),
        _decode_lcg_numerator_i32(st2, use_dp4a, use_sad_u8).to(tl.float32),
        _decode_lcg_numerator_i32(st3, use_dp4a, use_sad_u8).to(tl.float32),
    )


@triton.jit
def _decode_group_4_nums(
    u32_ptr,
    byte_ptr,
    base_byte,
    sbit0,
    sbit1,
    sbit2,
    sbit3,
    first_byte,
    full_words,
    limit_bytes,
    valid,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    evict_first: tl.constexpr = False,
    use_sad_u8: tl.constexpr = False,
):
    """Load LE uint32 words for a 4-step group and return 4 float32 code numerators."""
    w0, w1, w2, bit_bias = _decode_words_4(
        u32_ptr, byte_ptr, base_byte, first_byte, full_words, limit_bytes, valid,
        needs_w2=needs_w2, has_tail_bytes=has_tail_bytes, ptr_aligned=ptr_aligned,
        evict_first=evict_first,
    )  # fmt: skip
    return _decode_nums_from_words(
        w0, w1, w2, bit_bias + sbit0, bit_bias + sbit1, bit_bias + sbit2, bit_bias + sbit3,
        needs_w2=needs_w2, use_dp4a=use_dp4a, use_sad_u8=use_sad_u8,
    )  # fmt: skip


@triton.jit
def _resolve_decode_addresses(
    row_widths,
    row_offsets,
    slot_offsets_ptr,
    blocks,
    blocks_per_row,
    bitstreams_bytes,
    denom,
    rows_per_slot,
    slot_bytes,
    block_mask,
    has_slot_offsets: tl.constexpr,
    per_slot_2d: tl.constexpr,
):
    """Resolve variable-row block base offsets, byte bounds, and step widths."""
    row = blocks // blocks_per_row
    block_in_row = blocks - row * blocks_per_row
    width = tl.load(row_widths + row, mask=block_mask, other=0).to(tl.int32)
    row_offset = tl.load(row_offsets + row, mask=block_mask, other=0).to(tl.int64)
    within_slot = row_offset * blocks_per_row + block_in_row.to(tl.int64) * width.to(tl.int64)
    if per_slot_2d:
        local_slot = row // rows_per_slot
        slot_base = local_slot.to(tl.int64) * tl.full((), slot_bytes, tl.int64)
        base_bytes = (slot_base + within_slot)[:, None]
        limit_bytes = tl.minimum(
            slot_base + tl.full((), slot_bytes, tl.int64), bitstreams_bytes.to(tl.int64)
        )[:, None]
        valid_base = block_mask & (width > _WRAP_GUARD_BYTES)
    elif has_slot_offsets:
        local_slot = row // rows_per_slot
        raw_slot_base = tl.load(slot_offsets_ptr + local_slot, mask=block_mask, other=0).to(
            tl.int64
        )
        valid_s = raw_slot_base >= 0
        slot_base = tl.where(valid_s, raw_slot_base, 0)
        base_bytes = (slot_base + within_slot)[:, None]
        limit_bytes = bitstreams_bytes.to(tl.int64)
        valid_base = block_mask & valid_s & (width > _WRAP_GUARD_BYTES)
    else:
        base_bytes = within_slot[:, None]
        limit_bytes = bitstreams_bytes.to(tl.int64)
        valid_base = block_mask & (width > _WRAP_GUARD_BYTES)
    eff_code_bits = tl.maximum(width - _WRAP_GUARD_BYTES, 0) * 8
    eff_step_bits = (eff_code_bits // denom)[:, None]
    eff_wide_steps = eff_code_bits[:, None] - eff_step_bits * denom
    return base_bytes, limit_bytes, valid_base, eff_step_bits, eff_wide_steps


@triton.jit
def _store_decoded_group(
    out_ptr,
    blocks,
    g,
    first_step,
    block_mask,
    group_valid,
    n0,
    n1,
    n2,
    n3,
    mult,
    block_size,
    denom,
    exact_pow2: tl.constexpr,
    q_u32_aligned: tl.constexpr,
    round_int8: tl.constexpr,
):
    """Scale decoded 4-step numerators and store as packed INT8 or FP32."""
    v0, v1, v2, v3 = n0 * mult, n1 * mult, n2 * mult, n3 * mult
    if round_int8 and q_u32_aligned:
        q0 = libdevice.rint(tl.clamp(v0, -127.0, 127.0)).to(tl.int8)
        q1 = libdevice.rint(tl.clamp(v1, -127.0, 127.0)).to(tl.int8)
        q2 = libdevice.rint(tl.clamp(v2, -127.0, 127.0)).to(tl.int8)
        q3 = libdevice.rint(tl.clamp(v3, -127.0, 127.0)).to(tl.int8)
        packed = (
            q0.to(tl.uint8).to(tl.uint32)
            | (q1.to(tl.uint8).to(tl.uint32) << 8)
            | (q2.to(tl.uint8).to(tl.uint32) << 16)
            | (q3.to(tl.uint8).to(tl.uint32) << 24)
        )
        q_u32_ptr = out_ptr.to(tl.pointer_type(tl.uint32))
        dst_u32 = blocks.to(tl.int64)[:, None] * (denom >> 2) + g[None, :]
        tl.store(q_u32_ptr + dst_u32, packed, mask=group_valid)
    else:
        if round_int8:
            o0 = libdevice.rint(tl.clamp(v0, -127.0, 127.0)).to(tl.int8)
            o1 = libdevice.rint(tl.clamp(v1, -127.0, 127.0)).to(tl.int8)
            o2 = libdevice.rint(tl.clamp(v2, -127.0, 127.0)).to(tl.int8)
            o3 = libdevice.rint(tl.clamp(v3, -127.0, 127.0)).to(tl.int8)
        else:
            o0, o1, o2, o3 = v0, v1, v2, v3
        dst_base = blocks.to(tl.int64)[:, None] * denom
        m_base = block_mask[:, None]
        for k in tl.static_range(4):  # pyright: ignore[reportGeneralTypeIssues]
            val = o0 if k == 0 else (o1 if k == 1 else (o2 if k == 2 else o3))
            m_k = m_base if exact_pow2 else (m_base & ((first_step + k) < block_size)[None, :])
            tl.store(out_ptr + dst_base + (first_step + k)[None, :], val, mask=m_k)


@triton.jit
def _decode_blocks_le(
    bitstreams,
    block_scales,
    out_ptr,
    row_widths,
    row_offsets,
    num_blocks,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    active_counts_ptr,
    slot_offsets_ptr,
    rows_per_slot,
    slot_bytes,
    tile_steps: tl.constexpr,
    tile_blocks: tl.constexpr,
    variable_rows: tl.constexpr,
    exact_pow2: tl.constexpr,
    filter_active: tl.constexpr,
    has_slot_offsets: tl.constexpr,
    per_slot_2d: tl.constexpr,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    q_u32_aligned: tl.constexpr,
    round_int8: tl.constexpr,
    use_sad_u8: tl.constexpr = False,
    clamped_load: tl.constexpr = False,
):
    blocks = tl.program_id(0) * tile_blocks + tl.arange(0, tile_blocks)
    block_mask = blocks < num_blocks
    if filter_active:
        local_slot_all = (blocks // blocks_per_row) // rows_per_slot
        slot_active = tl.load(active_counts_ptr + local_slot_all, mask=block_mask, other=0) > 0
        if tl.sum(slot_active.to(tl.int32)) == 0:
            return
        block_mask = block_mask & slot_active

    denom = tile_steps if exact_pow2 else block_size
    num_groups: tl.constexpr = tile_steps // 4
    g = tl.arange(0, num_groups)
    first_step = g * 4

    if variable_rows:
        base_bytes, limit_bytes, valid_base, eff_step_bits, eff_wide_steps = (
            _resolve_decode_addresses(
                row_widths, row_offsets, slot_offsets_ptr, blocks, blocks_per_row,
                bitstreams_bytes, denom, rows_per_slot, slot_bytes,
                block_mask, has_slot_offsets, per_slot_2d,
            )
        )  # fmt: skip
    else:
        base_bytes = (blocks.to(tl.int64) * bytes_per_block.to(tl.int64))[:, None]
        limit_bytes = bitstreams_bytes.to(tl.int64)
        valid_base = block_mask
        eff_step_bits = step_bits
        eff_wide_steps = num_wide_steps

    sbit0 = _le_step_bit(
        first_step[None, :], eff_step_bits, eff_wide_steps, denom, block_size, exact_pow2
    )
    first_byte = sbit0 >> 3
    group_valid = (
        block_mask[:, None]
        if exact_pow2
        else (block_mask[:, None] & (first_step < block_size)[None, :])
    )
    decode_valid = (
        valid_base[:, None]
        if exact_pow2
        else (valid_base[:, None] & (first_step < block_size)[None, :])
    )
    w0, w1, w2, bit_bias = _decode_words_4(
        bitstreams.to(tl.pointer_type(tl.uint32)), bitstreams, base_bytes, first_byte,
        limit_bytes >> 2, limit_bytes, decode_valid, needs_w2=needs_w2,
        has_tail_bytes=has_tail_bytes, ptr_aligned=ptr_aligned, clamped_load=clamped_load,
    )  # fmt: skip
    sbit1 = _le_step_bit(
        (first_step + 1)[None, :], eff_step_bits, eff_wide_steps, denom, block_size, exact_pow2
    )
    sbit2 = _le_step_bit(
        (first_step + 2)[None, :], eff_step_bits, eff_wide_steps, denom, block_size, exact_pow2
    )
    sbit3 = _le_step_bit(
        (first_step + 3)[None, :], eff_step_bits, eff_wide_steps, denom, block_size, exact_pow2
    )
    n0, n1, n2, n3 = _decode_nums_from_words(
        w0, w1, w2, bit_bias + sbit0, bit_bias + sbit1, bit_bias + sbit2, bit_bias + sbit3,
        needs_w2=needs_w2, use_dp4a=use_dp4a, use_sad_u8=use_sad_u8,
    )  # fmt: skip
    sc = tl.load(block_scales + blocks.to(tl.int64), mask=block_mask, other=0)
    mult = (_decode_e4m3(sc.to(tl.uint32)) * _CODE_RECIPROCAL)[:, None]
    _store_decoded_group(
        out_ptr, blocks, g, first_step, block_mask, group_valid, n0, n1, n2, n3, mult,
        block_size, denom, exact_pow2, q_u32_aligned, round_int8,
    )  # fmt: skip


@triton.jit(
    do_not_specialize=[
        "num_blocks",
        "blocks_per_row",
        "bitstreams_bytes",
        "block_size",
        "step_bits",
        "num_wide_steps",
        "bytes_per_block",
    ]
)
def decode_int8_kernel(
    bitstreams,
    block_scales,
    qdata,
    row_widths,
    row_offsets,
    num_blocks,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    tile_steps: tl.constexpr,
    tile_blocks: tl.constexpr,
    variable_rows: tl.constexpr,
    exact_pow2: tl.constexpr,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    q_u32_aligned: tl.constexpr,
    round_int8: tl.constexpr = True,
    use_sad_u8: tl.constexpr = False,
    clamped_load: tl.constexpr = False,
):
    """Unified LE decode of uniform or variable-row trellis blocks into INT8 or FP32."""
    _decode_blocks_le(
        bitstreams, block_scales, qdata, row_widths, row_offsets, num_blocks, blocks_per_row,
        bitstreams_bytes, block_size, step_bits, num_wide_steps, bytes_per_block, None, None, 0,
        0, tile_steps=tile_steps, tile_blocks=tile_blocks, variable_rows=variable_rows,
        exact_pow2=exact_pow2, filter_active=False, has_slot_offsets=False, per_slot_2d=False,
        needs_w2=needs_w2, use_dp4a=use_dp4a, has_tail_bytes=has_tail_bytes,
        ptr_aligned=ptr_aligned, q_u32_aligned=q_u32_aligned, round_int8=round_int8,
        use_sad_u8=use_sad_u8, clamped_load=clamped_load,
    )  # fmt: skip


@triton.jit(
    do_not_specialize=[
        "num_blocks",
        "blocks_per_row",
        "bitstreams_bytes",
        "block_size",
        "step_bits",
        "num_wide_steps",
        "bytes_per_block",
        "rows_per_slot",
        "slot_bytes",
    ]
)
def decode_slots_int8_kernel(
    bitstreams,
    block_scales,
    qdata,
    row_widths,
    row_offsets,
    active_counts_ptr,
    slot_offsets_ptr,
    num_blocks,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    rows_per_slot,
    slot_bytes,
    tile_steps: tl.constexpr,
    tile_blocks: tl.constexpr,
    variable_rows: tl.constexpr,
    exact_pow2: tl.constexpr,
    filter_active: tl.constexpr,
    has_slot_offsets: tl.constexpr,
    per_slot_2d: tl.constexpr,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    q_u32_aligned: tl.constexpr,
    use_sad_u8: tl.constexpr = False,
    clamped_load: tl.constexpr = False,
):
    """Unified LE decode of MoE slot-major uniform or variable-row trellis blocks into INT8."""
    _decode_blocks_le(
        bitstreams, block_scales, qdata, row_widths, row_offsets, num_blocks, blocks_per_row,
        bitstreams_bytes, block_size, step_bits, num_wide_steps, bytes_per_block,
        active_counts_ptr, slot_offsets_ptr, rows_per_slot, slot_bytes, tile_steps=tile_steps,
        tile_blocks=tile_blocks, variable_rows=variable_rows, exact_pow2=exact_pow2,
        filter_active=filter_active, has_slot_offsets=has_slot_offsets, per_slot_2d=per_slot_2d,
        needs_w2=needs_w2, use_dp4a=use_dp4a, has_tail_bytes=has_tail_bytes,
        ptr_aligned=ptr_aligned, q_u32_aligned=q_u32_aligned, round_int8=True,
        use_sad_u8=use_sad_u8, clamped_load=clamped_load,
    )  # fmt: skip


def _decode_num_warps(target: AcceleratorTarget) -> int:
    """Return the optimal warp count for Trellis block decoding on ``target``."""
    return 1 if target.is_hip_architecture(*GFX120X_ARCHITECTURES) else _DEFAULT_DECODE_WARPS


def _decode_stream_flags(
    target: AcceleratorTarget,
    stream_bytes: int,
    *,
    variable_rows: bool = False,
    ptr_aligned: bool = True,
    per_slot_2d: bool = False,
) -> tuple[bool, bool, bool, bool, int]:
    """Return ``(use_dp4a, use_sad_u8, has_tail_bytes, clamped_load, num_warps)``."""
    has_tail_bytes = per_slot_2d or (stream_bytes % 4 != 0)
    is_gfx120x = target.is_hip_architecture(*GFX120X_ARCHITECTURES)
    clamped_load = (
        is_gfx120x
        and (not variable_rows)
        and ptr_aligned
        and (not has_tail_bytes)
        and stream_bytes >= 4
    )
    return (
        target.is_nvidia_cuda,
        is_gfx120x,
        has_tail_bytes,
        clamped_load,
        1 if is_gfx120x else _DEFAULT_DECODE_WARPS,
    )


def _decode_launch_flags(
    bitstreams: torch.Tensor,
    out: torch.Tensor,
    block_size: int,
    *,
    variable_rows: bool = False,
    ptr_aligned: bool = True,
    per_slot_2d: bool = False,
    round_int8: bool = True,
) -> tuple[bool, bool, bool, bool, bool, int]:
    """Return ``(use_dp4a, use_sad_u8, clamped_load, has_tail_bytes, q_u32_aligned, num_warps)``."""
    target = AcceleratorTarget.from_device(bitstreams.device)
    use_dp4a, use_sad_u8, has_tail_bytes, clamped_load, num_warps = _decode_stream_flags(
        target,
        bitstreams.numel(),
        variable_rows=variable_rows,
        ptr_aligned=ptr_aligned,
        per_slot_2d=per_slot_2d,
    )
    q_u32_aligned = (
        round_int8 and block_size % 4 == 0 and out.is_contiguous() and out.data_ptr() % 4 == 0
    )
    return (
        use_dp4a,
        use_sad_u8,
        clamped_load,
        has_tail_bytes,
        q_u32_aligned,
        num_warps,
    )


def _launch_uniform_decode(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    layout: TrellisLayout,
    out: torch.Tensor,
    *,
    round_int8: bool,
) -> None:
    num_blocks = bitstreams.shape[0]
    if num_blocks == 0:
        return
    tile_steps = triton.next_power_of_2(layout.block_size)
    grid = (triton.cdiv(num_blocks, _BLOCKS_PER_PROGRAM),)
    needs_w2 = layout.block_bits > 5 * layout.block_size
    ptr_aligned = bitstreams.is_contiguous() and bitstreams.data_ptr() % 4 == 0
    use_dp4a, use_sad_u8, clamped_load, has_tail_bytes, q_u32_aligned, num_warps = (
        _decode_launch_flags(
            bitstreams,
            out,
            layout.block_size,
            variable_rows=False,
            ptr_aligned=ptr_aligned,
            round_int8=round_int8,
        )
    )
    with device_context(bitstreams.device):
        decode_int8_kernel[grid](
            bitstreams,
            block_scales,
            out,
            None,
            None,
            num_blocks,
            0,
            bitstreams.numel(),
            layout.block_size,
            layout.step_bits,
            layout.num_wide_steps,
            layout.bytes_per_block,
            tile_steps=tile_steps,
            tile_blocks=_BLOCKS_PER_PROGRAM,
            variable_rows=False,
            exact_pow2=layout.block_size == tile_steps,
            needs_w2=needs_w2,
            use_dp4a=use_dp4a,
            has_tail_bytes=has_tail_bytes,
            ptr_aligned=ptr_aligned,
            q_u32_aligned=q_u32_aligned,
            round_int8=round_int8,
            use_sad_u8=use_sad_u8,
            clamped_load=clamped_load,
            num_warps=num_warps,
        )


def decode_int8_out(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    layout: TrellisLayout,
    qdata: torch.Tensor,
) -> None:
    """Decode contiguous uint8 ``[S, bytes]`` streams and uint8 E4M3 scales ``[S]`` into ``qdata``.

    ``qdata`` is contiguous INT8 holding ``S · block_size`` values in block order.
    """
    _launch_uniform_decode(bitstreams, block_scales, layout, qdata, round_int8=True)


def decode_fp32_out(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    layout: TrellisLayout,
    out: torch.Tensor,
) -> None:
    """Decode contiguous uint8 ``[S, bytes]`` streams into unrounded block-scaled FP32 ``out``."""
    _launch_uniform_decode(bitstreams, block_scales, layout, out, round_int8=False)


def compute_row_offsets(row_widths: torch.Tensor) -> torch.Tensor:
    """Compute per-row byte-offset prefixes (in blocks) for variable-row trellis streams."""
    rows = row_widths.shape[0]
    row_offsets = torch.empty(rows, dtype=torch.int64, device=row_widths.device)
    if rows > 0:
        row_offsets[:1].zero_()
    if rows > 1:
        torch.cumsum(row_widths[:-1], dim=0, dtype=torch.int64, out=row_offsets[1:])
    return row_offsets


def compute_row_offsets_2d(
    row_widths: torch.Tensor,
    *,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Compute per-slot byte-offset prefixes (in blocks) for 2-D ``[slots, rows]`` widths."""
    num_slots, rows = row_widths.shape
    row_offsets = torch.empty((num_slots, rows), dtype=dtype, device=row_widths.device)
    if num_slots > 0 and rows > 0:
        row_offsets[:, :1].zero_()
    if num_slots > 0 and rows > 1:
        torch.cumsum(row_widths[:, :-1], dim=1, dtype=dtype, out=row_offsets[:, 1:])
    return row_offsets


def _launch_variable_rows_decode(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    block_size: int,
    out: torch.Tensor,
    *,
    row_offsets: torch.Tensor | None = None,
    round_int8: bool,
) -> None:
    rows, blocks_per_row = block_scales.shape
    num_blocks = rows * blocks_per_row
    if num_blocks == 0:
        return
    tile_steps = triton.next_power_of_2(block_size)
    grid = (triton.cdiv(num_blocks, _BLOCKS_PER_PROGRAM),)
    ptr_aligned = bitstreams.is_contiguous() and bitstreams.data_ptr() % 4 == 0
    use_dp4a, use_sad_u8, clamped_load, has_tail_bytes, q_u32_aligned, num_warps = (
        _decode_launch_flags(
            bitstreams,
            out,
            block_size,
            variable_rows=True,
            ptr_aligned=ptr_aligned,
            round_int8=round_int8,
        )
    )
    with device_context(bitstreams.device):
        resolved_row_offsets = (
            compute_row_offsets(row_widths) if row_offsets is None else row_offsets
        )
        decode_int8_kernel[grid](
            bitstreams,
            block_scales.view(-1),
            out,
            row_widths,
            resolved_row_offsets,
            num_blocks,
            blocks_per_row,
            bitstreams.numel(),
            block_size,
            0,
            0,
            0,
            tile_steps=tile_steps,
            tile_blocks=_BLOCKS_PER_PROGRAM,
            variable_rows=True,
            exact_pow2=block_size == tile_steps,
            needs_w2=True,
            use_dp4a=use_dp4a,
            has_tail_bytes=has_tail_bytes,
            ptr_aligned=ptr_aligned,
            q_u32_aligned=q_u32_aligned,
            round_int8=round_int8,
            use_sad_u8=use_sad_u8,
            clamped_load=clamped_load,
            num_warps=num_warps,
        )


def decode_variable_rows_int8_out(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    block_size: int,
    qdata: torch.Tensor,
    *,
    row_offsets: torch.Tensor | None = None,
) -> None:
    """Decode variable-width row streams ``[total_bytes]`` and uint8 E4M3 scales ``[R, B]``.

    ``row_widths`` is contiguous uint8 ``[R]`` holding each row's stored bytes per block;
    ``qdata`` is contiguous INT8 holding ``R · B · block_size`` values in row-major order.
    """
    _launch_variable_rows_decode(
        bitstreams,
        block_scales,
        row_widths,
        block_size,
        qdata,
        row_offsets=row_offsets,
        round_int8=True,
    )


def decode_variable_rows_fp32_out(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    block_size: int,
    out: torch.Tensor,
    *,
    row_offsets: torch.Tensor | None = None,
) -> None:
    """Decode variable-width row streams into unrounded block-scaled FP32 ``out``."""
    _launch_variable_rows_decode(
        bitstreams,
        block_scales,
        row_widths,
        block_size,
        out,
        row_offsets=row_offsets,
        round_int8=False,
    )


def decode_slots_int8_out(
    bitstreams: torch.Tensor,
    block_scales_u8: torch.Tensor,
    q_out: torch.Tensor,
    *,
    block_size: int,
    row_widths: torch.Tensor | None = None,
    slot_offsets: torch.Tensor | None = None,
    active_counts: torch.Tensor | None = None,
    row_offsets: torch.Tensor | None = None,
) -> None:
    """Decode uniform or variable-row ``[chunk_slots, rows, ...]`` trellis weights to INT8."""
    chunk_slots, rows_per_slot, blocks_per_row = block_scales_u8.shape
    num_blocks = chunk_slots * rows_per_slot * blocks_per_row
    if num_blocks == 0:
        return
    tile_steps = triton.next_power_of_2(block_size)
    grid = (triton.cdiv(num_blocks, _BLOCKS_PER_PROGRAM),)
    filter_active = active_counts is not None
    variable_rows = row_widths is not None
    has_slot_offsets = variable_rows and slot_offsets is not None
    per_slot_2d = variable_rows and not has_slot_offsets
    if variable_rows:
        assert row_widths is not None
        resolved_block_size = block_size
        bitstreams_bytes = bitstreams.numel()
        step_bits = num_wide_steps = bytes_per_block = 0
        slot_bytes = bitstreams.shape[1] if per_slot_2d else 0
        needs_w2 = True
        row_widths_arg = row_widths.contiguous().view(-1)
    else:
        layout = TrellisLayout.from_storage(block_size, bitstreams.shape[3])
        resolved_block_size = layout.block_size
        bitstreams_bytes = bitstreams.numel()
        step_bits = layout.step_bits
        num_wide_steps = layout.num_wide_steps
        bytes_per_block = layout.bytes_per_block
        slot_bytes = 0
        needs_w2 = layout.block_bits > 5 * layout.block_size
        row_widths_arg = None

    ptr_aligned = bitstreams.is_contiguous() and bitstreams.data_ptr() % 4 == 0
    use_dp4a, use_sad_u8, clamped_load, has_tail_bytes, q_u32_aligned, num_warps = (
        _decode_launch_flags(
            bitstreams,
            q_out,
            resolved_block_size,
            variable_rows=variable_rows,
            ptr_aligned=ptr_aligned,
            per_slot_2d=per_slot_2d,
        )
    )

    with device_context(bitstreams.device):
        row_offsets_arg = (
            None
            if row_widths is None
            else (
                compute_row_offsets_2d(row_widths, dtype=torch.int32)
                if row_offsets is None
                else row_offsets
            )
            .contiguous()
            .view(-1)
        )
        decode_slots_int8_kernel[grid](
            bitstreams,
            block_scales_u8.contiguous().view(-1),
            q_out,
            row_widths_arg,
            row_offsets_arg,
            active_counts,
            slot_offsets if has_slot_offsets else None,
            num_blocks,
            blocks_per_row,
            bitstreams_bytes,
            resolved_block_size,
            step_bits,
            num_wide_steps,
            bytes_per_block,
            rows_per_slot,
            slot_bytes,
            tile_steps=tile_steps,
            tile_blocks=_BLOCKS_PER_PROGRAM,
            variable_rows=variable_rows,
            exact_pow2=resolved_block_size == tile_steps,
            filter_active=filter_active,
            has_slot_offsets=has_slot_offsets,
            per_slot_2d=per_slot_2d,
            needs_w2=needs_w2,
            use_dp4a=use_dp4a,
            has_tail_bytes=has_tail_bytes,
            ptr_aligned=ptr_aligned,
            q_u32_aligned=q_u32_aligned,
            use_sad_u8=use_sad_u8,
            clamped_load=clamped_load,
            num_warps=num_warps,
        )


__all__ = [
    "compute_row_offsets",
    "compute_row_offsets_2d",
    "decode_fp32_out",
    "decode_int8_kernel",
    "decode_int8_out",
    "decode_slots_int8_kernel",
    "decode_slots_int8_out",
    "decode_variable_rows_fp32_out",
    "decode_variable_rows_int8_out",
]
