"""Triton decode of trellis-coded blocks into INT8 values.

Every weight decodes independently: its state is a 16-bit window of the stream at an
offset the layout gives in closed form, so a tile of blocks decodes without a
sequential walk. See the package README for the format and measured throughput.
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
from piper_kernels._triton.targets import AcceleratorTarget

from ._layout import (
    CODE_OFFSET,
    CODE_RECIPROCAL,
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    STATE_BITS,
    TrellisLayout,
)

# Tuned on RTX 5090 (SM120): 4 blocks/program x 2 warps minimizes wrap-load and barrier overhead.
_BLOCKS_PER_PROGRAM = 4
_NUM_WARPS = 2
_STATE_BITS = tl.constexpr(STATE_BITS)
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
def _decode_states_lcg_numerators(states):
    """Compute signed LCG byte-sum numerators from 16-bit trellis states."""
    x = states * _LCG_MULTIPLIER + _LCG_INCREMENT
    numerators = ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)).to(tl.int32)
    return numerators - _CODE_OFFSET


@triton.jit
def _decode_states_to_q_fp32(states, scale_codes):
    """Decode 16-bit trellis states and raw E4M3 scale codes to quantized INT8 values in FP32."""
    numerators = _decode_states_lcg_numerators(states)
    multipliers = _decode_e4m3(scale_codes.to(tl.uint32)) * _CODE_RECIPROCAL
    scaled = numerators.to(tl.float32) * multipliers[:, None]
    return libdevice.rint(tl.clamp(scaled, -127.0, 127.0))


@triton.jit
def _uniform_row_start_bits(step_bits, num_wide_steps, clamped_steps, denom):
    """Compute uniform per-step start bit offsets for one block."""
    return step_bits * clamped_steps + (clamped_steps + 1) * num_wide_steps // denom


@triton.jit
def _variable_row_start_bits(width, clamped_steps, denom):
    """Compute per-row variable-width start bit offsets and total stream bits."""
    row_bits = width * 8
    row_step_bits = row_bits // denom
    row_wide_steps = row_bits - row_step_bits * denom
    start_bits = (
        row_step_bits[:, None] * clamped_steps[None, :]
        + row_wide_steps[:, None] * (clamped_steps[None, :] + 1) // denom
    )
    return start_bits, row_bits[:, None]


@triton.jit
def _bit_window_params(start_bits, stream_bits):
    """Compute closed-form byte indices and cyclic 16-bit head/tail masks."""
    head_bits = tl.minimum(stream_bits - start_bits, _STATE_BITS)
    tail_bits = _STATE_BITS - head_bits
    byte = start_bits >> 3
    bit_in_byte = start_bits & 7
    shift = (24 - bit_in_byte - head_bits).to(tl.uint32)
    head_mask = (tl.full(start_bits.shape, 1, tl.uint32) << head_bits.to(tl.uint32)) - 1
    tail_shift = (16 - tail_bits).to(tl.uint32)
    has_tail = tail_bits > 0
    tail_bits_u32 = tail_bits.to(tl.uint32)
    return byte, shift, head_mask, has_tail, tail_bits_u32, tail_shift


@triton.jit
def _load_cyclic_states(
    block_ptr_1d,
    block_bytes,
    byte,
    shift,
    head_mask,
    has_tail,
    tail_bits_u32,
    tail_shift,
    b0_mask,
    b1_mask,
    m0,
    m1,
    m2,
):
    """Load three stream bytes + two wrap bytes and assemble 16-bit cyclic trellis states."""
    b0 = tl.load(block_ptr_1d, mask=b0_mask, other=0).to(tl.uint32)
    b1 = tl.load(block_ptr_1d + 1, mask=b1_mask, other=0).to(tl.uint32)
    wrap_u16 = (b0 << 8) | b1
    w0 = tl.load(block_bytes + byte, mask=m0, other=0).to(tl.uint32)
    w1 = tl.load(block_bytes + byte + 1, mask=m1, other=0).to(tl.uint32)
    w2 = tl.load(block_bytes + byte + 2, mask=m2, other=0).to(tl.uint32)
    window = (w0 << 16) | (w1 << 8) | w2
    head = (window >> shift) & head_mask
    tail = tl.where(has_tail, wrap_u16[:, None] >> tail_shift, 0)
    return (head << tail_bits_u32) | tail


@triton.jit
def _decode_variable_addresses(
    bitstreams,
    row_widths,
    row_offsets,
    slot_offsets_ptr,
    blocks,
    blocks_per_row,
    bitstreams_bytes,
    denom,
    clamped_steps,
    block_mask,
    rows_per_slot,
    slot_bytes,
    has_slot_offsets: tl.constexpr,
    per_slot_2d: tl.constexpr,
):
    row = blocks // blocks_per_row
    block_in_row = blocks - row * blocks_per_row
    width = tl.load(row_widths + row, mask=block_mask, other=0).to(tl.int32)
    row_offset = tl.load(row_offsets + row, mask=block_mask, other=0).to(tl.int64)
    within_slot = row_offset * blocks_per_row + block_in_row.to(tl.int64) * width.to(tl.int64)
    if per_slot_2d:
        local_slot = row // rows_per_slot
        slot_base = local_slot.to(tl.int64) * tl.full((), slot_bytes, tl.int64)
        base = slot_base + within_slot
        remaining_bytes = tl.maximum(tl.full((), slot_bytes, tl.int64) - within_slot, 0)
    elif has_slot_offsets:
        local_slot = row // rows_per_slot
        raw_slot_base = tl.load(slot_offsets_ptr + local_slot, mask=block_mask, other=0).to(
            tl.int64
        )
        valid_base = raw_slot_base >= 0
        slot_base = tl.where(valid_base, raw_slot_base, 0)
        base = slot_base + within_slot
        remaining_bytes = tl.where(
            valid_base, tl.maximum(bitstreams_bytes.to(tl.int64) - base, 0), 0
        )
    else:
        base = within_slot
        # Bound loads to bitstreams even if caller-owned row_widths are malformed.
        remaining_bytes = tl.maximum(bitstreams_bytes.to(tl.int64) - base, 0)
    start_bits, stream_bits = _variable_row_start_bits(width, clamped_steps, denom)
    block_ptr_1d = bitstreams + base
    stream_bytes = tl.minimum(width.to(tl.int64), remaining_bytes).to(tl.int32)
    return block_ptr_1d, start_bits, stream_bits, stream_bytes


@triton.jit
def _decode_int8_blocks(
    bitstreams,
    block_scales,
    qdata,
    row_widths,
    row_offsets,
    num_blocks,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    block_bits,
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
):
    blocks = tl.program_id(0) * tile_blocks + tl.arange(0, tile_blocks)
    steps = tl.arange(0, tile_steps)
    clamped_steps = steps if exact_pow2 else tl.minimum(steps, block_size - 1)
    block_mask = blocks < num_blocks
    if filter_active:
        row_all = blocks // blocks_per_row
        local_slot_all = row_all // rows_per_slot
        slot_active = tl.load(active_counts_ptr + local_slot_all, mask=block_mask, other=0) > 0
        if tl.sum(slot_active.to(tl.int32)) == 0:
            return
        block_mask = block_mask & slot_active
    mask = (
        block_mask[:, None] if exact_pow2 else (block_mask[:, None] & (steps < block_size)[None, :])
    )
    denom = tile_steps if exact_pow2 else block_size

    if variable_rows:
        block_ptr_1d, start_bits, stream_bits, stream_bytes = _decode_variable_addresses(
            bitstreams,
            row_widths,
            row_offsets,
            slot_offsets_ptr,
            blocks,
            blocks_per_row,
            bitstreams_bytes,
            denom,
            clamped_steps,
            block_mask,
            rows_per_slot,
            slot_bytes,
            has_slot_offsets=has_slot_offsets,
            per_slot_2d=per_slot_2d,
        )
        stream_bytes_2d = stream_bytes[:, None]
    else:
        offsets = _uniform_row_start_bits(step_bits, num_wide_steps, clamped_steps, denom)
        block_ptr_1d = bitstreams + blocks.to(tl.int64) * bytes_per_block
        start_bits = tl.broadcast_to(offsets[None, :], (tile_blocks, tile_steps))
        stream_bits = block_bits
        stream_bytes = tl.full((tile_blocks,), bytes_per_block, tl.int32)
        stream_bytes_2d = bytes_per_block

    byte, shift, head_mask, has_tail, tail_bits_u32, tail_shift = _bit_window_params(
        start_bits, stream_bits
    )
    states = _load_cyclic_states(
        block_ptr_1d,
        block_ptr_1d[:, None],
        byte,
        shift,
        head_mask,
        has_tail,
        tail_bits_u32,
        tail_shift,
        block_mask & (stream_bytes > 0),
        block_mask & (stream_bytes > 1),
        mask & (byte < stream_bytes_2d),
        mask & ((byte + 1) < stream_bytes_2d),
        mask & ((byte + 2) < stream_bytes_2d),
    )

    scale_codes = tl.load(block_scales + blocks.to(tl.int64), mask=block_mask, other=0)
    values = _decode_states_to_q_fp32(states, scale_codes).to(tl.int8)
    destination = blocks.to(tl.int64)[:, None] * denom + steps[None, :]
    tl.store(qdata + destination, values, mask=mask)


@triton.jit
def _extract_16_from_3be_4step(be0, be1, be2, rel_bit):
    """Extract a 16-bit big-endian trellis state across up to three BE uint32 words."""
    hi = tl.where(rel_bit < 32, be0, tl.where(rel_bit < 64, be1, be2))
    lo = tl.where(rel_bit < 32, be1, tl.where(rel_bit < 64, be2, 0))
    sh = (rel_bit & 31).to(tl.uint32)
    top32 = tl.inline_asm_elementwise(
        asm="shf.l.clamp.b32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[lo, hi, sh],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return top32 >> 16


@triton.jit
def _load_3be_and_head_u32(
    u32_ptr,
    total_full_words,
    base_byte,
    first_byte,
    group_valid,
    head_valid,
):
    """Load three byte-swapped uint32 words per 4-step group plus the block wrap head uint16."""
    blk_word_base = base_byte >> 2
    blk_align = (base_byte & 3).to(tl.int32)
    blk_align_bits = blk_align << 3
    first_word = (blk_align + first_byte) >> 2
    w0_idx = blk_word_base + first_word
    word_base_bit = first_word << 5

    w0 = tl.load(u32_ptr + w0_idx, mask=group_valid & (w0_idx < total_full_words), other=0)
    w1 = tl.load(
        u32_ptr + w0_idx + 1,
        mask=group_valid & ((w0_idx + 1) < total_full_words),
        other=0,
    )
    w2 = tl.load(
        u32_ptr + w0_idx + 2,
        mask=group_valid & ((w0_idx + 2) < total_full_words),
        other=0,
    )
    be0 = tl.inline_asm_elementwise(
        asm="prmt.b32 $0, $1, 0, 0x0123;",
        constraints="=r,r",
        args=[w0],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    be1 = tl.inline_asm_elementwise(
        asm="prmt.b32 $0, $1, 0, 0x0123;",
        constraints="=r,r",
        args=[w1],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    be2 = tl.inline_asm_elementwise(
        asm="prmt.b32 $0, $1, 0, 0x0123;",
        constraints="=r,r",
        args=[w2],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    h0 = tl.load(
        u32_ptr + blk_word_base,
        mask=head_valid & (blk_word_base < total_full_words),
        other=0,
    )
    h1 = tl.load(
        u32_ptr + blk_word_base + 1,
        mask=head_valid & ((blk_word_base + 1) < total_full_words),
        other=0,
    )
    head_raw = tl.inline_asm_elementwise(
        asm="shf.r.clamp.b32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[h0, h1, blk_align_bits.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    head_u16 = tl.inline_asm_elementwise(
        asm="prmt.b32 $0, $1, 0, 0x4401;",
        constraints="=r,r",
        args=[head_raw],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return be0, be1, be2, head_u16, blk_align_bits, word_base_bit


@triton.jit
def _decode_step_3be_sm120(
    be0,
    be1,
    be2,
    head_u16,
    blk_align_bits,
    word_base_bit,
    step_j,
    step_bits,
    num_wide_steps,
    denom,
    block_bits,
    mult,
    exact_pow2: tl.constexpr,
    block_size,
):
    """Extract and decode one trellis step via 3-word BE shifts and SM120 dp4a."""
    clamped = step_j if exact_pow2 else tl.minimum(step_j, block_size - 1)
    sbit = step_bits * clamped + (clamped + 1) * num_wide_steps // denom
    rel_bit = blk_align_bits + sbit - word_base_bit
    st = _extract_16_from_3be_4step(be0, be1, be2, rel_bit)
    rem = block_bits - sbit
    wrap_mask = rem < _STATE_BITS
    rem_u = tl.minimum(rem, _STATE_BITS).to(tl.uint32)
    tail_bits = _STATE_BITS - rem_u
    keep_mask = (0xFFFF << tail_bits) & 0xFFFF
    st = tl.where(wrap_mask, (st & keep_mask) | (head_u16 >> rem_u), st)
    x = st * _LCG_MULTIPLIER + _LCG_INCREMENT
    # CODE_OFFSET (= 510) is folded directly into the dp4a accumulator seed.
    num = tl.inline_asm_elementwise(
        asm="dp4a.s32.u32 $0, 0x01010101, $1, -510;",
        constraints="=r,r",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q = libdevice.rint(tl.clamp(num.to(tl.float32) * mult, -127.0, 127.0))
    return q if exact_pow2 else tl.where(step_j < block_size, q, 0.0)


@triton.jit
def _decode_int8_blocks_sm120_3word(
    bitstreams,
    block_scales,
    qdata,
    row_widths,
    row_offsets,
    num_blocks,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    block_bits,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    tile_steps: tl.constexpr,
    tile_blocks: tl.constexpr,
    variable_rows: tl.constexpr,
    exact_pow2: tl.constexpr,
):
    """SM120 3-word uint32 bitstream decode with packed 4x INT8 uint32 stores."""
    blocks = tl.program_id(0) * tile_blocks + tl.arange(0, tile_blocks)
    block_mask = blocks < num_blocks
    denom = tile_steps if exact_pow2 else block_size
    num_groups: tl.constexpr = tile_steps // 4
    g = tl.arange(0, num_groups)
    first_step = g * 4
    clamped_first = first_step if exact_pow2 else tl.minimum(first_step, block_size - 1)

    if variable_rows:
        row = blocks // blocks_per_row
        block_in_row = blocks - row * blocks_per_row
        width = tl.load(row_widths + row, mask=block_mask, other=0).to(tl.int32)
        row_offset = tl.load(row_offsets + row, mask=block_mask, other=0).to(tl.int64)
        base_bytes = row_offset * blocks_per_row + block_in_row.to(tl.int64) * width.to(tl.int64)
        eff_block_bits = (width * 8)[:, None]
        eff_step_bits = eff_block_bits // denom
        eff_wide_steps = eff_block_bits - eff_step_bits * denom
        first_bit = (
            eff_step_bits * clamped_first[None, :]
            + eff_wide_steps * (clamped_first[None, :] + 1) // denom
        )
        remaining_bytes = tl.maximum(bitstreams_bytes.to(tl.int64) - base_bytes, 0)
        stream_bytes = tl.minimum(width.to(tl.int64), remaining_bytes).to(tl.int32)
    else:
        base_bytes = blocks.to(tl.int64) * bytes_per_block
        eff_block_bits = block_bits
        eff_step_bits = step_bits
        eff_wide_steps = num_wide_steps
        first_bit = (step_bits * clamped_first + (clamped_first + 1) * num_wide_steps // denom)[
            None, :
        ]
        stream_bytes = tl.full((tile_blocks,), bytes_per_block, tl.int32)
    first_byte = first_bit >> 3

    u32_ptr = bitstreams.to(tl.pointer_type(tl.uint32))
    total_full_words = bitstreams_bytes.to(tl.int64) >> 2
    group_valid = (
        block_mask[:, None]
        if exact_pow2
        else (block_mask[:, None] & (first_step < block_size)[None, :])
    )
    non_empty = block_mask & (stream_bytes > 0)
    be0, be1, be2, head_u16, blk_align_bits, word_base_bit = _load_3be_and_head_u32(
        u32_ptr,
        total_full_words,
        base_bytes[:, None],
        first_byte,
        group_valid & (stream_bytes > 0)[:, None],
        non_empty[:, None],
    )
    head_clamp_shift = 16 - tl.minimum(tl.maximum(stream_bytes, 0) * 8, 16).to(tl.uint32)
    head_clamp = (0xFFFF << head_clamp_shift) & 0xFFFF
    head_u16 = head_u16 & head_clamp[:, None]

    sc = tl.load(block_scales + blocks.to(tl.int64), mask=block_mask, other=0)
    mult = (sc.to(tl.float8e4nv, bitcast=True).to(tl.float32) * _CODE_RECIPROCAL)[:, None]

    q0 = (
        _decode_step_3be_sm120(
            be0,
            be1,
            be2,
            head_u16,
            blk_align_bits,
            word_base_bit,
            first_step[None, :],
            eff_step_bits,
            eff_wide_steps,
            denom,
            eff_block_bits,
            mult,
            exact_pow2,
            block_size,
        )
        .to(tl.int8)
        .to(tl.uint8)
        .to(tl.uint32)
    )
    q1 = (
        _decode_step_3be_sm120(
            be0,
            be1,
            be2,
            head_u16,
            blk_align_bits,
            word_base_bit,
            (first_step + 1)[None, :],
            eff_step_bits,
            eff_wide_steps,
            denom,
            eff_block_bits,
            mult,
            exact_pow2,
            block_size,
        )
        .to(tl.int8)
        .to(tl.uint8)
        .to(tl.uint32)
    )
    q2 = (
        _decode_step_3be_sm120(
            be0,
            be1,
            be2,
            head_u16,
            blk_align_bits,
            word_base_bit,
            (first_step + 2)[None, :],
            eff_step_bits,
            eff_wide_steps,
            denom,
            eff_block_bits,
            mult,
            exact_pow2,
            block_size,
        )
        .to(tl.int8)
        .to(tl.uint8)
        .to(tl.uint32)
    )
    q3 = (
        _decode_step_3be_sm120(
            be0,
            be1,
            be2,
            head_u16,
            blk_align_bits,
            word_base_bit,
            (first_step + 3)[None, :],
            eff_step_bits,
            eff_wide_steps,
            denom,
            eff_block_bits,
            mult,
            exact_pow2,
            block_size,
        )
        .to(tl.int8)
        .to(tl.uint8)
        .to(tl.uint32)
    )
    packed = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)

    q_u32_ptr = qdata.to(tl.pointer_type(tl.uint32))
    dst_u32_base = blocks.to(tl.int64)[:, None] * (denom >> 2)
    tl.store(q_u32_ptr + dst_u32_base + g[None, :], packed, mask=group_valid)


@triton.jit(
    do_not_specialize=[
        "num_blocks",
        "blocks_per_row",
        "bitstreams_bytes",
        "block_size",
        "block_bits",
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
    block_bits,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    tile_steps: tl.constexpr,
    tile_blocks: tl.constexpr,
    variable_rows: tl.constexpr,
    exact_pow2: tl.constexpr,
    use_sm120_3word: tl.constexpr = False,
):
    if use_sm120_3word:
        _decode_int8_blocks_sm120_3word(
            bitstreams,
            block_scales,
            qdata,
            row_widths,
            row_offsets,
            num_blocks,
            blocks_per_row,
            bitstreams_bytes,
            block_size,
            block_bits,
            step_bits,
            num_wide_steps,
            bytes_per_block,
            tile_steps=tile_steps,
            tile_blocks=tile_blocks,
            variable_rows=variable_rows,
            exact_pow2=exact_pow2,
        )
    else:
        _decode_int8_blocks(
            bitstreams,
            block_scales,
            qdata,
            row_widths,
            row_offsets,
            num_blocks,
            blocks_per_row,
            bitstreams_bytes,
            block_size,
            block_bits,
            step_bits,
            num_wide_steps,
            bytes_per_block,
            None,
            None,
            0,
            0,
            tile_steps=tile_steps,
            tile_blocks=tile_blocks,
            variable_rows=variable_rows,
            exact_pow2=exact_pow2,
            filter_active=False,
            has_slot_offsets=False,
            per_slot_2d=False,
        )


@triton.jit(
    do_not_specialize=[
        "num_blocks",
        "blocks_per_row",
        "bitstreams_bytes",
        "block_size",
        "block_bits",
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
    block_bits,
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
):
    _decode_int8_blocks(
        bitstreams,
        block_scales,
        qdata,
        row_widths,
        row_offsets,
        num_blocks,
        blocks_per_row,
        bitstreams_bytes,
        block_size,
        block_bits,
        step_bits,
        num_wide_steps,
        bytes_per_block,
        active_counts_ptr,
        slot_offsets_ptr,
        rows_per_slot,
        slot_bytes,
        tile_steps=tile_steps,
        tile_blocks=tile_blocks,
        variable_rows=variable_rows,
        exact_pow2=exact_pow2,
        filter_active=filter_active,
        has_slot_offsets=has_slot_offsets,
        per_slot_2d=per_slot_2d,
    )


def _use_sm120_3word_decode(
    bitstreams: torch.Tensor,
    qdata: torch.Tensor,
    block_size: int,
    tile_steps: int,
) -> bool:
    return (
        AcceleratorTarget.from_device(bitstreams.device).is_cuda_capability(12)
        and tile_steps >= 64
        and block_size % 4 == 0
        and bitstreams.is_contiguous()
        and qdata.is_contiguous()
        and bitstreams.data_ptr() % 4 == 0
        and bitstreams.numel() % 4 == 0
        and qdata.data_ptr() % 4 == 0
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
    num_blocks = bitstreams.shape[0]
    if num_blocks == 0:
        return
    tile_steps = triton.next_power_of_2(layout.block_size)
    grid = (triton.cdiv(num_blocks, _BLOCKS_PER_PROGRAM),)
    use_sm120_3word = _use_sm120_3word_decode(bitstreams, qdata, layout.block_size, tile_steps)
    with device_context(bitstreams.device):
        decode_int8_kernel[grid](
            bitstreams,
            block_scales,
            qdata,
            None,
            None,
            num_blocks,
            0,
            bitstreams.numel(),
            layout.block_size,
            layout.block_bits,
            layout.step_bits,
            layout.num_wide_steps,
            layout.bytes_per_block,
            tile_steps=tile_steps,
            tile_blocks=_BLOCKS_PER_PROGRAM,
            variable_rows=False,
            exact_pow2=layout.block_size == tile_steps,
            use_sm120_3word=use_sm120_3word,
            num_warps=_NUM_WARPS,
        )


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

    ``row_widths`` is contiguous uint8 ``[R]`` holding each row's bytes per block;
    ``qdata`` is contiguous INT8 holding ``R · B · block_size`` values in row-major
    order.
    """
    rows, blocks_per_row = block_scales.shape
    num_blocks = rows * blocks_per_row
    if num_blocks == 0:
        return
    tile_steps = triton.next_power_of_2(block_size)
    grid = (triton.cdiv(num_blocks, _BLOCKS_PER_PROGRAM),)
    use_sm120_3word = _use_sm120_3word_decode(bitstreams, qdata, block_size, tile_steps)
    with device_context(bitstreams.device):
        resolved_row_offsets = (
            compute_row_offsets(row_widths) if row_offsets is None else row_offsets
        )
        decode_int8_kernel[grid](
            bitstreams,
            block_scales.view(-1),
            qdata,
            row_widths,
            resolved_row_offsets,
            num_blocks,
            blocks_per_row,
            bitstreams.numel(),
            block_size,
            0,
            0,
            0,
            0,
            tile_steps=tile_steps,
            tile_blocks=_BLOCKS_PER_PROGRAM,
            variable_rows=True,
            exact_pow2=block_size == tile_steps,
            use_sm120_3word=use_sm120_3word,
            num_warps=_NUM_WARPS,
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
        block_bits = step_bits = num_wide_steps = bytes_per_block = 0
        slot_bytes = bitstreams.shape[1] if per_slot_2d else 0
        row_widths_arg = row_widths.contiguous().view(-1)
    else:
        layout = TrellisLayout.from_storage(block_size, bitstreams.shape[3])
        resolved_block_size = layout.block_size
        bitstreams_bytes = 0
        block_bits = layout.block_bits
        step_bits = layout.step_bits
        num_wide_steps = layout.num_wide_steps
        bytes_per_block = layout.bytes_per_block
        slot_bytes = 0
        row_widths_arg = None

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
            block_bits,
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
            num_warps=_NUM_WARPS,
        )


__all__ = [
    "compute_row_offsets",
    "compute_row_offsets_2d",
    "decode_int8_kernel",
    "decode_int8_out",
    "decode_slots_int8_kernel",
    "decode_slots_int8_out",
    "decode_variable_rows_int8_out",
]
