"""Triton decode of trellis-coded blocks into INT8 values.

Every weight decodes independently: its state is a 16-bit window of the stream at an
offset the layout gives in closed form, so a tile of blocks decodes without a
sequential walk. See the package README for the format and measured throughput.
"""

# Triton's JIT launcher and constexpr arguments are not fully represented by its
# Python typing surface.
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.runtime import device_context

from ._layout import (
    CODE_OFFSET,
    CODE_RECIPROCAL,
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    STATE_BITS,
    TrellisLayout,
)

# Untuned launch shape: decode is off the critical path (see the README).
_BLOCKS_PER_PROGRAM = 8
_NUM_WARPS = 4
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
def _read_bits(block_bytes, bytes_per_block, valid, start_bit, num_bits):
    """``num_bits`` (at most 16) stream bits from ``start_bit``, most significant first."""
    byte = start_bit // 8
    window = tl.zeros(start_bit.shape, tl.uint32)
    # Triton's typing does not declare static_range iterable.
    for index in tl.static_range(3):  # pyright: ignore[reportGeneralTypeIssues]
        in_stream = valid & (byte + index < bytes_per_block)
        value = tl.load(block_bytes + byte + index, mask=in_stream, other=0)
        window = (window << 8) | value.to(tl.uint32)
    shift = (24 - start_bit % 8 - num_bits).to(tl.uint32)
    ones = (tl.full(start_bit.shape, 1, tl.uint32) << num_bits.to(tl.uint32)) - 1
    return (window >> shift) & ones


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
):
    blocks = tl.program_id(0) * tile_blocks + tl.arange(0, tile_blocks)
    steps = tl.arange(0, tile_steps)
    clamped_steps = tl.minimum(steps, block_size - 1)
    block_mask = blocks < num_blocks
    step_mask = steps < block_size
    mask = block_mask[:, None] & step_mask[None, :]

    if variable_rows:
        row = blocks // blocks_per_row
        block_in_row = blocks % blocks_per_row
        width = tl.load(row_widths + row, mask=block_mask, other=0).to(tl.int32)
        row_offset = tl.load(row_offsets + row, mask=block_mask, other=0).to(tl.int64)
        base = row_offset * blocks_per_row + block_in_row.to(tl.int64) * width.to(tl.int64)
        row_bits = width * 8
        row_step_bits = row_bits // block_size
        row_wide_steps = row_bits % block_size
        start_bits = (
            row_step_bits[:, None] * clamped_steps[None, :]
            + row_wide_steps[:, None] * (clamped_steps[None, :] + 1) // block_size
        )
        head_bits = tl.minimum(row_bits[:, None] - start_bits, _STATE_BITS)
        tail_bits = _STATE_BITS - head_bits
        block_bytes = bitstreams + base[:, None]
        # Bound loads to bitstreams even if caller-owned row_widths are malformed.
        remaining_bytes = tl.maximum(bitstreams_bytes - base, 0)
        stream_bytes = tl.minimum(width.to(tl.int64), remaining_bytes).to(tl.int32)[:, None]
    else:
        offsets = step_bits * clamped_steps + (clamped_steps + 1) * num_wide_steps // block_size
        head_bits = tl.minimum(block_bits - offsets, _STATE_BITS)
        tail_bits = _STATE_BITS - head_bits
        block_bytes = bitstreams + blocks.to(tl.int64)[:, None] * bytes_per_block
        start_bits = tl.broadcast_to(offsets[None, :], (tile_blocks, tile_steps))
        head_bits = tl.broadcast_to(head_bits[None, :], (tile_blocks, tile_steps))
        tail_bits = tl.broadcast_to(tail_bits[None, :], (tile_blocks, tile_steps))
        stream_bytes = bytes_per_block

    stream_starts = tl.zeros_like(start_bits)
    head = _read_bits(block_bytes, stream_bytes, mask, start_bits, head_bits)
    tail = _read_bits(block_bytes, stream_bytes, mask, stream_starts, tail_bits)
    states = (head << tail_bits.to(tl.uint32)) | tail

    x = states * _LCG_MULTIPLIER + _LCG_INCREMENT
    numerators = ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)).to(tl.int32)
    numerators -= _CODE_OFFSET

    scale_codes = tl.load(block_scales + blocks.to(tl.int64), mask=block_mask, other=0)
    multipliers = _decode_e4m3(scale_codes.to(tl.uint32)) * _CODE_RECIPROCAL
    scaled = numerators.to(tl.float32) * multipliers[:, None]
    values = libdevice.rint(tl.clamp(scaled, -127.0, 127.0)).to(tl.int8)
    destination = blocks.to(tl.int64)[:, None] * block_size + steps[None, :]
    tl.store(qdata + destination, values, mask=mask)


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
    grid = (triton.cdiv(num_blocks, _BLOCKS_PER_PROGRAM),)
    with device_context(bitstreams.device):
        decode_int8_kernel[grid](
            bitstreams,
            block_scales,
            qdata,
            None,
            None,
            num_blocks,
            0,
            0,
            layout.block_size,
            layout.block_bits,
            layout.step_bits,
            layout.num_wide_steps,
            layout.bytes_per_block,
            tile_steps=triton.next_power_of_2(layout.block_size),
            tile_blocks=_BLOCKS_PER_PROGRAM,
            variable_rows=False,
            num_warps=_NUM_WARPS,
        )


def decode_variable_rows_int8_out(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    block_size: int,
    qdata: torch.Tensor,
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
    grid = (triton.cdiv(num_blocks, _BLOCKS_PER_PROGRAM),)
    with device_context(bitstreams.device):
        row_offsets = torch.empty(rows, dtype=torch.int64, device=bitstreams.device)
        row_offsets[:1].zero_()
        if rows > 1:
            torch.cumsum(row_widths[:-1], dim=0, dtype=torch.int64, out=row_offsets[1:])
        decode_int8_kernel[grid](
            bitstreams,
            block_scales.view(-1),
            qdata,
            row_widths,
            row_offsets,
            num_blocks,
            blocks_per_row,
            bitstreams.numel(),
            block_size,
            0,
            0,
            0,
            0,
            tile_steps=triton.next_power_of_2(block_size),
            tile_blocks=_BLOCKS_PER_PROGRAM,
            variable_rows=True,
            num_warps=_NUM_WARPS,
        )


__all__ = ["decode_int8_kernel", "decode_int8_out", "decode_variable_rows_int8_out"]
