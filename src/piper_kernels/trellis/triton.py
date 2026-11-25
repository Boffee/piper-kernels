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
def _decode_states_to_q_fp32(states, scale_codes):
    """Decode 16-bit trellis states and raw E4M3 scale codes to quantized INT8 values in FP32."""
    x = states * _LCG_MULTIPLIER + _LCG_INCREMENT
    numerators = ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)).to(tl.int32)
    numerators -= _CODE_OFFSET
    multipliers = _decode_e4m3(scale_codes.to(tl.uint32)) * _CODE_RECIPROCAL
    scaled = numerators.to(tl.float32) * multipliers[:, None]
    return libdevice.rint(tl.clamp(scaled, -127.0, 127.0))


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
):
    blocks = tl.program_id(0) * tile_blocks + tl.arange(0, tile_blocks)
    steps = tl.arange(0, tile_steps)
    clamped_steps = steps if exact_pow2 else tl.minimum(steps, block_size - 1)
    block_mask = blocks < num_blocks
    mask = (
        block_mask[:, None] if exact_pow2 else (block_mask[:, None] & (steps < block_size)[None, :])
    )
    denom = tile_steps if exact_pow2 else block_size

    if variable_rows:
        row = blocks // blocks_per_row
        block_in_row = blocks - row * blocks_per_row
        width = tl.load(row_widths + row, mask=block_mask, other=0).to(tl.int32)
        row_offset = tl.load(row_offsets + row, mask=block_mask, other=0).to(tl.int64)
        base = row_offset * blocks_per_row + block_in_row.to(tl.int64) * width.to(tl.int64)
        row_bits = width * 8
        row_step_bits = row_bits // denom
        row_wide_steps = row_bits - row_step_bits * denom
        start_bits = (
            row_step_bits[:, None] * clamped_steps[None, :]
            + row_wide_steps[:, None] * (clamped_steps[None, :] + 1) // denom
        )
        stream_bits = row_bits[:, None]
        block_ptr_1d = bitstreams + base
        # Bound loads to bitstreams even if caller-owned row_widths are malformed.
        remaining_bytes = tl.maximum(bitstreams_bytes - base, 0)
        stream_bytes = tl.minimum(width.to(tl.int64), remaining_bytes).to(tl.int32)
        stream_bytes_2d = stream_bytes[:, None]
    else:
        offsets = step_bits * clamped_steps + (clamped_steps + 1) * num_wide_steps // denom
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
            tile_steps=tile_steps,
            tile_blocks=_BLOCKS_PER_PROGRAM,
            variable_rows=False,
            exact_pow2=layout.block_size == tile_steps,
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
            num_warps=_NUM_WARPS,
        )


__all__ = [
    "compute_row_offsets",
    "decode_int8_kernel",
    "decode_int8_out",
    "decode_variable_rows_int8_out",
]
