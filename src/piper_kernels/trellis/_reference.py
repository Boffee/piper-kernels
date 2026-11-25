"""Portable PyTorch decode of trellis-coded blocks, the Triton kernel's oracle.

It materializes one ``[blocks, T, 16]`` bit window, so it suits tests and small
tensors rather than model loading.
"""

from __future__ import annotations

import torch

from ._layout import (
    CODE_OFFSET,
    CODE_RECIPROCAL,
    LCG_INCREMENT,
    LCG_MULTIPLIER,
    STATE_BITS,
    TrellisLayout,
    validate_block_size,
)

_INT8_LIMIT = 127


def compute_states(bitstreams: torch.Tensor, layout: TrellisLayout) -> torch.Tensor:
    """Each step's 16-bit state, int64 ``[..., T]``, from uint8 ``[..., bytes_per_block]``.

    State ``t`` is the cyclic window of bits ``offset_t .. offset_t + 15`` (most
    significant first) of the ``block_bits``-bit stream.
    """
    device = bitstreams.device
    shifts = torch.arange(7, -1, -1, device=device)
    bits = ((bitstreams.long().unsqueeze(-1) >> shifts) & 1).flatten(-2)
    steps = torch.arange(layout.block_size, device=device, dtype=torch.int64)
    offsets = layout.step_bits * steps + ((steps + 1) * layout.num_wide_steps) // layout.block_size
    window = torch.arange(STATE_BITS, device=device)
    positions = (offsets[:, None] + window) % layout.block_bits
    weights = 1 << torch.arange(STATE_BITS - 1, -1, -1, device=device)
    return (bits[..., positions] * weights).sum(-1)


def compute_code_numerators(states: torch.Tensor) -> torch.Tensor:
    """1MAD numerators, int64 in ``[-510, 510]``: the byte sum of the LCG output minus 510."""
    x = (states * LCG_MULTIPLIER + LCG_INCREMENT) & 0xFFFFFFFF
    return (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24) - CODE_OFFSET


def decode_states(states: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """INT8 values ``[..., T]`` of 16-bit states ``[..., T]`` and E4M3 scales ``[...]``.

    ``q = clamp(round_half_even(fp32(n) · fp32(scale · R)), ±127)`` with both
    products single FP32 multiplies, as every backend computes them.
    """
    numerators = compute_code_numerators(states.to(torch.int64))
    multipliers = block_scales.to(torch.float32) * CODE_RECIPROCAL
    scaled = numerators.to(torch.float32) * multipliers.unsqueeze(-1)
    return scaled.round().clamp(-_INT8_LIMIT, _INT8_LIMIT).to(torch.int8)


def decode_reference(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    layout: TrellisLayout,
) -> torch.Tensor:
    """INT8 values ``[..., T]`` of uint8 ``bitstreams`` ``[..., bytes]`` and E4M3 scales ``[...]``.

    ``q = clamp(round_half_even(fp32(n) · fp32(scale · R)), ±127)`` with both
    products single FP32 multiplies, as every backend computes them.
    """
    return decode_states(compute_states(bitstreams, layout), block_scales)


def decode_variable_rows_reference(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    block_size: int,
    *,
    base_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """INT8 values ``[rows, blocks * block_size]`` for variable-width rows.

    ``bitstreams`` is 1-D uint8 containing concatenated rows of ``blocks``
    blocks each (optionally starting at scalar ``base_offset``), where row
    ``r`` stores ``row_widths[r]`` bytes per block, and ``block_scales`` is
    ``[rows, blocks]`` float8_e4m3fn.
    """
    validate_block_size(block_size)
    if (
        bitstreams.ndim != 1
        or block_scales.ndim != 2
        or tuple(row_widths.shape) != (block_scales.shape[0],)
    ):
        raise ValueError("invalid variable-row trellis reference input shapes")
    if base_offset is not None and base_offset.ndim != 0:
        raise ValueError("base_offset must be a 0-D tensor")
    rows, blocks_per_row = block_scales.shape
    device = bitstreams.device
    if rows == 0 or blocks_per_row == 0:
        return torch.empty((rows, blocks_per_row * block_size), dtype=torch.int8, device=device)
    total_bytes = bitstreams.numel()
    if total_bytes == 0:
        states = torch.zeros((rows, blocks_per_row, block_size), dtype=torch.int64, device=device)
        return decode_states(states, block_scales).reshape(rows, -1)

    widths = row_widths.to(torch.int64)
    safe_widths = widths.clamp(min=1)
    row_prefix = torch.cumsum(widths, dim=0) - widths
    row_base = row_prefix * blocks_per_row
    if base_offset is not None:
        raw_base = base_offset.to(device=device, dtype=torch.int64)
        valid_base = raw_base >= 0
        row_base = row_base + torch.where(valid_base, raw_base, 0)
        stream_limit: int | torch.Tensor = torch.where(valid_base, total_bytes, 0)
    else:
        stream_limit = total_bytes

    block_idx = torch.arange(blocks_per_row, device=device, dtype=torch.int64)
    block_base = row_base[:, None] + block_idx[None, :] * widths[:, None]
    row_bits = widths * 8
    step_bits = row_bits // block_size
    wide_steps = row_bits - step_bits * block_size
    steps = torch.arange(block_size, device=device, dtype=torch.int64)
    offsets = (
        step_bits[:, None] * steps[None, :]
        + ((steps[None, :] + 1) * wide_steps[:, None]) // block_size
    )

    byte0 = offsets >> 3
    bit_in_byte = offsets & 7
    byte1 = (byte0 + 1) % safe_widths[:, None]
    byte2 = (byte0 + 2) % safe_widths[:, None]

    valid_width = widths[:, None, None] > 0
    max_idx = total_bytes - 1
    loaded_bytes: list[torch.Tensor] = []
    for byte_off in (byte0, byte1, byte2):
        idx = block_base[:, :, None] + byte_off[:, None, :]
        mask = valid_width & (idx >= 0) & (idx < stream_limit)
        gathered = bitstreams[idx.clamp(0, max_idx)].to(torch.int64)
        loaded_bytes.append(torch.where(mask, gathered, 0))

    pack24 = (loaded_bytes[0] << 16) | (loaded_bytes[1] << 8) | loaded_bytes[2]
    states = (pack24 >> (8 - bit_in_byte[:, None, :])) & 0xFFFF
    return decode_states(states, block_scales).reshape(rows, blocks_per_row * block_size)


__all__ = [
    "compute_code_numerators",
    "compute_states",
    "decode_reference",
    "decode_states",
    "decode_variable_rows_reference",
]
