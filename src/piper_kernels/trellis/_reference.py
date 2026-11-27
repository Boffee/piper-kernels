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
    WRAP_GUARD_BYTES,
    TrellisLayout,
    validate_block_size,
)

_INT8_LIMIT = 127


def compute_states(bitstreams: torch.Tensor, layout: TrellisLayout) -> torch.Tensor:
    """Each step's 16-bit state, int64 ``[..., T]``, from uint8 ``[..., bytes_per_block]``.

    State ``t`` is the linear 16-bit little-endian window starting at bit
    ``offset_t`` inside the ``bytes_per_block``-byte stored block (whose trailing
    ``WRAP_GUARD_BYTES`` duplicate bytes ``0`` and ``1``).
    """
    device = bitstreams.device
    shifts = torch.arange(8, device=device)
    bits = ((bitstreams.long().unsqueeze(-1) >> shifts) & 1).flatten(-2)
    steps = torch.arange(layout.block_size, device=device, dtype=torch.int64)
    offsets = layout.step_bits * steps + ((steps + 1) * layout.num_wide_steps) // layout.block_size
    window = torch.arange(STATE_BITS, device=device)
    positions = offsets[:, None] + window
    weights = 1 << torch.arange(STATE_BITS, device=device)
    return (bits[..., positions] * weights).sum(-1)


def compute_code_numerators(states: torch.Tensor) -> torch.Tensor:
    """1MAD numerators, int64 in ``[-510, 510]``: the byte sum of the LCG output minus 510."""
    x = (states * LCG_MULTIPLIER + LCG_INCREMENT) & 0xFFFFFFFF
    return (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24) - CODE_OFFSET


def decode_states_fp32(states: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """Unrounded block-scaled FP32 values ``[..., T]`` of 16-bit states and E4M3 scales.

    Computes ``fp32(n) · (fp32(scale) · R)`` without per-weight INT8 rounding or clamping.
    """
    numerators = compute_code_numerators(states.to(torch.int64))
    multipliers = block_scales.to(torch.float32) * CODE_RECIPROCAL
    return numerators.to(torch.float32) * multipliers.unsqueeze(-1)


def decode_states(states: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """INT8 values ``[..., T]`` of 16-bit states ``[..., T]`` and E4M3 scales ``[...]``.

    ``q = clamp(round_half_even(fp32(n) · fp32(scale · R)), ±127)`` with both
    products single FP32 multiplies, as every backend computes them.
    """
    return (
        decode_states_fp32(states, block_scales)
        .round()
        .clamp(-_INT8_LIMIT, _INT8_LIMIT)
        .to(torch.int8)
    )


def decode_reference(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    layout: TrellisLayout,
    *,
    round_int8: bool = True,
) -> torch.Tensor:
    """Decoded values ``[..., T]`` of uint8 ``bitstreams`` ``[..., bytes]`` and E4M3 scales.

    Returns quantized INT8 values when ``round_int8=True`` or unrounded
    block-scaled FP32 values when ``round_int8=False``.
    """
    states = compute_states(bitstreams, layout)
    return (
        decode_states(states, block_scales)
        if round_int8
        else decode_states_fp32(states, block_scales)
    )


def decode_variable_rows_reference(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    block_size: int,
    *,
    base_offset: torch.Tensor | None = None,
    round_int8: bool = True,
) -> torch.Tensor:
    """Decoded values ``[rows, blocks * block_size]`` for variable-width rows.

    ``bitstreams`` is 1-D uint8 containing concatenated rows of ``blocks``
    blocks each (optionally starting at scalar ``base_offset``), where row
    ``r`` stores ``row_widths[r]`` bytes per block (including ``WRAP_GUARD_BYTES``),
    and ``block_scales`` is ``[rows, blocks]`` float8_e4m3fn.
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
    out_dtype = torch.int8 if round_int8 else torch.float32
    if rows == 0 or blocks_per_row == 0:
        return torch.empty((rows, blocks_per_row * block_size), dtype=out_dtype, device=device)
    total_bytes = bitstreams.numel()
    if total_bytes == 0:
        states = torch.zeros((rows, blocks_per_row, block_size), dtype=torch.int64, device=device)
        values = (
            decode_states(states, block_scales)
            if round_int8
            else decode_states_fp32(states, block_scales)
        )
        return values.reshape(rows, -1)

    widths = row_widths.to(torch.int64)
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
    row_bits = (widths - WRAP_GUARD_BYTES).clamp(min=0) * 8
    step_bits = row_bits // block_size
    wide_steps = row_bits - step_bits * block_size
    steps = torch.arange(block_size, device=device, dtype=torch.int64)
    offsets = (
        step_bits[:, None] * steps[None, :]
        + ((steps[None, :] + 1) * wide_steps[:, None]) // block_size
    )

    byte0 = offsets >> 3
    bit_in_byte = offsets & 7
    valid_width = widths[:, None, None] > WRAP_GUARD_BYTES
    max_idx = total_bytes - 1
    loaded_bytes: list[torch.Tensor] = []
    for delta in (0, 1, 2):
        byte_off = byte0 + delta
        idx = block_base[:, :, None] + byte_off[:, None, :]
        mask = (
            valid_width
            & (byte_off[:, None, :] < widths[:, None, None])
            & (idx >= 0)
            & (idx < stream_limit)
        )
        gathered = bitstreams[idx.clamp(0, max_idx)].to(torch.int64)
        loaded_bytes.append(torch.where(mask, gathered, 0))

    pack24 = loaded_bytes[0] | (loaded_bytes[1] << 8) | (loaded_bytes[2] << 16)
    states = (pack24 >> bit_in_byte[:, None, :]) & 0xFFFF
    values = (
        decode_states(states, block_scales)
        if round_int8
        else decode_states_fp32(states, block_scales)
    )
    return values.reshape(rows, blocks_per_row * block_size)


__all__ = [
    "compute_code_numerators",
    "compute_states",
    "decode_reference",
    "decode_states",
    "decode_states_fp32",
    "decode_variable_rows_reference",
]
