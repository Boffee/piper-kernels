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
    shifts = torch.arange(7, -1, -1, device=bitstreams.device)
    bits = ((bitstreams.long().unsqueeze(-1) >> shifts) & 1).flatten(-2)
    offsets = torch.tensor(layout.compute_state_offsets(), device=bitstreams.device)
    window = torch.arange(STATE_BITS, device=bitstreams.device)
    positions = (offsets[:, None] + window) % layout.block_bits
    weights = 1 << torch.arange(STATE_BITS - 1, -1, -1, device=bitstreams.device)
    return (bits[..., positions] * weights).sum(-1)


def compute_code_numerators(states: torch.Tensor) -> torch.Tensor:
    """1MAD numerators, int64 in ``[-510, 510]``: the byte sum of the LCG output minus 510."""
    x = (states * LCG_MULTIPLIER + LCG_INCREMENT) & 0xFFFFFFFF
    return (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24) - CODE_OFFSET


def decode_reference(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    layout: TrellisLayout,
) -> torch.Tensor:
    """INT8 values ``[..., T]`` of uint8 ``bitstreams`` ``[..., bytes]`` and E4M3 scales ``[...]``.

    ``q = clamp(round_half_even(fp32(n) · fp32(scale · R)), ±127)`` with both
    products single FP32 multiplies, as every backend computes them.
    """
    numerators = compute_code_numerators(compute_states(bitstreams, layout))
    reciprocal = torch.tensor(CODE_RECIPROCAL, dtype=torch.float32, device=bitstreams.device)
    multipliers = block_scales.to(torch.float32) * reciprocal
    scaled = numerators.to(torch.float32) * multipliers.unsqueeze(-1)
    return scaled.round().clamp(-_INT8_LIMIT, _INT8_LIMIT).to(torch.int8)


def decode_variable_rows_reference(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """INT8 values ``[rows, blocks * block_size]`` for variable-width rows.

    ``bitstreams`` is 1-D uint8 containing concatenated rows of ``blocks``
    blocks each, where row ``r`` stores ``row_widths[r]`` bytes per block, and
    ``block_scales`` is ``[rows, blocks]`` float8_e4m3fn.
    """
    validate_block_size(block_size)
    if (
        bitstreams.ndim != 1
        or block_scales.ndim != 2
        or tuple(row_widths.shape) != (block_scales.shape[0],)
    ):
        raise ValueError("invalid variable-row trellis reference input shapes")
    rows, blocks_per_row = block_scales.shape
    output = torch.empty(
        rows, blocks_per_row * block_size, dtype=torch.int8, device=bitstreams.device
    )
    offset = 0
    for row, width in enumerate(row_widths.tolist()):
        row_bytes = blocks_per_row * width
        layout = TrellisLayout.from_storage(block_size, width)
        row_streams = bitstreams[offset : offset + row_bytes].reshape(blocks_per_row, width)
        output[row] = decode_reference(row_streams, block_scales[row], layout).reshape(-1)
        offset += row_bytes
    return output


__all__ = [
    "compute_code_numerators",
    "compute_states",
    "decode_reference",
    "decode_variable_rows_reference",
]
