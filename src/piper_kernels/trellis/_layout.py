"""Block layout and decode constants of trellis-coded weights.

The format is described in this package's README. This module owns the parameters
that vary per tensor (``TrellisLayout``) and the constants the format fixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

STATE_BITS = 16
"""Bits in a trellis state: each weight decodes from a 16-bit window of its stream."""

MIN_BITS_PER_WEIGHT = 2
MAX_BITS_PER_WEIGHT = 6
"""Every step shifts in between 2 and 6 new bits."""

MIN_BLOCK_SIZE = STATE_BITS
MAX_BLOCK_SIZE = 256
"""Weights per block. At least 16, so a state window never wraps twice."""

BLOCK_SCALE_BITS = 8
"""Each block stores one FP8 E4M3 scale."""

WRAP_GUARD_BYTES = 2
"""Trailing wrap guard bytes per block (copy of bytes 0 and 1 for wrap-free 16-bit LE loads)."""

BLOCK_OVERHEAD_BITS = BLOCK_SCALE_BITS + 8 * WRAP_GUARD_BYTES
"""Total non-code metadata overhead bits per block (1 FP8 scale + 2 wrap guard bytes)."""

LCG_MULTIPLIER = 34038481
LCG_INCREMENT = 76625530
"""QTIP's 1MAD code: ``x = (34038481 · state + 76625530) mod 2^32``."""

CODE_OFFSET = 510
"""The code numerator is the sum of ``x``'s four bytes minus 510, in ``[-510, 510]``."""

CODE_DIVISOR = 147.800537109375
"""QTIP's 1MAD normalizer, exact in FP32: the code value is ``numerator / 147.8``."""

CODE_RECIPROCAL = float.fromhex("0x1.bb688ep-8")
"""FP32 value nearest ``1 / CODE_DIVISOR``, exact in FP32."""


@dataclass(frozen=True, slots=True)
class TrellisLayout:
    """How each block of a trellis-coded matrix stores its code bits.

    A ``[N, K]`` matrix is cut into ``K / block_size`` blocks per row, each
    encoding ``block_bits`` code bits in ``block_bits // 8`` whole bytes plus
    ``WRAP_GUARD_BYTES`` (2) trailing wrap guard bytes: one step per weight,
    shifting in ``step_bits`` or ``step_bits + 1`` bits with
    ``step_bits = block_bits // block_size``. Steps of both widths are
    interleaved by a fixed rule, so these two numbers determine every step's
    width. ``block_size`` is from 16 to 256, and ``block_bits`` is a multiple of
    8 from 2 to 6 code bits per weight.
    """

    block_size: int
    block_bits: int

    def __post_init__(self) -> None:
        size = validate_block_size(self.block_size)
        if isinstance(self.block_bits, bool) or not isinstance(self.block_bits, int):
            raise TypeError(f"trellis block_bits must be an integer, got {self.block_bits!r}")
        lowest, highest = MIN_BITS_PER_WEIGHT * size, MAX_BITS_PER_WEIGHT * size
        if self.block_bits % 8 or not lowest <= self.block_bits <= highest:
            min_stored, max_stored = stored_bytes_bounds(size)
            raise ValueError(
                f"trellis blocks of {size} weights take {(lowest + 7) // 8} to "
                f"{highest // 8} whole code bytes ({min_stored} to {max_stored} stored bytes "
                f"with {WRAP_GUARD_BYTES} guard bytes, {MIN_BITS_PER_WEIGHT} to "
                f"{MAX_BITS_PER_WEIGHT} bits per weight), got {self.block_bits} bits"
            )

    @classmethod
    def from_storage(cls, block_size: int, bytes_per_block: int) -> TrellisLayout:
        """The layout of streams holding ``bytes_per_block`` stored bytes per block."""
        if isinstance(bytes_per_block, bool) or not isinstance(bytes_per_block, int):
            raise TypeError(f"trellis bytes_per_block must be an integer, got {bytes_per_block!r}")
        return cls(block_size, 8 * (bytes_per_block - WRAP_GUARD_BYTES))

    @property
    def step_bits(self) -> int:
        """Bits shifted in at a narrow step; wide steps shift in one more."""
        return self.block_bits // self.block_size

    @property
    def num_wide_steps(self) -> int:
        """Steps that shift in ``step_bits + 1`` bits."""
        return self.block_bits - self.step_bits * self.block_size

    @property
    def code_bytes(self) -> int:
        """Payload code bytes per block (before the trailing wrap guard bytes)."""
        return self.block_bits // 8

    @property
    def bytes_per_block(self) -> int:
        """Stored bytes per block (``code_bytes`` plus ``WRAP_GUARD_BYTES``)."""
        return self.code_bytes + WRAP_GUARD_BYTES

    def compute_step_widths(self) -> tuple[int, ...]:
        """Bits shifted in at each step: ``step_bits`` plus one at evenly spaced steps.

        Step ``t`` is wide when ``⌊(t + 1)·w / T⌋ > ⌊t·w / T⌋`` for ``w`` wide steps
        among ``T``, so the first ``t`` steps hold ``⌊t·w / T⌋`` wide ones.
        """
        size, wide, narrow = self.block_size, self.num_wide_steps, self.step_bits
        return tuple(narrow + (t + 1) * wide // size - t * wide // size for t in range(size))

    def compute_state_offsets(self) -> tuple[int, ...]:
        """Bit offset of each step's state window in the block's stream.

        ``offset_t`` is the sum of the widths of steps ``1..t`` (``offset_0 = 0``):
        ``step_bits·t + ⌊(t + 1)·w / T⌋``.
        """
        size, wide, narrow = self.block_size, self.num_wide_steps, self.step_bits
        return tuple(narrow * t + (t + 1) * wide // size for t in range(size))


def stored_bytes_bounds(block_size: int) -> tuple[int, int]:
    """Return inclusive ``(min_stored_bytes, max_stored_bytes)`` per block."""
    size = validate_block_size(block_size)
    lowest, highest = MIN_BITS_PER_WEIGHT * size, MAX_BITS_PER_WEIGHT * size
    return (
        (lowest + 7) // 8 + WRAP_GUARD_BYTES,
        highest // 8 + WRAP_GUARD_BYTES,
    )


def validate_block_size(block_size: int) -> int:
    """Validate that ``block_size`` is an integer in ``[16, 256]`` and return it."""
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError(f"trellis block_size must be an integer, got {block_size!r}")
    if not MIN_BLOCK_SIZE <= block_size <= MAX_BLOCK_SIZE:
        raise ValueError(
            f"trellis blocks hold {MIN_BLOCK_SIZE} to {MAX_BLOCK_SIZE} weights, got {block_size}"
        )
    return block_size


def resolve_block_size(in_features: int) -> int:
    """The encoder's block size for ``in_features``-wide rows: the largest multiple
    of 16, up to 256, that divides it. Raises ``TypeError`` or ``ValueError`` when invalid."""
    if isinstance(in_features, bool) or not isinstance(in_features, int):
        raise TypeError(f"trellis in_features must be an integer, got {in_features!r}")
    if in_features <= 0:
        raise ValueError(f"trellis rows must be a positive multiple of 16 wide, got {in_features}")
    for block_size in range(MAX_BLOCK_SIZE, 0, -16):
        if in_features % block_size == 0:
            return block_size
    raise ValueError(f"trellis rows must be a positive multiple of 16 wide, got {in_features}")


def resolve_layout(in_features: int, bits_per_weight: float) -> TrellisLayout:
    """The layout that encodes ``in_features``-wide rows in at most ``bits_per_weight``.

    ``bits_per_weight`` counts each block's code bits, its ``WRAP_GUARD_BYTES``
    wrap guard bytes, and its FP8 scale; a block of
    ``resolve_block_size(in_features)`` weights takes the most whole code bytes
    within it. Raises ``ValueError`` when that leaves a block outside 2 to 6
    code bits per weight.
    """
    block_size = resolve_block_size(in_features)
    if isinstance(bits_per_weight, bool) or not isinstance(bits_per_weight, (int, float)):
        raise TypeError(f"trellis bits_per_weight must be a real number, got {bits_per_weight!r}")
    try:
        budget = math.floor(Fraction(bits_per_weight) * block_size) - BLOCK_OVERHEAD_BITS
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"trellis bits_per_weight must be finite, got {bits_per_weight!r}"
        ) from error
    block_bits = budget - budget % 8
    lowest, highest = MIN_BITS_PER_WEIGHT * block_size, MAX_BITS_PER_WEIGHT * block_size
    if not lowest <= block_bits <= highest:
        low = (lowest + BLOCK_OVERHEAD_BITS) / block_size
        high = (highest + BLOCK_OVERHEAD_BITS) / block_size
        raise ValueError(
            f"trellis rows {in_features} wide take blocks of {block_size} weights, which "
            f"store {low:g} to {high:g} bits per weight with their guard bytes and scales, "
            f"got {bits_per_weight}"
        )
    return TrellisLayout(block_size, block_bits)


def default_layout_for_block_size(
    block_size: int, bytes_per_block: int | None = None
) -> TrellisLayout:
    """Return ``TrellisLayout`` for ``block_size`` from explicit bytes or a ~4.5 bpw default."""
    size = validate_block_size(block_size)
    if bytes_per_block is not None:
        return TrellisLayout.from_storage(size, bytes_per_block)
    return TrellisLayout(size, ((9 * size) // 2 - BLOCK_OVERHEAD_BITS) & ~7)


__all__ = [
    "BLOCK_OVERHEAD_BITS",
    "BLOCK_SCALE_BITS",
    "CODE_DIVISOR",
    "CODE_OFFSET",
    "CODE_RECIPROCAL",
    "LCG_INCREMENT",
    "LCG_MULTIPLIER",
    "MAX_BITS_PER_WEIGHT",
    "MAX_BLOCK_SIZE",
    "MIN_BITS_PER_WEIGHT",
    "MIN_BLOCK_SIZE",
    "STATE_BITS",
    "WRAP_GUARD_BYTES",
    "TrellisLayout",
    "resolve_block_size",
    "resolve_layout",
    "stored_bytes_bounds",
    "validate_block_size",
]
