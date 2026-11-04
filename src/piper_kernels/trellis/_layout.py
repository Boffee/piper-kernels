"""Block layout and decode constants of trellis-coded weights.

The format is described in this package's README. This module owns the parameters
that vary per tensor (``TrellisLayout``) and the constants the format fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

STATE_BITS = 16
"""Bits in a trellis state: each weight decodes from a 16-bit window of its stream."""

MIN_BITS_PER_WEIGHT = 3
MAX_BITS_PER_WEIGHT = 6
"""Every step shifts in between 3 and 6 new bits."""

MIN_BLOCK_SIZE = STATE_BITS
MAX_BLOCK_SIZE = 256
"""Weights per block. At least 16, so a state window never wraps twice."""

LCG_MULTIPLIER = 34038481
LCG_INCREMENT = 76625530
"""QTIP's 1MAD code: ``x = (34038481 · state + 76625530) mod 2^32``."""

CODE_OFFSET = 510
"""The code numerator is the sum of ``x``'s four bytes minus 510, in ``[-510, 510]``."""

CODE_RECIPROCAL = float.fromhex("0x1.bb688ep-8")
"""FP32 value nearest ``1 / 147.800537109375``, QTIP's 1MAD normalizer, exact in FP32."""


@dataclass(frozen=True, slots=True)
class TrellisLayout:
    """How each block of a trellis-coded matrix stores its code bits.

    A ``[N, K]`` matrix is cut into ``K / block_size`` blocks per row, each
    storing ``block_bits`` bits in whole bytes: one step per weight, shifting in
    ``step_bits`` or ``step_bits + 1`` bits with ``step_bits = block_bits //
    block_size``. Steps of both widths are interleaved by a fixed rule, so these
    two numbers determine every step's width. ``block_size`` is from 16 to 256,
    and ``block_bits`` a multiple of 8 from 3 to 6 bits per weight.
    """

    block_size: int
    block_bits: int

    def __post_init__(self) -> None:
        for name in ("block_size", "block_bits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"trellis {name} must be an integer, got {value!r}")
        size = self.block_size
        if not MIN_BLOCK_SIZE <= size <= MAX_BLOCK_SIZE:
            raise ValueError(
                f"trellis blocks hold {MIN_BLOCK_SIZE} to {MAX_BLOCK_SIZE} weights, got {size}"
            )
        lowest, highest = MIN_BITS_PER_WEIGHT * size, MAX_BITS_PER_WEIGHT * size
        if self.block_bits % 8 or not lowest <= self.block_bits <= highest:
            raise ValueError(
                f"trellis blocks of {size} weights take {lowest // 8} to "
                f"{highest // 8} whole bytes (3 to 6 bits per weight), got "
                f"{self.block_bits} bits"
            )

    @classmethod
    def from_storage(cls, block_size: int, bytes_per_block: int) -> TrellisLayout:
        """The layout of streams holding ``bytes_per_block`` bytes per block."""
        return cls(block_size, 8 * bytes_per_block)

    @property
    def step_bits(self) -> int:
        """Bits shifted in at a narrow step; wide steps shift in one more."""
        return self.block_bits // self.block_size

    @property
    def num_wide_steps(self) -> int:
        """Steps that shift in ``step_bits + 1`` bits."""
        return self.block_bits - self.step_bits * self.block_size

    @property
    def bytes_per_block(self) -> int:
        """Stored bytes per block."""
        return self.block_bits // 8

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


__all__ = [
    "CODE_OFFSET",
    "CODE_RECIPROCAL",
    "LCG_INCREMENT",
    "LCG_MULTIPLIER",
    "MAX_BITS_PER_WEIGHT",
    "MAX_BLOCK_SIZE",
    "MIN_BITS_PER_WEIGHT",
    "MIN_BLOCK_SIZE",
    "STATE_BITS",
    "TrellisLayout",
]
