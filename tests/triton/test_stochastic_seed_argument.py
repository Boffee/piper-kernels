"""Tests for the Triton stochastic terminal-code primitives."""

import pytest

from piper_kernels.stochastic_quantization.triton import seed_argument


def test_seed_argument_round_trips_the_unsigned_range() -> None:
    assert seed_argument(None) == 0
    assert seed_argument(0) == 0
    assert seed_argument((1 << 63) - 1) == (1 << 63) - 1
    assert seed_argument(1 << 63) == -(1 << 63)
    assert seed_argument((1 << 64) - 1) == -1


def test_seed_argument_passes_through_an_already_signed_seed() -> None:
    # The weight update paths convert once, then hand the same scalar to the
    # reference and Triton backends so both sample identically.
    for unsigned in (1 << 63, (1 << 64) - 1, (1 << 63) + 12345):
        assert seed_argument(seed_argument(unsigned)) == seed_argument(unsigned)


def test_seed_argument_rejects_seeds_that_do_not_fit_64_bits() -> None:
    # Wrapping would map these onto 0, which is also the no-seed argument.
    with pytest.raises(ValueError, match="64 bits"):
        seed_argument(1 << 64)
    with pytest.raises(ValueError, match="64 bits"):
        seed_argument(-(1 << 63) - 1)
    with pytest.raises(TypeError, match="64-bit integer"):
        seed_argument(True)
