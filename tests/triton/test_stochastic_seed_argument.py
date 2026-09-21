"""Tests for the Triton stochastic terminal-code primitives."""

import pytest

from piper_kernels.stochastic_quantization.triton import seed_argument


def test_seed_argument_round_trips_the_unsigned_range() -> None:
    assert seed_argument(None) == 0
    assert seed_argument(0) == 0
    assert seed_argument((1 << 63) - 1) == (1 << 63) - 1
    assert seed_argument(1 << 63) == -(1 << 63)
    assert seed_argument((1 << 64) - 1) == -1


def test_seed_argument_rejects_an_already_narrowed_seed() -> None:
    # A signed seed here means a caller narrowed it before the launch, which is
    # the conversion this helper owns. Wrapping it would alias a valid seed.
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        seed_argument(-1)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        seed_argument(1 << 64)
    with pytest.raises(TypeError, match="unsigned 64-bit"):
        seed_argument(True)
