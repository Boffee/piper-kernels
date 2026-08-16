"""Shared Triton primitives for stochastic terminal-code selection."""

# Triton JIT helper signatures intentionally use untyped tensor parameters
# and upper-case constexpr names.
# ruff: noqa: N803

import triton
import triton.language as tl


def seed_argument(seed: int | None) -> int:
    """Return a launch-safe signed scalar with the seed's uint64 bit pattern."""
    if seed is None:
        return 0
    return seed if seed < (1 << 63) else seed - (1 << 64)


@triton.jit
def _random(seed, offsets):
    """Draw by logical element offset so launch geometry cannot affect samples."""
    return tl.rand(seed, offsets.to(tl.uint64))


@triton.jit
def stochastic_round_to_int(
    values,
    deterministic,
    seed,
    offsets,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
):
    """Round finite interior values to adjacent integers."""
    interior = (values > QMIN) & (values < QMAX)
    safe = tl.where(interior, values, 0.0)
    lower = tl.floor(safe)
    probability = safe - lower
    rounded = lower + (_random(seed, offsets) < probability)
    return tl.where(interior & (probability > 0.0), rounded, deterministic)


__all__ = ["seed_argument", "stochastic_round_to_int"]
