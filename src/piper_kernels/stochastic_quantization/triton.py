"""Triton primitives for stochastic terminal-code selection.

The kernel-side counterpart of :mod:`piper_kernels.stochastic_quantization`.
A caller draws with :func:`random_uniform` by logical element offset, so a
kernel's launch geometry cannot change which values it samples.
"""

# Triton JIT helper signatures intentionally use untyped tensor parameters
# and upper-case constexpr names.
# ruff: noqa: N803

import triton
import triton.language as tl


def seed_argument(seed: int | None) -> int:
    """Return a launch-safe signed scalar with the seed's uint64 bit pattern.

    Accepts a seed already in that signed form, because the weight update paths
    convert once before choosing a backend and hand the same scalar to both the
    reference and Triton implementations. The range check is only wide enough to
    admit that: a seed outside 64 bits either way is rejected rather than
    wrapped, since wrapping would alias it onto 0, which means no seed at all.
    Callers validate their own seeds; `piper_kernels.weights` does so with
    `validate_rounding_seed`.
    """
    if seed is None:
        return 0
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"rounding seed must be a 64-bit integer, got {seed!r}")
    if not -(1 << 63) <= seed < (1 << 64):
        raise ValueError(f"rounding seed must fit 64 bits, got {seed}")
    return seed if seed < (1 << 63) else seed - (1 << 64)


@triton.jit
def random_uniform(seed, offsets):
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
    rounded = lower + (random_uniform(seed, offsets) < probability)
    return tl.where(interior & (probability > 0.0), rounded, deterministic)


__all__ = ["random_uniform", "seed_argument", "stochastic_round_to_int"]
