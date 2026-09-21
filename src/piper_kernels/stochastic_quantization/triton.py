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

from . import signed_seed


def seed_argument(seed: int | None) -> int:
    """Return a launch-safe signed scalar with an unsigned seed's uint64 bits.

    For a caller launching a kernel directly with the seed it was given. A
    caller whose seed has already crossed an operator boundary holds the signed
    form and passes it through, supplying 0 for no seed; narrowing it again
    would reject it here.
    """
    if seed is None:
        return 0
    return signed_seed(seed)


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
