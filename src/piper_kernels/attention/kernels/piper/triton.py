"""Shared vendor-neutral Triton primitives for dense and sparse Piper Attention."""

# Triton device-function signatures are not Python runtime values.
# pyright: reportArgumentType=false

import triton
import triton.language as tl

from . import LOG2_255, VALUE_LOG_BOUND_CORRECTION

_P_UINT8_LOG2_RANGE = tl.constexpr(LOG2_255)
_VALUE_LOG_BOUND_CORRECTION = tl.constexpr(VALUE_LOG_BOUND_CORRECTION)


@triton.jit
def conservative_value_log_scale_bound(value_scale_multiplier):
    """Bound log2(scale) from the positive FP32 multiplier's IEEE-754 bits."""
    multiplier_bits = value_scale_multiplier.to(tl.int32, bitcast=True)
    return multiplier_bits.to(tl.float32) * (1.0 / 8388608.0) - (
        127.0 + _P_UINT8_LOG2_RANGE - _VALUE_LOG_BOUND_CORRECTION
    )


@triton.jit
def pack_amd_wmma_token(token):
    """Exchange token bits 3 and 4 within a K64 tile for RDNA4 WMMA reduction order."""
    return (token & ~24) | ((token & 8) << 1) | ((token & 16) >> 1)
