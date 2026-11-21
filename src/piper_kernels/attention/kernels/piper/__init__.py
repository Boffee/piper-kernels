"""Shared vendor-neutral Piper Attention constants (Triton-free)."""

LOG2_255 = 7.994353436858858
# Pad the analytical maximum to the next effective FP32 constant so the
# lowered expression remains an upper bound after integer-to-FP32 rounding.
VALUE_LOG_BOUND_CORRECTION = 0.086085
