"""Sparse-Piper-specific operand preparation primitives."""

from __future__ import annotations

import triton
import triton.language as tl

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

_P_UINT8_RANGE = tl.constexpr(255.0)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)


@triton.jit
def quantize_value_tile(
    values,
    value_mean,
    valid_rows,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Center and encode one projected tile in sparse-Piper's transposed V format."""
    centered = tl.where(
        valid_rows[:, None, None],
        values - value_mean[None, :, :],
        0.0,
    )
    grouped = tl.reshape(
        tl.permute(centered, (1, 0, 2)),
        (
            heads_per_program,
            block_m // scale_rows,
            scale_rows,
            head_dim,
        ),
    )
    maximum = tl.max(tl.max(tl.abs(grouped), axis=3), axis=2)
    value_scale = maximum / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(grouped / value_scale[:, :, None, None])
    return (
        tl.reshape(quantized, (heads_per_program, block_m, head_dim)),
        value_scale * _P_UINT8_RANGE,
    )
