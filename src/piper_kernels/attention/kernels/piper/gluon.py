"""Shared vendor-neutral Gluon primitives for dense and sparse Piper Attention."""

# Gluon device parameters and layouts are not Python runtime values.
# ruff: noqa: ANN001, ANN201
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from . import LOG2_255, VALUE_LOG_BOUND_CORRECTION

_GL_LOG2_255 = gl.constexpr(LOG2_255)
_GL_VALUE_LOG_BOUND_CORRECTION = gl.constexpr(VALUE_LOG_BOUND_CORRECTION)
TILE_COLUMN_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 1], [16, 2], [4, 1], [0, 1]))


@gluon.jit
def advance_tile_max(running_max, running_bias, block_max, k_bias):
    """Update winner-frame running maximum across chunks without round-tripping."""
    has_block = block_max != -float("inf")
    has_running = running_max != -float("inf")
    raw_delta = (block_max - running_max) + (k_bias - running_bias)
    delta = gl.where(
        has_block,
        gl.where(has_running, raw_delta, float("inf")),
        -float("inf"),
    )
    tile_wins = delta > 0.0
    next_max = gl.where(tile_wins, block_max, running_max)
    next_bias = gl.where(tile_wins, k_bias, running_bias)
    decay = gl.exp2(-gl.abs(delta))
    old_weight = gl.where(tile_wins, decay, 1.0)
    current_weight = gl.where(tile_wins, 1.0, decay)
    return next_max, next_bias, old_weight, current_weight


@gluon.jit
def conservative_value_log_scale_bound(value_scale_multiplier):
    """Bound log2(scale) from the positive FP32 multiplier's IEEE-754 bits."""
    return value_scale_multiplier.to(gl.int32, bitcast=True).to(gl.float32) * (1.0 / 8388608.0) - (
        127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION
    )


@gluon.jit
def split_bf16_pair(value):
    """Split finite non-negative FP32 values into two BF16 terms (high + low)."""
    # Bitwise shifts avoid tt.fp_to_fp and extra arith.truncf ops in TTGIR.
    bits = value.to(gl.uint32, bitcast=True)
    high = (bits >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    high_fp32 = (bits & 0xFFFF0000).to(gl.float32, bitcast=True)
    residual_bits = (value - high_fp32).to(gl.uint32, bitcast=True)
    low = ((residual_bits + 0x8000) >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    return high, low


@gluon.jit
def load_tile_column(shared, column_index, row_layout: gl.constexpr):
    """Gather one 64-row FP32 column from scratch SMEM/LDS into ``row_layout``."""
    columns = gl.full([64, 1], column_index, gl.int32, TILE_COLUMN_LAYOUT)
    values = gl.reshape(shared.gather(columns, 1), [64])
    return gl.convert_layout(values, row_layout)


@gluon.jit
def store_tile_column(shared, values, column_index):
    """Scatter one 64-row FP32 vector into column ``column_index`` of scratch SMEM/LDS."""
    row_1d_layout: gl.constexpr = gl.SliceLayout(1, TILE_COLUMN_LAYOUT)
    shared.scatter(
        gl.convert_layout(gl.reshape(values, [64]), row_1d_layout)[:, None],
        gl.full([64, 1], column_index, gl.int32, TILE_COLUMN_LAYOUT),
        1,
    )
