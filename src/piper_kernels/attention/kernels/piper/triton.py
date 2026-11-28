"""Shared vendor-neutral Triton primitives for dense and sparse Piper Attention."""

# Triton device-function signatures are not Python runtime values.
# pyright: reportArgumentType=false

import triton
import triton.language as tl

from . import LOG2_255, VALUE_LOG_BOUND_CORRECTION

_P_UINT8_LOG2_RANGE = tl.constexpr(LOG2_255)
_VALUE_LOG_BOUND_CORRECTION = tl.constexpr(VALUE_LOG_BOUND_CORRECTION)


@triton.jit
def max_propagate_nan(a, b):
    """Max reducer that propagates NaNs across lanes."""
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def advance_tile_max_safe(running_max, running_bias, block_max, k_bias, valid_queries=None):
    """Update winner-frame running maximum with NaN, -inf, and same-bias guards."""
    has_nan = (block_max + k_bias) != (block_max + k_bias)
    m_clean = tl.where(has_nan, float("nan"), block_max)
    has_block = ((m_clean != -float("inf")) & (k_bias != -float("inf"))) | has_nan
    has_running = running_max != -float("inf")
    bias_diff = tl.where(k_bias == running_bias, 0.0, k_bias - running_bias)
    raw_delta = (m_clean - running_max) + bias_diff
    delta = tl.where(
        has_block,
        tl.where(has_running, raw_delta, float("inf")),
        -float("inf"),
    )
    tile_wins = (delta > 0.0) | has_nan
    next_max = tl.where(tile_wins, m_clean, running_max)
    next_bias = tl.where(tile_wins, k_bias, running_bias)
    decay = tl.exp2(-tl.abs(delta))
    old_weight = tl.where(has_running, tl.where(tile_wins, decay, 1.0), 0.0)
    current_weight = tl.where(has_block, tl.where(tile_wins, 1.0, decay), 0.0)
    if valid_queries is not None:
        old_weight = tl.where(valid_queries, old_weight, 0.0)
        current_weight = tl.where(valid_queries, current_weight, 0.0)
    safe_block_max = tl.where(has_block, m_clean, 0.0)
    return next_max, next_bias, old_weight, current_weight, has_block, safe_block_max


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


@triton.jit
def clip_kv_window_tiles(
    query_block,
    total_q_rows,
    logical_seq_q,
    q_base_pos,
    key_limit,
    left_clamp_cap,
    window_left,
    window_right,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    packed_gqa: tl.constexpr,
    window_mode: tl.constexpr,
):
    """Clip visible key tile range ``[first_tile, last_tile)`` and clamped window bounds."""
    is_causal_mode: tl.constexpr = window_mode == "causal"
    is_windowed_mode: tl.constexpr = window_mode == "windowed"
    key_limit64 = key_limit.to(tl.int64)
    total_tiles = tl.cdiv(key_limit64, block_n)
    if is_windowed_mode:
        eff_wl = tl.minimum(window_left.to(tl.int64), left_clamp_cap.to(tl.int64))
        eff_wr = tl.minimum(
            window_right.to(tl.int64),
            left_clamp_cap.to(tl.int64) + logical_seq_q.to(tl.int64),
        )
    else:
        eff_wl = tl.zeros((), dtype=tl.int64)
        eff_wr = tl.zeros((), dtype=tl.int64)
    if is_causal_mode or is_windowed_mode:
        q_tile_first = query_block.to(tl.int64) * block_m
        q_tile_last = tl.minimum(q_tile_first + block_m, total_q_rows.to(tl.int64)) - 1
        if packed_gqa:
            seq64 = logical_seq_q.to(tl.int64)
            same_head = (q_tile_first // seq64) == (q_tile_last // seq64)
            q_idx_min = tl.where(same_head, q_tile_first % seq64, 0)
            q_idx_max = tl.where(same_head, q_tile_last % seq64, seq64 - 1)
        else:
            q_idx_min = q_tile_first
            q_idx_max = q_tile_last
        q_min = q_base_pos.to(tl.int64) + q_idx_min
        q_max = q_base_pos.to(tl.int64) + q_idx_max
        k_min = (
            tl.maximum(0, q_min - eff_wl)
            if (is_windowed_mode and window_left >= 0)
            else tl.zeros((), dtype=tl.int64)
        )
        if is_causal_mode:
            k_max_excl = tl.minimum(key_limit64, tl.maximum(0, q_max + 1))
        elif window_right >= 0:
            k_max_excl = tl.minimum(key_limit64, tl.maximum(0, q_max + eff_wr + 1))
        else:
            k_max_excl = key_limit64
        first_tile = tl.minimum(total_tiles, k_min // block_n).to(tl.int32)
        last_tile = tl.maximum(
            first_tile,
            tl.minimum(total_tiles, tl.cdiv(k_max_excl, block_n)).to(tl.int32),
        )
    else:
        first_tile = tl.zeros((), dtype=tl.int32)
        last_tile = total_tiles.to(tl.int32)
    return first_tile, last_tile, eff_wl, eff_wr
