"""Shared Triton-free Split-KV decode schedules and split sizing for Piper KV pools."""

from __future__ import annotations

from dataclasses import dataclass

CENTER_CHUNK = 1024
VALUE_TILE_TOKENS = 64
KEY_SCALES_GROUPED = CENTER_CHUNK // VALUE_TILE_TOKENS
BLOCK_QH = 16
MAX_DECODE_SPLITS = 32
_FALLBACK_TILES_PER_CHUNK_CAP = 8
TARGET_ACTIVE_CTAS = 64
SPLIT_NUM_STAGES = 1
REDUCE_NUM_WARPS = 4
PREFILL_MIN_SEQ_Q = 64


def _cdiv(value: int, divisor: int) -> int:
    """Return ceiling integer division ``ceil(value / divisor)``."""
    return (value + divisor - 1) // divisor


def _split_num_warps(head_dim: int) -> int:
    return 16 if head_dim == 128 else 8


@dataclass(frozen=True, slots=True)
class DecodeSchedule:
    """Target-specific kernel and Split-KV schedule for ``piper_attention_decode``."""

    block_n: int
    num_warps: int
    num_stages: int
    tail_qk_precision: str
    pv_precision: str
    max_tail_splits: int
    compiler_options: tuple[tuple[str, object], ...] = ()


def default_decode_schedule(head_dim: int) -> DecodeSchedule:
    """Return the portable baseline Split-KV decode schedule for ``head_dim``."""
    return DecodeSchedule(
        block_n=16,
        num_warps=_split_num_warps(head_dim),
        num_stages=SPLIT_NUM_STAGES,
        tail_qk_precision="ieee",
        pv_precision="ieee",
        max_tail_splits=2,
    )


def select_split_counts(  # noqa: PLR0913
    *,
    batch: int,
    kv_heads: int,
    gqa_group: int,
    seq_q: int,
    max_chunks: int,
    num_slots: int,
    tail_cap: int,
    active_chunks_hint: int,
    sm_count: int,
    schedule: DecodeSchedule,
    window_left: int = -1,
) -> tuple[int, int]:
    """Select frozen-chunk and live-tail Split-KV counts for one decode workload."""
    has_chunks = max_chunks > 0 and num_slots > 0
    has_tail = tail_cap > 0
    q_tiles = _cdiv(gqa_group * seq_q, BLOCK_QH)
    base_blocks = max(1, batch * kv_heads * q_tiles)

    eff_tail_cap = tail_cap if window_left < 0 else min(tail_cap, seq_q + window_left)
    tail_splits = 0
    if has_tail:
        if (
            schedule.max_tail_splits > 2
            and eff_tail_cap >= 128
            and base_blocks <= max(16, sm_count // 4)
        ):
            tail_splits = min(schedule.max_tail_splits, _cdiv(eff_tail_cap, schedule.block_n))
        elif schedule.max_tail_splits <= 2 and eff_tail_cap >= 128 and base_blocks <= 16:
            tail_splits = schedule.max_tail_splits
        else:
            tail_splits = 1

    int8_splits = 0
    if has_chunks:
        hint_chunks = max_chunks if active_chunks_hint <= 0 else min(max_chunks, active_chunks_hint)
        if window_left >= 0:
            hint_chunks = min(hint_chunks, _cdiv(seq_q + window_left, CENTER_CHUNK))
        max_int8 = MAX_DECODE_SPLITS - tail_splits
        target_splits = max(1, TARGET_ACTIVE_CTAS // base_blocks)
        int8_splits = min(
            max_int8,
            max(1, min(hint_chunks * _FALLBACK_TILES_PER_CHUNK_CAP, target_splits)),
        )

    return int8_splits, tail_splits
