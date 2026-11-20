"""Immutable launch schedules for sparse Piper attention."""

from typing import NamedTuple


class AttentionSchedule(NamedTuple):
    """Query tile size and warp count selected for an attention launch."""

    block_m: int
    num_warps: int


class MinmaxScoreSchedule(NamedTuple):
    """Tile dimensions and fence policy selected for fused FP32 min/max summary scoring."""

    block_m: int
    block_n: int
    block_k: int
    lds_only_fences: bool = False
