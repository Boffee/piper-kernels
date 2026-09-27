"""Immutable launch schedules for sparse Piper attention."""

from typing import NamedTuple


class AttentionSchedule(NamedTuple):
    """Query tile size and warp count selected for an attention launch."""

    block_m: int
    num_warps: int
