"""AMD RDNA4 support and scheduling policy for sparse Piper attention."""

from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from .._contracts import MinmaxScoreSchedule

_MINMAX_SMALL_QUERY_SCHEDULE = MinmaxScoreSchedule(32, 64, 16, lds_only_fences=True)
_MINMAX_LARGE_QUERY_SCHEDULE = MinmaxScoreSchedule(64, 32, 16, lds_only_fences=True)
_MINMAX_LARGE_QUERY_THRESHOLD = 128


def supports_target(target: AcceleratorTarget) -> bool:
    """Return whether target runs the wave32 RDNA4 sparse-attention kernels."""
    return target.is_hip_architecture(*GFX120X_ARCHITECTURES)


def select_minmax_score_schedule(query_blocks: int) -> MinmaxScoreSchedule:
    """Return the measured FP32 min/max routing-score tile schedule for RDNA4.

    All RDNA4 schedules enable LDS-only fences to omit global cache invalidations
    at shared-memory barriers without changing output bits; 64x32x16 wins from
    128 query blocks upward while 32x64x16 avoids idle waves on shorter chunks.
    """
    if query_blocks >= _MINMAX_LARGE_QUERY_THRESHOLD:
        return _MINMAX_LARGE_QUERY_SCHEDULE
    return _MINMAX_SMALL_QUERY_SCHEDULE
