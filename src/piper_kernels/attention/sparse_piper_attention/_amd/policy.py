"""Targets supported by the wave32 RDNA4 sparse-attention kernel."""

from piper_kernels._triton.targets import AcceleratorTarget

from .._plan import MinmaxScoreSchedule

_RDNA4_MINMAX_SCORE_SCHEDULE = MinmaxScoreSchedule(32, 64, 16)


def supports_target(target: AcceleratorTarget) -> bool:
    return target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201")


def select_minmax_score_schedule() -> MinmaxScoreSchedule:
    """Return the measured FP32 min/max routing-score tile schedule for RDNA4."""
    return _RDNA4_MINMAX_SCORE_SCHEDULE
