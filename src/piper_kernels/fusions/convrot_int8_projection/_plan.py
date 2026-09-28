"""Immutable execution choices for shared ConvRot INT8 projections."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProjectionExecutionPlan:
    """Compute settings; each consumer owns its numerical groups and target tuning."""

    block_k: int
    heads_per_program: int
    num_warps: int
    num_stages: int
    block_m: int = 64
    group_m: int = 0
    round_rsqrt_to_nearest: bool = False
