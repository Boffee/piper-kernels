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
    value_mean_block_n: int | None = None

    def resolve_value_mean_block_n(self, head_dim: int) -> int:
        """Return explicit chunk-mean N tiling or fall back to one CTA per head."""
        if self.value_mean_block_n is not None:
            return self.value_mean_block_n
        return head_dim * self.heads_per_program
