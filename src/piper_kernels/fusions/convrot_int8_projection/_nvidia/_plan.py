"""Resolved NVIDIA fused-projection implementation and launch choices for dense and sparse Piper."""

from dataclasses import dataclass
from typing import Literal

from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

type ProjectionOperation = Literal["query", "key", "value"]

# The Gluon feature layout and register epilogues require this fixed tile.
GLUON_BLOCK_M = 128
GLUON_BLOCK_K = 64
GLUON_NUM_WARPS = 4
GLUON_NUM_STAGES = 3


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan(ProjectionExecutionPlan):
    """Complete implementation and launch choices for one NVIDIA projection."""

    kernel: Literal["gluon_async_copy", "triton"] = "triton"

    def __post_init__(self) -> None:
        if self.kernel not in ("gluon_async_copy", "triton"):
            raise ValueError("unsupported NVIDIA projection kernel")
        if self.kernel == "gluon_async_copy" and (
            self.block_m != GLUON_BLOCK_M
            or self.block_k != GLUON_BLOCK_K
            or self.num_warps != GLUON_NUM_WARPS
            or self.num_stages != GLUON_NUM_STAGES
            or self.heads_per_program != 1
            or self.group_m != 0
            or not self.round_rsqrt_to_nearest
        ):
            raise ValueError("Gluon projections require the fixed D128 async-copy schedule")
