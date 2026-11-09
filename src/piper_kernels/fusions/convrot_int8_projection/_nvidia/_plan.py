"""Resolved NVIDIA fused-projection implementation and launch choices for dense and sparse Piper."""

from dataclasses import dataclass
from typing import Literal

from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

type ProjectionOperation = Literal["query", "key", "value"]

# The Gluon feature layout and register epilogues require this fixed tile. The shared
# policy builds the one Gluon plan from these constants, which the fragments compile with.
GLUON_BLOCK_M = 128
GLUON_BLOCK_K = 64
GLUON_NUM_WARPS = 4
GLUON_NUM_STAGES = 3


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan(ProjectionExecutionPlan):
    """Complete implementation and launch choices for one NVIDIA projection."""

    kernel: Literal["gluon_async_copy", "triton"] = "triton"
