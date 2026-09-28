"""Resolved NVIDIA sparse-projection implementation and launch choices."""

from dataclasses import dataclass
from typing import Literal

from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan:
    """Select fixed Gluon arithmetic or a configurable shared Triton projection."""

    kernel: Literal["gluon_async_copy", "triton"]
    execution_plan: ProjectionExecutionPlan | None = None

    def __post_init__(self) -> None:
        if self.kernel not in ("gluon_async_copy", "triton"):
            raise ValueError("unsupported NVIDIA projection kernel")
        if (self.kernel == "triton") != (self.execution_plan is not None):
            raise ValueError("only Triton projections take a shared execution plan")
