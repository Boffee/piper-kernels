"""Validated ROCm fused sparse projection and output integrations on RDNA4."""

import sys
from typing import Literal

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .._layout import TILE_ROWS

type ProjectionOperation = Literal["query", "key", "value"]


def supports_head_dim(head_dim: int) -> bool:
    """Head widths handled by the RDNA4 projection schedules."""
    return head_dim in (64, 128)


def supports_target(target: AcceleratorTarget) -> bool:
    return (
        sys.platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    )


_QK_EXECUTION_PLAN = ProjectionExecutionPlan(
    block_m=TILE_ROWS,
    block_k=64,
    heads_per_program=1,
    num_warps=4,
    num_stages=2,
    group_m=8,
    lds_only_fences=True,
)
_D64_VALUE_EXECUTION_PLAN = ProjectionExecutionPlan(
    block_m=2 * TILE_ROWS,
    block_k=64,
    heads_per_program=2,
    num_warps=8,
    num_stages=2,
    group_m=8,
    lds_only_fences=True,
)
_D128_VALUE_EXECUTION_PLAN = ProjectionExecutionPlan(
    block_m=2 * TILE_ROWS,
    block_k=64,
    heads_per_program=1,
    num_warps=8,
    num_stages=2,
    group_m=8,
    lds_only_fences=True,
)


def select_execution_plan(
    operation: ProjectionOperation,
    *,
    head_dim: int,
) -> ProjectionExecutionPlan:
    """Select RDNA4 Q/K/V projection plans, keeping V tiles at 128 columns across 8 warps."""
    if operation != "value":
        return _QK_EXECUTION_PLAN
    return _D64_VALUE_EXECUTION_PLAN if head_dim == 64 else _D128_VALUE_EXECUTION_PLAN
