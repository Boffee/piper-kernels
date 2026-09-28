"""Validated NVIDIA projection and chunked output integration targets."""

from dataclasses import replace

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .._layout import TILE_ROWS
from ._plan import NvidiaExecutionPlan


def uses_async_copies(target: AcceleratorTarget) -> bool:
    """Select SM89's projections, whose D128 kernels stage operands with ``cp.async``."""
    return target.is_cuda_capability(8, 9)


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native sparse Piper attention targets: exact SM120 and SM89."""
    return target.is_cuda_capability(12, 0) or uses_async_copies(target)


# Triton configuration for the shapes that the Gluon kernels do not cover. SM120's tiles need more
# than SM89's 99 KiB of shared memory (K) or spill (V's 128x256 accumulator). Tiles of 64 rows by
# one head keep four warps within the register budget, and grouping eight row blocks reuses input
# rows across heads.
_SM89_TRITON_EXECUTION_PLAN = ProjectionExecutionPlan(
    block_m=TILE_ROWS,
    block_k=128,
    heads_per_program=1,
    num_warps=4,
    num_stages=3,
    group_m=8,
    round_rsqrt_to_nearest=True,
)


_GLUON_EXECUTION_PLAN = NvidiaExecutionPlan("gluon_async_copy")
_TRITON_EXECUTION_PLAN = NvidiaExecutionPlan("triton", _SM89_TRITON_EXECUTION_PLAN)


def select_execution_plan(
    *, input_features: int, head_dim: int, rotary_dim: int = 0, operands_aligned: bool
) -> NvidiaExecutionPlan:
    """Resolve the async-copy backend's implementation from host metadata.

    Gluon's fixed D128 epilogue pairs RoPE features within registers. Whole K64
    slices and aligned input/weight pointers keep the shared GEMM's 16-byte copies valid.
    """
    if head_dim == 128 and input_features % 64 == 0 and rotary_dim % 32 == 0 and operands_aligned:
        return _GLUON_EXECUTION_PLAN
    return _TRITON_EXECUTION_PLAN


QUERY_EXECUTION_PLAN = ProjectionExecutionPlan(
    block_m=TILE_ROWS,
    block_k=128,
    heads_per_program=2,
    num_warps=8,
    num_stages=3,
    round_rsqrt_to_nearest=True,
)
CONTEXT_EXECUTION_PLAN = replace(QUERY_EXECUTION_PLAN, block_m=2 * TILE_ROWS)
