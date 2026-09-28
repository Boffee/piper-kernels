"""Fixed dense projection choices for this validated target family."""

from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

EXECUTION_PLAN = ProjectionExecutionPlan(
    block_k=64, heads_per_program=1, num_warps=4, num_stages=2, group_m=8
)

PACKED_VALUE = True
VALUE_MEAN_BLOCK_N = None
