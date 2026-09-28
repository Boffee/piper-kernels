"""Fixed dense projection choices for this validated target family."""

from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

EXECUTION_PLAN = ProjectionExecutionPlan(
    block_k=128,
    heads_per_program=2,
    num_warps=8,
    num_stages=3,
    round_rsqrt_to_nearest=True,
)

PACKED_VALUE = False
VALUE_MEAN_BLOCK_N = 32
