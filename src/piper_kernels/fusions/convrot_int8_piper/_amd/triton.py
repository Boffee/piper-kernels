"""RDNA4 configurations for dense projection launchers."""

from functools import partial

from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .. import triton as projection

_CONFIG = ProjectionExecutionPlan(
    block_k=64, heads_per_program=1, num_warps=4, num_stages=2, group_m=8
)

project_query = partial(projection.project_query, execution_plan=_CONFIG)
project_key = partial(projection.project_key, execution_plan=_CONFIG)
project_value = partial(
    projection.project_value, execution_plan=_CONFIG, packed_amd=True, mean_block_n=None
)
