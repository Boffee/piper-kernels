"""Bind shared dense projection launchers to target policy."""

from functools import partial

from .. import triton as projection
from . import policy

project_query = partial(projection.project_query, execution_plan=policy.EXECUTION_PLAN)
project_key = partial(projection.project_key, execution_plan=policy.EXECUTION_PLAN)
project_value = partial(
    projection.project_value,
    execution_plan=policy.EXECUTION_PLAN,
    packed_amd=policy.PACKED_VALUE,
    mean_block_n=policy.VALUE_MEAN_BLOCK_N,
)
