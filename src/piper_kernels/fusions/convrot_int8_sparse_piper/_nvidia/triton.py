"""Bind shared projection launchers to target policy."""

from functools import partial

from .. import triton as projection
from . import policy

project_query = partial(projection.project_query, execution_plan=policy.QUERY_EXECUTION_PLAN)
project_key = partial(projection.project_key, execution_plan=policy.CONTEXT_EXECUTION_PLAN)
project_value = partial(projection.project_value, execution_plan=policy.CONTEXT_EXECUTION_PLAN)
