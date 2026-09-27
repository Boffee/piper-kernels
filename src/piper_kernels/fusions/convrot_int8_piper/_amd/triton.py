"""RDNA4 configurations for dense projection launchers."""

from functools import partial

from .. import triton as projection

_CONFIG = projection.ProjectionConfig(
    block_k=64, heads_per_program=1, num_warps=4, num_stages=2, group_m=8
)

project_query = partial(projection.project_query, config=_CONFIG)
project_key = partial(projection.project_key, config=_CONFIG)
project_value = partial(
    projection.project_value, config=_CONFIG, packed_amd=True, mean_block_n=None
)
