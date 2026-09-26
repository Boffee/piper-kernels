"""SM120 configurations for dense projection launchers."""

from functools import partial

from .. import triton as projection

_CONFIG = projection.ProjectionConfig(
    block_k=128,
    heads_per_program=2,
    num_warps=8,
    num_stages=3,
    round_rsqrt_to_nearest=True,
)

project_query = partial(projection.project_query, config=_CONFIG)
project_key = partial(projection.project_key, config=_CONFIG)
# The compact mean projection needs more CTAs than the full matrix projection.
project_value = partial(projection.project_value, config=_CONFIG, packed_amd=False, mean_block_n=32)
