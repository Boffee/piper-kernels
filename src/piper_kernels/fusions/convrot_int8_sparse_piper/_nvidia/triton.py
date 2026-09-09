"""NVIDIA configurations for the shared fused projection launchers."""

from dataclasses import replace
from functools import partial

from triton.language.extra.cuda import libdevice

from .. import triton as projection
from .._layout import TILE_ROWS

_QUERY_CONFIG = projection.ProjectionConfig(
    block_m=TILE_ROWS,
    block_k=128,
    heads_per_program=2,
    num_warps=8,
    num_stages=3,
    rsqrt_fn=libdevice.rsqrt_rn,
)
_CONTEXT_CONFIG = replace(_QUERY_CONFIG, block_m=2 * TILE_ROWS)

project_query = partial(projection.project_query, config=_QUERY_CONFIG)
project_key = partial(projection.project_key, config=_CONTEXT_CONFIG)
project_value = partial(projection.project_value, config=_CONTEXT_CONFIG)
