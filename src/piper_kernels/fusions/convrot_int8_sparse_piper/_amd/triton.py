"""RDNA4 configurations for the shared fused projection launchers."""

from functools import partial

from .. import triton as projection
from .._layout import TILE_ROWS

# Q/K's Hadamard and RoPE epilogues need smaller tiles to bound LDS/register use.
# Reuse a small input row group across heads, including beyond the input cache size.
_QK_CONFIG = projection.ProjectionConfig(
    block_m=TILE_ROWS,
    block_k=64,
    heads_per_program=1,
    num_warps=4,
    num_stages=2,
    group_m=8,
)
_VALUE_CONFIG = projection.ProjectionConfig(
    block_m=2 * TILE_ROWS,
    block_k=64,
    heads_per_program=2,
    num_warps=8,
    num_stages=2,
    group_m=8,
)

project_query = partial(projection.project_query, config=_QK_CONFIG)
project_key = partial(projection.project_key, config=_QK_CONFIG)
project_value = partial(projection.project_value, config=_VALUE_CONFIG)
