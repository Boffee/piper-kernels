"""SM89 configuration for the shared fused projection launchers."""

from functools import partial

from .. import triton as projection
from .._layout import TILE_ROWS

# One configuration serves Q, K, and V. SM120's tiles need more than SM89's 99 KiB of shared
# memory (K) or spill (V's 128x256 accumulator). Tiles of 64 rows by one head keep four warps
# within the register budget, and grouping eight row blocks reuses input rows across heads.
_CONFIG = projection.ProjectionConfig(
    block_m=TILE_ROWS,
    block_k=128,
    heads_per_program=1,
    num_warps=4,
    num_stages=3,
    group_m=8,
    round_rsqrt_to_nearest=True,
)

project_query = partial(projection.project_query, config=_CONFIG)
project_key = partial(projection.project_key, config=_CONFIG)
project_value = partial(projection.project_value, config=_CONFIG)
