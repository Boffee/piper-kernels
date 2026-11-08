"""Shared sparse-prefix/dense-suffix traversal for Gluon attention kernels."""

# Gluon device parameters are not Python runtime values.
# ruff: noqa: ANN001, ANN201
# pyright: reportArgumentType=false

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def tile_offset(
    routes,
    position,
    routed_count,
    selected_count,
    sparse_blocks,
    route_stride,
    use_routes,
    skip_dense_routing: gl.constexpr = False,
    tile_stride: gl.constexpr = 1,
):
    """Map a traversal position to a K/V tile in the caller's indexing units."""
    if skip_dense_routing:
        return position * tile_stride
    else:
        safe_position = gl.minimum(position, routed_count - 1)
        route = gl.load(routes + safe_position * route_stride).to(gl.int32)
        sparse_tile = gl.where(use_routes, route, position)
        sparse_start = sparse_tile * tile_stride
        dense_start = sparse_blocks * tile_stride + (position - selected_count) * tile_stride
        return gl.where(position < selected_count, sparse_start, dense_start)


@gluon.jit
def pair_tile_offsets(
    routes,
    pair_index,
    tile_count,
    routed_count,
    selected_count,
    sparse_blocks,
    route_stride,
    use_routes,
    skip_dense_routing: gl.constexpr = False,
    tile_stride: gl.constexpr = 1,
):
    """Return two tile offsets, duplicating the last tile for an incomplete pair."""
    position_0 = pair_index * 2
    position_1 = gl.minimum(position_0 + 1, tile_count - 1)
    offset_0 = tile_offset(
        routes,
        position_0,
        routed_count,
        selected_count,
        sparse_blocks,
        route_stride,
        use_routes,
        skip_dense_routing,
        tile_stride,
    )
    offset_1 = tile_offset(
        routes,
        position_1,
        routed_count,
        selected_count,
        sparse_blocks,
        route_stride,
        use_routes,
        skip_dense_routing,
        tile_stride,
    )
    return offset_0, offset_1
