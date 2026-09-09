"""Shared Gluon traversal works with scalar and distributed tile positions."""

import sys

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels.attention.kernels.sparse_piper.gluon import tile_offset


@gluon.jit
def _tile_offset_kernel(  # noqa: PLR0913, PLR0917
    routes,
    output,
    routed_count,
    selected_count,
    sparse_blocks,
    route_stride,
    count,
    use_routes,
    distributed: gl.constexpr,
    skip_dense_routing: gl.constexpr,
    tile_stride: gl.constexpr,
):
    if distributed:
        position = gl.arange(0, 16, layout=gl.BlockedLayout([1], [32], [4], [0]))
    else:
        position = gl.program_id(0)
    tile = tile_offset(
        routes,
        position,
        routed_count,
        selected_count,
        sparse_blocks,
        route_stride,
        use_routes,
        skip_dense_routing,
        tile_stride,
    )
    gl.store(output + position, tile, position < count)


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("skip_dense_routing", [False, True])
@pytest.mark.parametrize("tile_stride", [1, 64])
@pytest.mark.parametrize(
    "target",
    [GPUTarget("cuda", 120, 32), GPUTarget("hip", "gfx1201", 32)],
)
def test_tile_offset_compiles_for_both_backends(
    target, distributed, skip_dense_routing, tile_stride
):
    if target.backend == "hip" and sys.platform != "linux":
        pytest.skip("ROCm support is Linux-only")
    constants = {
        "distributed": distributed,
        "skip_dense_routing": skip_dense_routing,
        "tile_stride": tile_stride,
    }
    signature = {name: "i32" for name in _tile_offset_kernel.arg_names if name not in constants}
    signature.update(routes="*u16", output="*i32", use_routes="i1")
    compiled = triton.compile(
        GluonASTSource(_tile_offset_kernel, signature, constexprs=constants),
        target=target,
        options={"num_warps": 4},
    )
    assert ("tt.load" not in compiled.asm["ttgir"]) is skip_dense_routing


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Triton GPU")
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("route_stride", [1, 3])
@pytest.mark.parametrize("tile_stride", [1, 64])
@pytest.mark.parametrize("mode", ["routed", "dense-query", "route-free"])
def test_tile_offset_traverses_sparse_prefix_and_dense_suffix(
    distributed, route_stride, mode, tile_stride
):
    skip_dense_routing = mode == "route-free"
    use_routes = mode == "routed"
    # The unused strided entries are invalid tile indices and must not be selected.
    routes = torch.full((3 * route_stride,), 65535, dtype=torch.uint16, device="cuda")
    routes[::route_stride] = torch.tensor([5, 1, 6], dtype=torch.uint16, device="cuda")
    expected = [5, 1, 6, 7, 8, 9] if use_routes else list(range(10))
    if skip_dense_routing:
        routes = routes[:0]  # No route allocation/load is needed in this specialization.
    output = torch.empty(len(expected), dtype=torch.int32, device="cuda")
    _tile_offset_kernel[(1 if distributed else len(expected),)](
        routes,
        output,
        0 if skip_dense_routing else 3,
        3 if use_routes else 7,
        7,
        route_stride,
        len(expected),
        use_routes,
        distributed,
        skip_dense_routing,
        tile_stride,
        num_warps=4,
    )
    torch.testing.assert_close(
        output,
        torch.tensor(expected, dtype=torch.int32, device="cuda") * tile_stride,
        rtol=0,
        atol=0,
    )
