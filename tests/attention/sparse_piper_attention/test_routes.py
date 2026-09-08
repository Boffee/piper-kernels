"""Shared GPU route selection preserves exact stable FP32 top-k semantics."""

from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
    _ResolvedRouteLayout,
)
from piper_kernels.attention.sparse_piper_attention._routes import (
    PackedRouteBuilder,
    _select_portable_routes,
)

pytestmark = pytest.mark.gpu


@pytest.fixture
def route_selector():
    if not torch.cuda.is_available():
        pytest.skip("requires a supported GPU route selector")
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not (
        target.is_cuda_capability(12, 0)
        or (target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201"))
    ):
        pytest.skip("requires SM120 or RDNA4 route selection")
    selector = _backend.select_route_selector(torch.empty(0, device="cuda", dtype=torch.uint16))
    assert selector is not None
    return selector


def _layout(counts, device):
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return _ResolvedRouteLayout(
        head_keep_blocks=torch.tensor(counts, device=device, dtype=torch.int32),
        route_head_offsets=torch.tensor(offsets, device=device, dtype=torch.int32),
        routes_per_query=offsets[-1],
    )


def _expected_routes(scores, counts, *, query_block_offset=0):
    layout = _layout(counts, "cpu")
    routes = torch.full(
        (scores.shape[0], query_block_offset + scores.shape[2] + 2, layout.routes_per_query),
        0xA5A5,
        dtype=torch.uint16,
    )
    _select_portable_routes(
        scores.cpu(),
        routes,
        layout.route_head_offsets.tolist(),
        counts,
        query_block_offset=query_block_offset,
    )
    return routes


@pytest.mark.parametrize("key_blocks", [1, 7, 511, 512, 513, 1562, 4097, 65_536])
@pytest.mark.parametrize("strided", [False, True])
def test_stable_routes_match_portable_policy(route_selector, key_blocks, strided):
    generator = torch.Generator().manual_seed(546)
    storage = torch.randint(-16, 17, (2, 4, 8, key_blocks + 9), generator=generator).float()
    scores = storage[:, :, 1:7:2, :key_blocks]
    if not strided:
        scores = scores.contiguous()
    counts = [1, max(1, key_blocks // 4), max(1, key_blocks - 1), key_blocks]
    expected = _expected_routes(scores, counts, query_block_offset=2)
    # Preserve noncontiguous batch/head/query strides and a dense key prefix on GPU.
    gpu_storage = storage.cuda()
    gpu_scores = gpu_storage[:, :, 1:7:2, :key_blocks] if strided else scores.cuda()
    layout = _layout(counts, "cuda")
    actual = torch.full_like(expected, 0xA5A5, device="cuda")
    route_selector(
        gpu_scores,
        actual,
        layout.head_keep_blocks,
        layout.route_head_offsets,
        query_block_offset=2,
    )
    # Compare guard rows as well as all retained indices, including key65535.
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("pattern", ["ties", "signed_zero", "adjacent", "extremes"])
def test_fp32_ordering_and_ties_across_selector_tiles(route_selector, pattern):
    one = torch.tensor(1.0)
    if pattern == "ties":
        values = torch.tensor([-1.0, 2.0, 2.0, -1.0])
    elif pattern == "signed_zero":
        values = torch.tensor([-0.0, 0.0])
    elif pattern == "adjacent":
        next_one = torch.nextafter(one, one + 1)
        values = torch.stack((one, next_one, -one, -next_one))
    else:
        limit = torch.finfo(torch.float32)
        values = torch.tensor([-limit.max, -limit.tiny, limit.tiny, limit.max])
    scores = values.repeat(513)[:1025].expand(2, 4, 3, -1).contiguous()
    counts = [1, 511, 512, 1025]
    expected = _expected_routes(scores, counts, query_block_offset=1)
    layout = _layout(counts, "cuda")
    actual = torch.full_like(expected, 0xA5A5, device="cuda")
    route_selector(
        scores.cuda(),
        actual,
        layout.head_keep_blocks,
        layout.route_head_offsets,
        query_block_offset=1,
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("query_blocks", [64, 27])
def test_h3_chunk_routes_avoid_host_readback(route_selector, monkeypatch, query_blocks):
    generator = torch.Generator(device="cuda").manual_seed(547)
    scores = torch.randint(
        -32, 33, (1, 56, query_blocks, 1562), device="cuda", generator=generator
    ).float()
    layout = _resolve_route_layout(_normalize_head_keep_ratios((0.25,) * 56), 1562, scores.device)
    counts = layout.head_keep_blocks.cpu().tolist()
    expected = _expected_routes(scores, counts)[:, :query_blocks]
    # A selected builder must not take the portable CPU-metadata or per-head-sort path.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", Mock(side_effect=AssertionError("host readback")))
        patch.setattr(torch.Tensor, "tolist", Mock(side_effect=AssertionError("host readback")))
        patch.setattr(torch, "argsort", Mock(side_effect=AssertionError("portable sort")))
        builder = PackedRouteBuilder(
            layout,
            batch=1,
            heads=56,
            query_blocks=query_blocks,
            sparse_key_blocks=1562,
            device=scores.device,
        )
        builder.write(scores, query_block_offset=0)
    torch.testing.assert_close(builder.routes.indices.cpu(), expected, atol=0, rtol=0)


def test_route_selector_uses_nondefault_stream(route_selector):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        scores = torch.randn((2, 3, 5, 1562), device="cuda")
        layout = _layout([1, 390, 1562], "cuda")
        actual = torch.empty((2, 5, layout.routes_per_query), device="cuda", dtype=torch.uint16)
        route_selector(
            scores,
            actual,
            layout.head_keep_blocks,
            layout.route_head_offsets,
            query_block_offset=0,
        )
    torch.cuda.current_stream().wait_stream(stream)
    expected = _expected_routes(scores, [1, 390, 1562])[:, :5]
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


def test_route_selector_graph_replay_uses_updated_scores(route_selector):
    scores = torch.randn((1, 3, 5, 513), device="cuda")
    layout = _layout([1, 129, 513], "cuda")
    actual = torch.empty((1, 5, layout.routes_per_query), device="cuda", dtype=torch.uint16)

    def launch():
        route_selector(
            scores,
            actual,
            layout.head_keep_blocks,
            layout.route_head_offsets,
            query_block_offset=0,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    for _ in range(3):
        scores.normal_()
        graph.replay()
        expected = _expected_routes(scores, [1, 129, 513])[:, :5]
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
