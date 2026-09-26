"""RDNA4 minmax scoring stays FP32 and shares the routing-score contract."""

from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._routing import (
    packed_routes_from_summaries,
    routing_scores,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING

pytestmark = pytest.mark.gpu


@pytest.fixture
def generator():
    if not torch.cuda.is_available():
        pytest.skip("requires RDNA4 scoring")
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not (target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201")):
        pytest.skip("requires RDNA4 scoring")
    return torch.Generator(device="cuda").manual_seed(671)


@pytest.fixture
def summaries(generator):
    def make(rows, keys, *, heads=3, kv_heads=None):
        # Dense features, strided query rows and padded key-prefix views.
        query = torch.randn((2, heads, rows * 2, 136), device="cuda", generator=generator)
        primary = torch.randn(
            (2, heads if kv_heads is None else kv_heads, keys + 3, 144),
            device="cuda",
            generator=generator,
        )
        auxiliary = torch.empty_like(primary).normal_(generator=generator)
        return query[:, :, ::2, :128], primary[:, :, :keys, :128], auxiliary[:, :, :keys, :128]

    return make


def _fp64_scores(query, primary, auxiliary, scale=None):
    head_groups = query.shape[1] // primary.shape[1]
    primary = primary.repeat_interleave(head_groups, dim=1)
    auxiliary = auxiliary.repeat_interleave(head_groups, dim=1)
    left = query.double() @ primary.double().transpose(-1, -2)
    right = query.double() @ auxiliary.double().transpose(-1, -2)
    if scale is not None:
        left, right = left * scale, right * scale
    return torch.maximum(left, right)


@pytest.mark.parametrize("rows", [1, 27, 64])
@pytest.mark.parametrize("keys", [7, 513, 1562])
@pytest.mark.parametrize("scale", [None, 0.125, -0.5])
def test_scores_match_fp64_with_strides_tails_and_scaling(summaries, rows, keys, scale):
    tensors = summaries(rows, keys)
    selected = _backend.select_minmax_scores(*tensors)
    assert selected is not None
    actual = routing_scores(*tensors, _MINMAX_ROUTING, score_scale=scale)
    expected = _fp64_scores(*tensors, scale)
    assert actual.is_contiguous()
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("rows", [65, 128, 384, 385])
@pytest.mark.parametrize("scale", [None, 0.125, -0.5])
def test_large_query_scores_match_fp64_with_shared_kv(summaries, rows, scale):
    tensors = summaries(rows, 513, kv_heads=1)
    selected = _backend.select_minmax_scores(*tensors)
    assert selected is not None
    actual = routing_scores(*tensors, _MINMAX_ROUTING, score_scale=scale)
    torch.testing.assert_close(actual.double(), _fp64_scores(*tensors, scale), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("rows", [384, 40], ids=["full_chunk", "final_chunk"])
def test_h56_150k_routing_chunks_match_fp64(generator, rows):
    # 150,000 tokens use 2,343 complete sparse key blocks and 2,344 query
    # blocks: six full 384-block chunks followed by one 40-block chunk.
    tensors = [
        torch.randn((1, 56, blocks, 128), device="cuda", generator=generator)
        for blocks in (rows, 2343, 2343)
    ]
    assert _backend.select_minmax_scores(*tensors) is not None
    actual = routing_scores(*tensors, _MINMAX_ROUTING)
    # Keep FP64 temporaries bounded while exercising the complete target launch.
    for head in range(56):
        expected = _fp64_scores(*(tensor[:, head : head + 1] for tensor in tensors))
        torch.testing.assert_close(
            actual[:, head : head + 1].double(), expected, atol=2e-5, rtol=2e-5
        )


def test_large_query_routes_preserve_exact_ties_across_chunks(summaries, monkeypatch):
    query, primary, auxiliary = [tensor.round() for tensor in summaries(385, 513, kv_heads=1)]
    # Repeated keys force ties with nonzero scores; zero query rows also check the
    # lower-index tie policy at both sides of the 384-row routing chunk boundary.
    primary[:, :, 1::2] = primary[:, :, :-1:2]
    auxiliary[:, :, 1::2] = auxiliary[:, :, :-1:2]
    # This duplicate pair straddles the scoring kernel's 64-key tile boundary.
    primary[:, :, 64] = primary[:, :, 63]
    auxiliary[:, :, 64] = auxiliary[:, :, 63]
    query[:, :, [0, 383, 384]] = 0
    counts = (1, 129, 513)
    layout = _resolve_route_layout(
        _normalize_head_keep_ratios(tuple(count / 513 for count in counts)), 513, query.device
    )
    expected_scores = _fp64_scores(query, primary, auxiliary).cpu()
    expected_routes = torch.cat(
        [
            expected_scores[:, head]
            .argsort(dim=-1, descending=True, stable=True)[..., :count]
            .sort(dim=-1)
            .values
            for head, count in enumerate(counts)
        ],
        dim=-1,
    )
    score = _backend.select_minmax_scores(query, primary, auxiliary)
    assert score is not None
    score_calls = Mock(wraps=score)
    monkeypatch.setattr(_backend._score_backend, "minmax_scores", score_calls)
    actual = packed_routes_from_summaries(query, primary, auxiliary, layout, _MINMAX_ROUTING)

    torch.testing.assert_close(actual.indices.cpu().long(), expected_routes, atol=0, rtol=0)
    assert [call.args[0].shape[2] for call in score_calls.call_args_list] == [384, 1]
    lower_indices = torch.cat([torch.arange(count) for count in counts])
    torch.testing.assert_close(expected_routes[0, [0, 383, 384]], lower_indices.expand(3, -1))


def test_minmax_scoring_uses_the_shared_keyword_contract(summaries):
    query, primary, auxiliary = summaries(27, 513)
    keywords = {"query_summary": query, "key_primary": primary, "key_aux": auxiliary}
    selected = _backend.select_minmax_scores(**keywords)
    assert selected is not None
    actual = selected(**keywords, score_scale=-0.5)
    expected = _fp64_scores(query, primary, auxiliary, -0.5)
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)


def test_scores_support_unaligned_feature_views(generator):
    tensors = [
        torch.randn((2, 3, rows, width), device="cuda", generator=generator)[..., 1:129]
        for rows, width in [(27, 137), (513, 139), (513, 141)]
    ]
    assert all(tensor.data_ptr() % 16 != 0 for tensor in tensors)
    actual = routing_scores(*tensors, _MINMAX_ROUTING)
    torch.testing.assert_close(actual.double(), _fp64_scores(*tensors), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("scale", [None, 0.125])
@pytest.mark.parametrize(("rows", "heads"), [(64, 56), (384, 3)])
def test_scores_bypass_torch_gemms_and_match_exact_integer_products(
    summaries, monkeypatch, scale, rows, heads
):
    tensors = [tensor.round() for tensor in summaries(rows, 1562, heads=heads)]
    expected = _fp64_scores(*tensors, scale).float()
    with monkeypatch.context() as patch:
        patch.setattr(torch, "bmm", Mock(side_effect=AssertionError("unfused scoring")))
        patch.setattr(torch, "baddbmm", Mock(side_effect=AssertionError("unfused scoring")))
        actual = routing_scores(*tensors, _MINMAX_ROUTING, score_scale=scale)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("query_scale", "key_scale"),
    [(1e20, 1e-20), (1e-20, 1e20), (torch.finfo(torch.float32).max, 1e-37)],
)
def test_scores_preserve_finite_fp32_range(summaries, generator, query_scale, key_scale):
    query, primary, auxiliary = summaries(27, 513)
    query.uniform_(-1, 1, generator=generator).mul_(query_scale)
    primary.uniform_(-1, 1, generator=generator).mul_(key_scale)
    auxiliary.uniform_(-1, 1, generator=generator).mul_(key_scale)
    actual = routing_scores(query, primary, auxiliary, _MINMAX_ROUTING)
    expected = _fp64_scores(query, primary, auxiliary)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=1e-4, rtol=2e-5)


def test_scores_propagate_nan(summaries):
    query, primary, auxiliary = summaries(1, 7)
    auxiliary[:, :, 0, 0] = float("nan")
    actual = routing_scores(query, primary, auxiliary, _MINMAX_ROUTING)
    expected = _fp64_scores(query, primary, auxiliary)
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5, equal_nan=True)


@pytest.mark.parametrize("rows", [64, 384])
def test_scores_use_nondefault_stream_without_extra_score_storage(summaries, rows):
    tensors = summaries(rows, 1562)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = routing_scores(*tensors, _MINMAX_ROUTING)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    # The only allocator-backed temporary is the returned score matrix. Large
    # requests may reuse a block rounded to the allocator's 2 MiB granularity.
    output_bytes = actual.numel() * actual.element_size()
    allocation_limit = output_bytes + 512
    if output_bytes >= 10 * 1024**2:
        allocation_quantum = 2 * 1024**2
        allocation_limit = (
            (output_bytes + allocation_quantum - 1) // allocation_quantum * allocation_quantum
        )
    assert torch.cuda.max_memory_allocated() - allocated <= allocation_limit
    torch.testing.assert_close(actual.double(), _fp64_scores(*tensors), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("rows", [27, 384])
def test_scores_graph_replay_uses_updated_summaries(summaries, generator, rows):
    tensors = summaries(rows, 513)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        routing_scores(*tensors, _MINMAX_ROUTING)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = routing_scores(*tensors, _MINMAX_ROUTING)
    for _ in range(3):
        tensors[0].normal_(generator=generator)
        graph.replay()
        torch.testing.assert_close(actual.double(), _fp64_scores(*tensors), atol=2e-5, rtol=2e-5)
