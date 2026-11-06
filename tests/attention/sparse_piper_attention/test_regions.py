"""Independent regional budgets and one shared softmax over interleaved keys."""

from unittest.mock import Mock

import pytest
import torch
from _compile_capture import TargetCapturePass

from piper_kernels import SparsePiperAttention, SparsePiperKeyRegion
from piper_kernels.attention.sparse_piper_attention import _backend, _routing
from piper_kernels.attention.sparse_piper_attention._budget import _normalize_head_keep_ratios
from piper_kernels.attention.sparse_piper_attention._regions import resolve_region_route_layout
from piper_kernels.attention.sparse_piper_attention._routes import PackedRouteBuilder


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    if request.param == "cuda" and (
        not torch.cuda.is_available()
        or _backend.select_attention_backend(torch.empty((), device="cuda")) is None
    ):
        pytest.skip("requires a native sparse Piper backend")
    return (
        torch.device("cuda", torch.cuda.current_device())
        if request.param == "cuda"
        else torch.device("cpu")
    )


def _layout(ratios, regions, device):
    return resolve_region_route_layout(
        _normalize_head_keep_ratios(ratios),
        tuple(region.num_blocks for region in regions),
        tuple(region.sparse for region in regions),
        device,
    )


def test_budget_rounding_is_independent_per_region():
    attention = SparsePiperAttention((0.5, 0.5))
    regions = (
        SparsePiperKeyRegion(3),
        SparsePiperKeyRegion(2, sparse=False),
        SparsePiperKeyRegion(3),
    )
    assert attention.resolve_region_keep_blocks(regions) == ((2, 1), (2, 2), (2, 1))
    layout = _layout((0.5, 0.5), regions, torch.device("cpu"))
    assert layout.head_keep_blocks.tolist() == [6, 4]
    assert layout.route_head_offsets.tolist() == [0, 6, 10]


def test_each_video_retains_its_own_ten_percent(device):
    regions = (SparsePiperKeyRegion(10),) * 3
    layout = _layout((0.1, 0.1), regions, device)
    # Every score in video 3 exceeds every score in videos 1 and 2.
    scores = torch.arange(30, device=device, dtype=torch.float32).expand(2, 2, 3, -1).clone()
    builder = PackedRouteBuilder(
        layout, batch=2, heads=2, query_blocks=3, sparse_key_blocks=30, device=device
    )
    builder.write(scores, query_block_offset=0)
    expected = torch.tensor([9, 19, 29, 9, 19, 29], dtype=torch.uint16, device=device)
    torch.testing.assert_close(builder.routes.indices, expected.expand(2, 3, -1), atol=0, rtol=0)


def test_regional_forcing_and_ties_use_absolute_ids(device):
    # Neither region boundary aligns with a forced-membership bitset word.
    regions = (
        SparsePiperKeyRegion(3),
        SparsePiperKeyRegion(2, sparse=False),
        SparsePiperKeyRegion(35),
    )
    layout = _layout((0.2, 0.5), regions, device)
    ids = torch.tensor([[2, 39], [1, 34]], dtype=torch.int32, device=device)
    counts = torch.tensor([2, 2], dtype=torch.int32, device=device)
    mask = _backend.build_force_tile_bitmask(ids, counts, 40)
    builder = PackedRouteBuilder(
        layout,
        batch=2,
        heads=2,
        query_blocks=2,
        sparse_key_blocks=40,
        device=device,
        force_tile_mask=mask,
    )
    scores = torch.zeros((2, 2, 2, 40), device=device)
    # Exercise chunk offsets while preserving guard rows from the other chunk.
    builder.write(scores[:, :, 1:], query_block_offset=1)
    builder.write(scores[:, :, :1], query_block_offset=0)
    actual = builder.routes.indices.cpu().long()
    budgets = SparsePiperAttention((0.2, 0.5)).resolve_region_keep_blocks(regions)
    for query in range(2):
        expected = []
        forced = [2, 39] if query == 0 else [1, 34]
        for head in range(2):
            expected.extend([forced[0], 3, 4])
            expected.extend([*range(5, 5 + budgets[2][head] - 1), forced[1]])
        torch.testing.assert_close(actual[:, query], torch.tensor(expected).expand(2, -1))


def test_absolute_routes_cover_uint16_limit(device):
    regions = (SparsePiperKeyRegion(65534, sparse=False), SparsePiperKeyRegion(2))
    layout = _layout((0.5,), regions, device)
    builder = PackedRouteBuilder(
        layout, batch=1, heads=1, query_blocks=1, sparse_key_blocks=65536, device=device
    )
    # Only the final sparse region occupies score storage.
    builder.write(
        torch.tensor([[[[0.0, 1.0]]]], device=device), query_block_offset=0, compact_scores=True
    )
    expected = torch.cat((torch.arange(65534), torch.tensor([65535])))
    torch.testing.assert_close(builder.routes.indices.cpu().long().flatten(), expected)


@pytest.mark.parametrize(
    ("head_dim", "routing", "padded"), [(64, "mean", False), (128, "minmax", True)]
)
def test_interleaved_regions_share_softmax_and_preserve_dense_queries(
    device, head_dim, routing, padded
):
    regions = (
        SparsePiperKeyRegion(2),
        SparsePiperKeyRegion(1, sparse=False),
        SparsePiperKeyRegion(4),
    )
    attention = SparsePiperAttention((0.5, 1.0), routing=routing)
    q = torch.zeros((1, 448, 2, head_dim), device=device)
    k = torch.zeros((1, 448, 1, head_dim), device=device)
    block_values = torch.tensor([1, 3, 10, 20, 30, 40, 50], dtype=torch.float32, device=device)
    v = block_values.repeat_interleave(64)[None, :, None, None].expand_as(k).contiguous()
    lengths = torch.tensor(
        [64, 3, 13, 5, 64, 7, 64] if padded else [64] * 7, device=device, dtype=torch.int32
    )
    valid = (torch.arange(64, device=device)[None] < lengths[:, None]).flatten()
    if padded:
        q[:, ~valid] = 1000
        k[:, ~valid] = -1000
        v[:, ~valid] = 2000
    actual = attention(
        q,
        k,
        v,
        key_regions=regions,
        sparse_query_blocks=2,
        block_lengths=lengths if padded else None,
    )
    selected = torch.tensor([0, 2, 3, 4], device=device)
    sparse_mean = (block_values[selected] * lengths[selected]).sum() / lengths[selected].sum()
    dense_mean = (block_values * lengths).sum() / lengths.sum()
    expected = torch.full_like(actual, dense_mean)
    expected[:, :128, 0] = sparse_mean
    torch.testing.assert_close(actual[:, valid], expected[:, valid], atol=0.01, rtol=0.001)


def test_dense_regions_skip_scoring_and_full_keep_skips_all_routing(device, monkeypatch):
    regions = (
        SparsePiperKeyRegion(2, sparse=False),
        SparsePiperKeyRegion(3),
        SparsePiperKeyRegion(1, sparse=False),
    )
    q = torch.randn((1, 384, 2, 64), device=device)
    original = _routing.routing_scores
    widths = []

    def record_scores(query, key, aux, mode, **kwargs):
        widths.append(key.shape[2])
        return original(query, key, aux, mode, **kwargs)

    monkeypatch.setattr(_routing, "routing_scores", record_scores)
    SparsePiperAttention((0.3, 0.5))(q, q, q, key_regions=regions)
    assert set(widths) == {3}
    monkeypatch.setattr(
        _routing, "routing_scores", Mock(side_effect=AssertionError("full keep scored"))
    )
    full = SparsePiperAttention((1.0, 1.0))
    torch.testing.assert_close(
        full(q, q, q, key_regions=regions), full(q, q, q, sparse_key_blocks=6), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    ("regions", "sequence"),
    [
        ((), 128),
        ((SparsePiperKeyRegion(1),), 128),
        ((SparsePiperKeyRegion(2),), 129),
        ((SparsePiperKeyRegion(65536), SparsePiperKeyRegion(1)), 128),
    ],
)
def test_region_partition_rejects_invalid_storage(regions, sequence):
    q = torch.empty((1, sequence, 1, 64))
    with pytest.raises(ValueError, match="key_regions"):
        SparsePiperAttention((0.5,))(q, q, q, key_regions=regions)


@pytest.mark.parametrize(
    ("num_blocks", "sparse", "error"),
    [(0, True, ValueError), (True, True, TypeError), (2, 1, TypeError)],
)
def test_region_configuration_validation(num_blocks, sparse, error):
    with pytest.raises(error, match="region"):
        SparsePiperKeyRegion(num_blocks, sparse=sparse)


def test_exactly_one_layout_is_required():
    q = torch.empty((1, 128, 1, 64))
    attention = SparsePiperAttention((0.5,))
    with pytest.raises(ValueError, match="exactly one"):
        attention(q, q, q)
    with pytest.raises(ValueError, match="exactly one"):
        attention(q, q, q, sparse_key_blocks=2, key_regions=(SparsePiperKeyRegion(2),))


def test_region_compile_preserves_runtime_ratio_profile(device):
    torch._dynamo.reset()
    regions = (
        SparsePiperKeyRegion(3),
        SparsePiperKeyRegion(1, sparse=False),
        SparsePiperKeyRegion(3),
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = SparsePiperAttention((0.2, 0.5))

        def forward(self, q, k, v):
            return self.attention(q, k, v, key_regions=regions)

    model = Model()
    capture = TargetCapturePass()
    compiled = torch.compile(model, fullgraph=True, options={"post_grad_custom_pre_pass": capture})
    q = torch.randn((1, 448, 2, 64), device=device)
    k, v = torch.randn_like(q), torch.randn_like(q)
    for ratios in ((0.2, 0.5), (0.8, 1.0)):
        model.attention = SparsePiperAttention(ratios)
        torch.testing.assert_close(compiled(q, k, v), model(q, k, v), atol=0, rtol=0)
    assert capture.calls == 1


@pytest.mark.gpu
def test_region_graph_replay_uses_updated_forced_ids_without_readbacks(monkeypatch):
    if (
        not torch.cuda.is_available()
        or _backend.select_attention_backend(torch.empty((), device="cuda")) is None
    ):
        pytest.skip("requires native sparse Piper")
    attention = SparsePiperAttention((0.5,))
    regions = (
        SparsePiperKeyRegion(2),
        SparsePiperKeyRegion(1, sparse=False),
        SparsePiperKeyRegion(2),
    )
    q = torch.zeros((1, 320, 1, 64), device="cuda")
    v = (
        torch.arange(5, device="cuda", dtype=q.dtype)
        .repeat_interleave(64)[None, :, None, None]
        .expand_as(q)
        .contiguous()
    )
    ids = torch.tensor([[0, 3]] * 5, dtype=torch.int32, device="cuda")
    counts = torch.full((5,), 2, dtype=torch.int32, device="cuda")

    def run():
        return attention(q, q, v, key_regions=regions, force_tile_ids=ids, force_tile_counts=counts)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:
        for name in ("cpu", "tolist", "item"):
            patch.setattr(torch.Tensor, name, Mock(side_effect=AssertionError("host readback")))
        with torch.cuda.graph(graph, stream=stream):
            result = run()
    for first, last in ((1, 4), (0, 3)):
        ids[:, 0] = first
        ids[:, 1] = last
        graph.replay()
        expected = torch.full_like(result, (first + 2 + last) / 3)
        torch.testing.assert_close(result, expected, atol=0.01, rtol=0)
