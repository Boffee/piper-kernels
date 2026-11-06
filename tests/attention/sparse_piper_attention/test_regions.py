"""Independent regional budgets and one shared softmax over interleaved keys."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
import torch
from _compile_capture import TargetCapturePass

from piper_kernels import SparsePiperAttention, SparsePiperKeyLayout, SparsePiperKeyRegion
from piper_kernels.attention.sparse_piper_attention import _backend, _routing
from piper_kernels.attention.sparse_piper_attention._budget import _normalize_head_keep_ratios
from piper_kernels.attention.sparse_piper_attention._regions import resolve_region_route_layout
from piper_kernels.attention.sparse_piper_attention._routes import PackedRouteBuilder
from piper_kernels.attention.sparse_piper_attention._routing_modes import routing_mode_from_name


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
        SparsePiperKeyLayout(regions),
        device,
    )


def test_budget_rounding_is_independent_per_region():
    attention = SparsePiperAttention((0.5, 0.5))
    regions = (
        SparsePiperKeyRegion(3),
        SparsePiperKeyRegion(2, sparse=False),
        SparsePiperKeyRegion(3),
    )
    assert attention.resolve_region_keep_blocks(SparsePiperKeyLayout(regions)) == (
        (2, 1),
        (2, 2),
        (2, 1),
    )
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
    budgets = SparsePiperAttention((0.2, 0.5)).resolve_region_keep_blocks(
        SparsePiperKeyLayout(regions)
    )
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


@pytest.mark.parametrize("device", [pytest.param("cuda", marks=pytest.mark.gpu)], indirect=True)
def test_many_regions_do_not_exceed_native_grid_limits(device):
    regions = (SparsePiperKeyRegion(2),) * 32768
    layout = _layout((0.5, 0.5), regions, device)
    builder = PackedRouteBuilder(
        layout, batch=1, heads=2, query_blocks=1, sparse_key_blocks=65536, device=device
    )
    scores = torch.arange(65536, device=device, dtype=torch.float32).expand(1, 2, 1, -1)
    builder.write(scores, query_block_offset=0)
    expected = torch.arange(1, 65536, 2).repeat(2)
    torch.testing.assert_close(builder.routes.indices.cpu().long().flatten(), expected)


@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_compact_summary_routes_match_independent_region_selection(device, routing):
    regions = (
        SparsePiperKeyRegion(2, sparse=False),
        SparsePiperKeyRegion(5),
        SparsePiperKeyRegion(1),  # Rounding retains this region without scoring it.
        SparsePiperKeyRegion(3, sparse=False),
        SparsePiperKeyRegion(513),  # Selection crosses a radix tile boundary.
        SparsePiperKeyRegion(2, sparse=False),
    )
    ratios = (0.2, 0.5, 1.0, 0.75)
    budgets = SparsePiperAttention(ratios).resolve_region_keep_blocks(SparsePiperKeyLayout(regions))
    generator = torch.Generator().manual_seed(298)
    # Integer-valued FP32 summaries make the score oracle exact on both devices.
    # 385 query blocks also crosses the routing chunk boundary.
    query = torch.randint(-8, 9, (2, 4, 385, 64), generator=generator).float()
    primary = torch.randint(-8, 9, (2, 2, 526, 64), generator=generator).float()
    auxiliary = (
        primary - torch.randint(0, 9, primary.shape, generator=generator)
        if routing == "minmax"
        else primary[:, :, :0]
    )
    scores = query @ primary.repeat_interleave(2, dim=1).transpose(-1, -2)
    if routing == "minmax":
        scores = torch.maximum(
            scores, query @ auxiliary.repeat_interleave(2, dim=1).transpose(-1, -2)
        )
    expected_heads = []
    for head in range(4):
        start = 0
        selected_regions = []
        for region, counts in zip(regions, budgets, strict=True):
            selected = scores[:, head, :, start : start + region.num_blocks].argsort(
                dim=-1, descending=True, stable=True
            )[..., : counts[head]]
            selected_regions.append(start + selected.sort(dim=-1).values)
            start += region.num_blocks
        expected_heads.append(torch.cat(selected_regions, dim=-1))
    expected = torch.cat(expected_heads, dim=-1)
    actual = _routing.packed_routes_from_summaries(
        query.to(device),
        primary.to(device),
        auxiliary.to(device),
        _layout(ratios, regions, device),
        routing_mode_from_name(routing),
    )
    torch.testing.assert_close(actual.indices.cpu().long(), expected, atol=0, rtol=0)


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
        key_layout=SparsePiperKeyLayout(regions),
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
    key_layout = SparsePiperKeyLayout(regions)
    SparsePiperAttention((0.3, 0.5))(q, q, q, key_layout=key_layout)
    assert set(widths) == {3}
    monkeypatch.setattr(
        _routing, "routing_scores", Mock(side_effect=AssertionError("full keep scored"))
    )
    full = SparsePiperAttention((1.0, 1.0))
    torch.testing.assert_close(
        full(q, q, q, key_layout=key_layout), full(q, q, q, sparse_key_blocks=6), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    ("regions", "sequence"),
    [
        ((SparsePiperKeyRegion(1),), 128),
        ((SparsePiperKeyRegion(3),), 128),
        ((SparsePiperKeyRegion(2),), 129),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_region_partition_rejects_invalid_storage(regions, sequence, device):
    q = torch.empty((1, sequence, 1, 64), device=device)
    with pytest.raises(ValueError, match="key_layout"):
        SparsePiperAttention((0.5,))(q, q, q, key_layout=SparsePiperKeyLayout(regions))


@pytest.mark.parametrize(
    ("num_blocks", "sparse", "error"),
    [(0, True, ValueError), (True, True, TypeError), (2, 1, TypeError)],
)
def test_region_configuration_validation(num_blocks, sparse, error):
    with pytest.raises(error, match="region"):
        SparsePiperKeyRegion(num_blocks, sparse=sparse)


@pytest.mark.parametrize(
    ("regions", "error"),
    [
        ((), ValueError),
        ([SparsePiperKeyRegion(1)], TypeError),
        ((1,), TypeError),
        ((SparsePiperKeyRegion(65536), SparsePiperKeyRegion(1)), ValueError),
    ],
)
def test_key_layout_validates_configuration_at_construction(regions, error):
    with pytest.raises(error, match="layout"):
        SparsePiperKeyLayout(regions)


def test_key_layout_is_immutable():
    regions = (SparsePiperKeyRegion(3), SparsePiperKeyRegion(2, sparse=False))
    layout = SparsePiperKeyLayout(regions)
    assert layout.regions == regions
    assert layout.num_blocks == 5
    with pytest.raises(FrozenInstanceError):
        layout.regions = (SparsePiperKeyRegion(5),)
    with pytest.raises(FrozenInstanceError):
        layout.regions[0].num_blocks = 4


def test_region_entry_points_require_a_layout_object():
    regions = (SparsePiperKeyRegion(2),)
    attention = SparsePiperAttention((0.5,))
    with pytest.raises(TypeError, match="SparsePiperKeyLayout"):
        attention.resolve_region_keep_blocks(regions)
    q = torch.empty((1, 128, 1, 64))
    with pytest.raises(TypeError, match="SparsePiperKeyLayout"):
        attention(q, q, q, key_layout=regions)


def test_exactly_one_layout_is_required():
    q = torch.empty((1, 128, 1, 64))
    attention = SparsePiperAttention((0.5,))
    with pytest.raises(ValueError, match="exactly one"):
        attention(q, q, q)
    with pytest.raises(ValueError, match="exactly one"):
        attention(
            q,
            q,
            q,
            sparse_key_blocks=2,
            key_layout=SparsePiperKeyLayout((SparsePiperKeyRegion(2),)),
        )


def test_compiled_layout_rechecks_total_block_count():
    torch._dynamo.reset()
    capture = TargetCapturePass()
    compiled = torch.compile(
        SparsePiperAttention((0.5,)),
        fullgraph=True,
        options={"post_grad_custom_pre_pass": capture},
    )
    q = torch.zeros((1, 128, 1, 64))
    compiled(q, q, q, key_layout=SparsePiperKeyLayout((SparsePiperKeyRegion(2),)))
    for num_blocks in (1, 3):
        with pytest.raises(ValueError, match="key_layout must cover exactly"):
            compiled(q, q, q, key_layout=SparsePiperKeyLayout((SparsePiperKeyRegion(num_blocks),)))
    assert capture.calls == 1


@pytest.mark.parametrize(
    ("layout_source", "padded_forcing"), [("argument", False), ("attribute", True)]
)
def test_region_compile_reuses_dynamic_lengths_and_layouts(device, layout_source, padded_forcing):
    torch._dynamo.reset()
    layouts = (
        SparsePiperKeyLayout((SparsePiperKeyRegion(12),)),
        SparsePiperKeyLayout((SparsePiperKeyRegion(6), SparsePiperKeyRegion(6))),
        SparsePiperKeyLayout((SparsePiperKeyRegion(6), SparsePiperKeyRegion(7))),
        SparsePiperKeyLayout((SparsePiperKeyRegion(5), SparsePiperKeyRegion(9))),
        SparsePiperKeyLayout((SparsePiperKeyRegion(3), SparsePiperKeyRegion(9))),
        SparsePiperKeyLayout((SparsePiperKeyRegion(3, sparse=False), SparsePiperKeyRegion(9))),
        SparsePiperKeyLayout((SparsePiperKeyRegion(3, sparse=False), SparsePiperKeyRegion(10))),
        SparsePiperKeyLayout((SparsePiperKeyRegion(16, sparse=False),)),
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = SparsePiperAttention((0.25, 0.5))
            self.key_layout = layouts[0]

        def forward(self, q, k, v, key_layout, lengths, ids, counts):
            return self.attention(
                q,
                k,
                v,
                key_layout=self.key_layout if key_layout is None else key_layout,
                block_lengths=lengths,
                force_tile_ids=ids,
                force_tile_counts=counts,
            )

    model = Model()
    capture = TargetCapturePass()
    compiled = torch.compile(
        model,
        fullgraph=True,
        options={
            "post_grad_custom_pre_pass": capture,
            "triton.cudagraphs": device.type == "cuda",
        },
    )
    cases = [(layout, (0.25, 0.5)) for layout in layouts]
    # Returning to the initial layout also changes the runtime ratio profile.
    cases.append((layouts[0], (0.75, 1.0)))
    for key_layout, ratios in cases:
        num_blocks = key_layout.num_blocks
        q = torch.zeros((1, num_blocks * 64, 2, 64), device=device)
        k = torch.zeros((1, num_blocks * 64, 1, 64), device=device)
        v = (
            torch.arange(num_blocks, device=device, dtype=q.dtype)
            .repeat_interleave(64)[None, :, None, None]
            .expand_as(k)
            .contiguous()
        )
        for tensor in (q, k, v):
            torch._dynamo.mark_dynamic(tensor, 1)
        lengths = ids = counts = None
        if padded_forcing:
            lengths = torch.full((num_blocks,), 64, device=device, dtype=torch.int32)
            ids = torch.full((num_blocks, 1), num_blocks - 1, device=device, dtype=torch.int32)
            counts = torch.ones(num_blocks, device=device, dtype=torch.int32)
            for tensor in (lengths, ids, counts):
                torch._dynamo.mark_dynamic(tensor, 0)
        model.attention = SparsePiperAttention(ratios)
        model.key_layout = key_layout
        argument = key_layout if layout_source == "argument" else None
        torch.compiler.cudagraph_mark_step_begin()
        actual = compiled(q, k, v, argument, lengths, ids, counts).clone()
        budgets = model.attention.resolve_region_keep_blocks(key_layout)
        expected = torch.empty_like(q)
        for head in range(2):
            start = 0
            selected = []
            for region, head_counts in zip(key_layout.regions, budgets, strict=True):
                selected.extend(range(start, start + head_counts[head]))
                start += region.num_blocks
            if padded_forcing:
                # Force the final physical block in the final region's budget.
                selected[-1] = num_blocks - 1
            expected[:, :, head] = sum(selected) / len(selected)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0)
        torch.testing.assert_close(
            actual, model(q, k, v, argument, lengths, ids, counts), atol=0, rtol=0
        )
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
    key_layout = SparsePiperKeyLayout(regions)
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
        return attention(
            q, q, v, key_layout=key_layout, force_tile_ids=ids, force_tile_counts=counts
        )

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
