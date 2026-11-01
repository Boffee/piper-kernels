"""Fixed-budget forced selection, dynamic tensor inputs, and shared scores."""

import pytest
import torch

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._budget import _ResolvedRouteLayout
from piper_kernels.attention.sparse_piper_attention._force_tiles import validate_force_tiles
from piper_kernels.attention.sparse_piper_attention._routes import (
    PackedRouteAndCoarseBuilder,
    PackedRouteBuilder,
)
from piper_kernels.attention.sparse_piper_attention.coarse import coarse_attention


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    if request.param == "cuda" and (
        not torch.cuda.is_available()
        or _backend.select_route_selector(torch.empty(0, device="cuda")) is None
    ):
        pytest.skip("requires native sparse routing")
    return torch.device("cuda", 0) if request.param == "cuda" else torch.device("cpu")


def _layout(counts, device):
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return _ResolvedRouteLayout(
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
        offsets[-1],
    )


def _oracle(scores, ids, counts, budgets):
    batch, heads, queries, keys = scores.shape
    result = torch.empty((batch, queries, sum(budgets)), dtype=torch.uint16)
    for b in range(batch):
        for q in range(queries):
            forced = set(ids[q, : counts[q]].tolist())
            cursor = 0
            for h in range(heads):
                others = sorted(
                    (k for k in range(keys) if k not in forced),
                    key=lambda k: (-float(scores[b, h, q, k]), k),
                )
                selected = sorted(forced | set(others[: budgets[h] - len(forced)]))
                result[b, q, cursor : cursor + budgets[h]] = torch.tensor(
                    selected, dtype=torch.uint16
                )
                cursor += budgets[h]
    return result


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("num_key_blocks", [65, 513])
def test_forced_routes_keep_exact_budget_and_scores(device, ties, num_key_blocks):
    budgets = [3, 17, num_key_blocks]
    scores = torch.randint(
        -5, 6, (2, 3, 6, num_key_blocks), generator=torch.Generator().manual_seed(83)
    ).float()
    if ties:
        scores.zero_()
    # Cover both membership-word boundaries and the selector's 512-key tile boundary.
    last_key = num_key_blocks - 1
    ids = torch.tensor(
        [[last_key, 31, 32, -999], [32, 0, last_key, -999], [31, 32, last_key, -999]] * 2,
        dtype=torch.int32,
    )
    counts = torch.tensor([0, 1, 3, 2, 3, 0], dtype=torch.int32)
    expected = _oracle(scores, ids, counts, budgets)
    builder = PackedRouteBuilder(
        _layout(budgets, device),
        batch=2,
        heads=3,
        query_blocks=6,
        sparse_key_blocks=num_key_blocks,
        device=device,
        force_tile_ids=ids.to(device),
        force_tile_counts=counts.to(device),
    )
    actual_scores = scores.to(device)
    for start, stop in ((0, 2), (2, 6)):
        builder.write(actual_scores[:, :, start:stop], query_block_offset=start)
    torch.testing.assert_close(builder.routes.indices.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_scores.cpu(), scores, atol=0, rtol=0)


def test_forced_routes_leave_coarse_attention_unchanged(device):
    scores = torch.randn(1, 2, 4, 9, device=device)
    values = torch.randn(1, 2, 9, 64, device=device)
    ids = torch.tensor([[7, 6]] * 4, device=device, dtype=torch.int32)
    counts = torch.full((4,), 2, dtype=torch.int32, device=device)
    expected = coarse_attention(scores.clone(), values)
    builder = PackedRouteAndCoarseBuilder(
        _layout([2, 4], device),
        values,
        batch=1,
        heads=2,
        query_blocks=4,
        sparse_key_blocks=8,
        device=device,
        force_tile_ids=ids,
        force_tile_counts=counts,
    )
    builder.write(scores, query_block_offset=0)
    result = builder.finish()
    torch.testing.assert_close(result.coarse_output, expected, atol=0, rtol=0)
    assert set(result.routes.indices[0, 0, :2].cpu().tolist()) == {6, 7}


@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_public_forced_routes_and_dense_query_suffix(device, routing):
    q = torch.zeros(1, 9 * 64, 2, 64, device=device, dtype=torch.bfloat16)
    v = (
        torch.arange(9, device=device)
        .repeat_interleave(64)[None, :, None, None]
        .expand_as(q)
        .contiguous()
        .to(q.dtype)
    )
    ids = torch.full((9, 1), 7, dtype=torch.int32, device=device)
    counts = torch.ones(9, dtype=torch.int32, device=device)
    attention = SparsePiperAttention((0.25, 0.5), routing=routing)
    result = attention(
        q,
        q,
        v,
        sparse_key_blocks=8,
        sparse_query_blocks=8,
        force_tile_ids=ids,
        force_tile_counts=counts,
    )
    expected = torch.tensor([(7 + 0 + 8) / 3, (7 + 0 + 1 + 2 + 8) / 5], device=device)
    torch.testing.assert_close(
        result[0, : 8 * 64, :, 0].float(), expected.expand(8 * 64, 2), atol=0.04, rtol=0
    )
    torch.testing.assert_close(
        result[:, 8 * 64 :], torch.full_like(result[:, 8 * 64 :], 4), atol=0.04, rtol=0
    )


def test_changing_forced_data_reuses_one_compiled_graph(device):
    attention = SparsePiperAttention((0.25,))
    q = torch.zeros(1, 8 * 64, 1, 64, device=device)
    v = (
        torch.arange(8, device=device)
        .repeat_interleave(64)[None, :, None, None]
        .expand_as(q)
        .contiguous()
        .float()
    )
    ids = torch.zeros((8, 2), dtype=torch.int32, device=device)
    counts = torch.zeros(8, dtype=torch.int32, device=device)
    graphs = []

    def backend(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(attention, backend=backend, fullgraph=True)
    for active, first, second in ((0, 7, 6), (1, 7, 6), (2, 7, 6), (1, 4, 3)):
        ids[:, 0], ids[:, 1] = first, second
        counts.fill_(active)
        actual = compiled(
            q, q, v, sparse_key_blocks=8, force_tile_ids=ids, force_tile_counts=counts
        )
        expected = attention(
            q, q, v, sparse_key_blocks=8, force_tile_ids=ids, force_tile_counts=counts
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(graphs) == 1


@pytest.mark.parametrize("routing", ["mean", "minmax"])
@pytest.mark.parametrize("ratio", [0.34, 1.0])
def test_empty_forces_and_full_keep_match_baseline_with_gqa(device, routing, ratio):
    q = torch.randn(1, 193, 2, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 193, 1, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ids = torch.full((4, 2), -999, dtype=torch.int32, device=device)
    counts = torch.zeros(4, dtype=torch.int32, device=device)
    if ratio == 1:
        ids[:, 0] = 2
        counts.fill_(1)
    attention = SparsePiperAttention((ratio, ratio), routing=routing)
    expected = attention(q, k, v, sparse_key_blocks=3)
    actual = attention(q, k, v, sparse_key_blocks=3, force_tile_ids=ids, force_tile_counts=counts)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mistake", ["missing", "dtype", "shape", "stride"])
def test_forced_metadata_is_validated(mistake):
    ids = torch.zeros((4, 2), dtype=torch.int32)
    counts = torch.ones(4, dtype=torch.int32)
    if mistake == "missing":
        counts = None
    elif mistake == "dtype":
        ids = ids.long()
    elif mistake == "shape":
        ids = ids[:3]
    else:
        ids = torch.zeros((4, 4), dtype=torch.int32)[:, ::2]
    with pytest.raises(ValueError, match="force"):
        validate_force_tiles(ids, counts, query_blocks=4, device=torch.device("cpu"))


@pytest.mark.gpu
def test_graph_replay_reads_mutated_forced_ids_and_counts():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    device = torch.device("cuda", 0)
    scores = torch.arange(65, device=device).float().expand(1, 2, 4, 65).contiguous()
    ids = torch.tensor([[0, 31, 32]] * 4, device=device, dtype=torch.int32)
    counts = torch.ones(4, device=device, dtype=torch.int32)
    layout = _layout([3, 17], device)

    def run():
        builder = PackedRouteBuilder(
            layout,
            batch=1,
            heads=2,
            query_blocks=4,
            sparse_key_blocks=65,
            device=device,
            force_tile_ids=ids,
            force_tile_counts=counts,
        )
        builder.write(scores, query_block_offset=0)
        return builder.routes.indices

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    from piper_kernels.attention.sparse_piper_attention._routes_triton import (  # noqa: PLC0415
        _scatter_force_tile_bits,
        _tiled_radix_select_packed_routes_kernel,
    )

    kernels = (_scatter_force_tile_bits, _tiled_radix_select_packed_routes_kernel)
    specializations = [sum(len(c[0]) for c in kernel.device_caches.values()) for kernel in kernels]
    for active, first in ((0, 1), (3, 2), (1, 64)):
        ids[:, 0] = first
        counts.fill_(active)
        graph.replay()
        expected = _oracle(scores.cpu(), ids.cpu(), counts.cpu(), [3, 17])
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(run().cpu(), expected, atol=0, rtol=0)
    assert specializations == [
        sum(len(c[0]) for c in kernel.device_caches.values()) for kernel in kernels
    ]
