"""Full physical budgets bypass selection while retaining coarse computation."""

import pytest
import torch

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention import _backend, _routes, _routing
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
    _ResolvedRouteLayout,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)

requires_native = pytest.mark.skipif(
    not torch.cuda.is_available()
    or _backend.select_attention_backend(torch.empty(0, device="cuda")) is None,
    reason="requires native sparse attention",
)


def _unexpected(*args, **kwargs):
    raise AssertionError("full keep must not run routing selection")


@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("ratio", [0.999, 1.0])
def test_full_keep_skips_summaries_and_scores(monkeypatch, routing_mode, head_dim, ratio):
    query = torch.randn(2, 3, 193, head_dim)
    key = torch.randn(2, 3, 192, head_dim)
    # A rounded physical budget can be full even if its semantic ratio is not 1.
    layout = _resolve_route_layout(_normalize_head_keep_ratios([ratio] * 3), 3, query.device)
    monkeypatch.setattr(_routing, "sequence_block_summaries", _unexpected)
    monkeypatch.setattr(_routing, "score_chunks", _unexpected)
    result = _routing.packed_routes_from_sequences(query, key, layout, routing_mode)
    expected = torch.tensor([0, 1, 2] * 3, dtype=torch.uint16).expand(2, 4, 9)
    torch.testing.assert_close(result.indices, expected, atol=0, rtol=0)


@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_full_keep_from_summaries_skips_score_chunks(monkeypatch, routing_mode, head_dim):
    query = torch.randn(1, 2, 4, head_dim)
    key = torch.randn(1, 2, 3, head_dim)
    auxiliary = key[:, :, :0] if routing_mode == _MEAN_ROUTING else torch.randn_like(key)
    layout = _resolve_route_layout(_normalize_head_keep_ratios([1.0] * 2), 3, query.device)
    monkeypatch.setattr(_routing, "score_chunks", _unexpected)
    result = _routing.packed_routes_from_summaries(query, key, auxiliary, layout, routing_mode)
    expected = torch.tensor([0, 1, 2] * 2, dtype=torch.uint16).expand(1, 4, 6)
    torch.testing.assert_close(result.indices, expected, atol=0, rtol=0)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_full_keep_preserves_coarse_scores_and_output(monkeypatch, head_dim):
    query = torch.randn(1, 2, 4, head_dim)
    key = torch.randn(1, 2, 3, head_dim)
    value = torch.randn(1, 2, 3, head_dim)
    layout = _resolve_route_layout(_normalize_head_keep_ratios([1.0] * 2), 2, query.device)
    monkeypatch.setattr(_routes, "_select_portable_routes", _unexpected)
    result = _routing.packed_routes_and_coarse_from_summaries(
        query,
        key,
        key[:, :, :0],
        value,
        layout,
        sparse_key_blocks=2,
        coarse_scale=0.125,
        routing_mode=_MEAN_ROUTING,
    )
    expected = torch.softmax((query @ key.mT) * 0.125, dim=-1) @ value
    torch.testing.assert_close(result.coarse_output, expected)
    expected_routes = torch.tensor([0, 1] * 2, dtype=torch.uint16).expand(1, 4, 4)
    torch.testing.assert_close(result.routes.indices, expected_routes, atol=0, rtol=0)


def test_full_keep_keeps_input_validation():
    query = torch.randn(1, 2, 64, 128)
    layout = _resolve_route_layout(_normalize_head_keep_ratios([1.0] * 2), 1, query.device)
    with pytest.raises(ValueError, match="batch/head/feature"):
        _routing.packed_routes_from_sequences(query, query[:, :1], layout, _MEAN_ROUTING)


@pytest.mark.parametrize("blocks", [3, 513, 65536])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, requires_native])]
)
def test_canonical_routes_cover_heads_tail_and_uint16_limit(blocks, device):
    routes = torch.empty((2, 3, 2 * blocks), dtype=torch.uint16, device=device)
    expected = torch.arange(blocks, dtype=torch.int32).repeat(2).to(torch.uint16).expand(2, 3, -1)
    _backend.fill_full_keep_routes(routes, blocks)
    torch.testing.assert_close(routes.cpu(), expected, atol=0, rtol=0)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _backend.fill_full_keep_routes(routes, blocks)
        routes.zero_()
        graph.replay()
        torch.testing.assert_close(routes.cpu(), expected, atol=0, rtol=0)


@pytest.mark.gpu
@requires_native
@pytest.mark.parametrize("routing", ["minmax", "mean"])
@pytest.mark.parametrize("sequence", [193, 1797])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_full_keep_attention_matches_score_selected_routes(
    monkeypatch, routing, sequence, head_dim
):
    operands = [
        torch.randn(1, sequence, 2, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    attention = SparsePiperAttention((1.0, 1.0), routing=routing)
    with monkeypatch.context() as control:
        control.setattr(_ResolvedRouteLayout, "keeps_all_blocks", lambda *_: False)
        expected = attention(*operands, sparse_key_blocks=sequence // 64)
    monkeypatch.setattr(_routing, "sequence_block_summaries", _unexpected)
    monkeypatch.setattr(_routing, "score_chunks", _unexpected)
    actual = attention(*operands, sparse_key_blocks=sequence // 64)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
