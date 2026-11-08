"""Full-keep routes can be represented without a materialized list."""

import pytest
import torch

from piper_kernels.attention.sparse_piper_attention import _routing
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention.reference import (
    reference_sparse_piper_attention,
)


def _unexpected(*args, **kwargs):
    raise AssertionError("full keep must not run routing selection")


@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
def test_skip_dense_routing_skips_summaries_and_scores(monkeypatch, routing_mode):
    query = torch.randn(2, 3, 193, 64)
    key = torch.randn(2, 3, 192, 64)
    layout = _resolve_route_layout(_normalize_head_keep_ratios([0.999] * 3), 3, query.device)
    monkeypatch.setattr(_routing, "sequence_block_summaries", _unexpected)
    monkeypatch.setattr(_routing, "score_chunks", _unexpected)
    result = _routing.packed_routes_from_sequences(
        query, key, layout, routing_mode, skip_dense_routing=True
    )
    assert result.indices.shape == (2, 4, 0)


def test_skip_dense_routing_preserves_coarse_output():
    query = torch.randn(1, 2, 4, 64)
    key = torch.randn(1, 2, 3, 64)
    value = torch.randn(1, 2, 3, 64)
    layout = _resolve_route_layout(_normalize_head_keep_ratios([1.0] * 2), 2, query.device)
    result = _routing.packed_routes_and_coarse_from_summaries(
        query,
        key,
        key[:, :, :0],
        value,
        layout,
        sparse_key_blocks=2,
        coarse_scale=0.125,
        routing_mode=_MEAN_ROUTING,
        skip_dense_routing=True,
    )
    expected = torch.softmax((query @ key.mT) * 0.125, dim=-1) @ value
    torch.testing.assert_close(result.coarse_output, expected)
    assert result.routes.indices.shape == (1, 4, 0)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_skip_dense_routing_matches_portable_reference(head_dim):
    operands = [torch.randn(1, 193, 2, head_dim, dtype=torch.bfloat16) for _ in range(3)]
    query, key, _ = operands
    layout = _resolve_route_layout(_normalize_head_keep_ratios([1.0, 1.0]), 3, query.device)
    outputs = []
    for skip_dense_routing in (False, True):
        routes = _routing.packed_routes_from_sequences(
            query.transpose(1, 2),
            key.transpose(1, 2)[:, :, :192],
            layout,
            _MINMAX_ROUTING,
            skip_dense_routing=skip_dense_routing,
        )
        outputs.append(
            reference_sparse_piper_attention(
                *operands,
                routes,
                sparse_key_blocks=3,
                scale=head_dim**-0.5,
            )
        )
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
