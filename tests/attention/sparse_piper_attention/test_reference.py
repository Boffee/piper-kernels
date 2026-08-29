"""Portable Sparse Piper reference tests."""

import torch

from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention.dsa import (
    PackedDsaRoutes,
    packed_dsa_routes_from_sequences,
)
from piper_kernels.attention.sparse_piper_attention.reference import (
    reference_exact_sparse_attention,
    reference_sparse_piper_attention,
)


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PackedDsaRoutes,
]:
    generator = torch.Generator().manual_seed(55)
    shape = (1, 3 * 64, 2, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    layout = _resolve_route_layout(
        _normalize_head_keep_ratios((0.5, 1.0)),
        2,
        torch.device("cpu"),
    )
    query_sequence = query.transpose(1, 2)
    sparse_key = key[:, :128].transpose(1, 2)
    routes = packed_dsa_routes_from_sequences(query_sequence, sparse_key, layout)
    return query, key, value, routes


def test_quantized_reference_stays_within_the_selected_exact_quality_gate() -> None:
    query, key, value, routes = _inputs()

    actual = reference_sparse_piper_attention(
        query,
        key,
        value,
        routes,
        sparse_key_blocks=2,
        scale=128**-0.5,
    )
    exact = reference_exact_sparse_attention(
        query,
        key,
        value,
        routes,
        sparse_key_blocks=2,
        scale=128**-0.5,
    )

    relative_l2 = (actual.float() - exact.float()).norm() / exact.float().norm()
    assert relative_l2 < 0.025


def test_quantized_reference_restores_constant_value_exactly() -> None:
    query, key, value, routes = _inputs()
    value_row = value[:, :1]
    value = value_row.expand_as(value).contiguous()

    actual = reference_sparse_piper_attention(
        query,
        key,
        value,
        routes,
        sparse_key_blocks=2,
        scale=128**-0.5,
    )

    torch.testing.assert_close(actual, value, atol=0, rtol=0)
