"""Portable Sparse Piper reference tests."""

import torch

from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._routes import (
    PackedRoutes,
)
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MINMAX_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention.reference import (
    reference_exact_sparse_attention,
    reference_sparse_piper_attention,
)


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PackedRoutes,
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
    routes = packed_routes_from_sequences(
        query_sequence,
        sparse_key,
        layout,
        _MINMAX_ROUTING,
    )
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


def test_quantized_reference_restores_large_constant_value() -> None:
    layout = _resolve_route_layout(_normalize_head_keep_ratios((1.0,)), 2, torch.device("cpu"))
    query = torch.zeros((1, 128, 1, 64), dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    routes = packed_routes_from_sequences(
        query.transpose(1, 2),
        key.transpose(1, 2),
        layout,
        _MINMAX_ROUTING,
    )
    for constant in (2.0**100, -(2.0**100)):
        value = torch.full_like(query, constant)
        actual = reference_sparse_piper_attention(
            query,
            key,
            value,
            routes,
            sparse_key_blocks=2,
            scale=64**-0.5,
        )
        torch.testing.assert_close(actual, value, atol=0, rtol=0)


def test_query_offset_reproduces_the_whole_sequence_reference_rows() -> None:
    generator = torch.Generator().manual_seed(56)
    query, key, value = (
        torch.randn((1, 5 * 64, 2, 128), dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    block_lengths = torch.tensor([64, 17, 64, 40, 64], dtype=torch.int32)
    layout = _resolve_route_layout(
        _normalize_head_keep_ratios((0.5, 1.0)),
        4,
        torch.device("cpu"),
    )
    options = {"sparse_key_blocks": 4, "scale": 128**-0.5, "block_lengths": block_lengths}

    def run(start: int, end: int) -> torch.Tensor:
        rows = query[:, start:end]
        routes = packed_routes_from_sequences(
            rows.transpose(1, 2),
            key[:, :256].transpose(1, 2),
            layout,
            _MINMAX_ROUTING,
            block_lengths,
            global_block_offset=start // 64,
        )
        # The routed scope ends inside the second slice.
        return reference_sparse_piper_attention(
            rows, key, value, routes, sparse_query_blocks=3, global_row_offset=start, **options
        )

    expected = run(0, 320)
    actual = torch.cat((run(0, 128), run(128, 320)), dim=1)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
