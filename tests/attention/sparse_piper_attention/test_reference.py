"""Portable Sparse Piper reference tests."""

import torch

from piper_kernels.attention.sparse_piper_attention.dsa import (
    PackedDsaRoutes,
    packed_dsa_routes_from_plan,
    prepare_dsa_route_plan,
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
    plan = prepare_dsa_route_plan(
        torch.tensor([1, 2], dtype=torch.int32),
    )
    query_blocks = query.transpose(1, 2).unflatten(2, (3, 64))
    key_blocks = key[:, :128].transpose(1, 2).unflatten(2, (2, 64))
    routes = packed_dsa_routes_from_plan(query_blocks, key_blocks, plan)
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
