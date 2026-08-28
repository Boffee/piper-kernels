"""Exact packed DSA routing tests."""

import pytest
import torch

from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention.dsa import (
    packed_dsa_routes_from_layout,
    packed_dsa_routes_from_summaries,
)


def _layout(
    keep_values: tuple[int, ...],
    key_blocks: int,
    device: torch.device | str = "cpu",
):
    ratios = tuple(value / key_blocks for value in keep_values)
    return _resolve_route_layout(
        _normalize_head_keep_ratios(ratios),
        key_blocks,
        torch.device(device),
    )


def test_packed_routes_store_only_active_uint16_indices() -> None:
    generator = torch.Generator().manual_seed(53)
    query = torch.randn((1, 3, 2, 64, 128), generator=generator)
    key = torch.randn((1, 3, 5, 64, 128), generator=generator)
    layout = _layout((1, 3, 2), 5)

    routes = packed_dsa_routes_from_layout(query, key, layout)

    assert routes.indices.shape == (1, 2, 6)
    assert routes.indices.dtype is torch.uint16
    assert routes.head_offsets.tolist() == [0, 1, 4, 6]
    assert bool((routes.indices.to(torch.int32) < 5).all())


def test_exact_score_ties_prefer_lower_key_index() -> None:
    query = torch.zeros((1, 1, 2, 64, 128))
    key = torch.zeros((1, 1, 4, 64, 128))
    layout = _layout((2,), 4)

    routes = packed_dsa_routes_from_layout(query, key, layout)

    assert routes.indices.tolist() == [[[0, 1], [0, 1]]]


def test_existing_summaries_select_the_same_routes_as_block_inputs() -> None:
    generator = torch.Generator().manual_seed(55)
    query = torch.randn((1, 3, 4, 64, 128), generator=generator)
    key = torch.randn((1, 3, 6, 64, 128), generator=generator)
    layout = _layout((1, 4, 2), 6)
    query_float = query.float()
    key_float = key.float()
    query_summary = query_float.amax(dim=3) + query_float.amin(dim=3)
    key_max = key_float.amax(dim=3)
    key_min = key_float.amin(dim=3)

    expected = packed_dsa_routes_from_layout(query, key, layout)
    actual = packed_dsa_routes_from_summaries(query_summary, key_max, key_min, layout)

    assert torch.equal(actual.indices, expected.indices)


def test_existing_summaries_accept_sparse_prefix_views() -> None:
    generator = torch.Generator().manual_seed(63)
    query_summary = torch.randn((1, 2, 3, 128), generator=generator)
    full_key_max = torch.randn((1, 2, 6, 128), generator=generator)
    full_key_min = torch.randn((1, 2, 6, 128), generator=generator)
    key_max = full_key_max[:, :, :4]
    key_min = full_key_min[:, :, :4]
    layout = _layout((2, 3), 4)

    expected = packed_dsa_routes_from_summaries(
        query_summary,
        key_max.contiguous(),
        key_min.contiguous(),
        layout,
    )
    actual = packed_dsa_routes_from_summaries(query_summary, key_max, key_min, layout)

    assert not key_max.is_contiguous()
    assert torch.equal(actual.indices, expected.indices)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_packed_routes_match_the_portable_exact_policy() -> None:
    generator = torch.Generator().manual_seed(54)
    query = torch.randn((1, 3, 5, 64, 128), dtype=torch.bfloat16, generator=generator)
    key = torch.randn((1, 3, 7, 64, 128), dtype=torch.bfloat16, generator=generator)
    cpu_layout = _layout((1, 4, 6), 7)
    cuda_layout = _layout((1, 4, 6), 7, "cuda")

    expected = packed_dsa_routes_from_layout(query, key, cpu_layout)
    actual = packed_dsa_routes_from_layout(query.cuda(), key.cuda(), cuda_layout)

    torch.testing.assert_close(
        actual.indices.cpu().to(torch.int32),
        expected.indices.to(torch.int32),
        atol=0,
        rtol=0,
    )
