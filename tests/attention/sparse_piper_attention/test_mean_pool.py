"""Mean-pool packed routing tests."""

import torch

from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention.mean_pool import (
    _sequence_block_means,
    packed_mean_pool_routes_from_sequences,
    packed_mean_pool_routes_from_summaries,
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


def test_summary_routes_use_per_head_mean_dot_products() -> None:
    query_mean = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, -1.0]]]],
        dtype=torch.float32,
    )
    key_mean = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]],
                [[1.0, 0.0], [0.0, 1.0], [-2.0, -2.0]],
            ]
        ],
        dtype=torch.float32,
    )
    layout = _layout((1, 2), 3)

    routes = packed_mean_pool_routes_from_summaries(query_mean, key_mean, layout)

    assert routes.head_offsets.tolist() == [0, 1, 3]
    assert routes.indices.tolist() == [[[2, 1, 2], [2, 0, 2]]]


def test_exact_score_ties_prefer_lower_key_index() -> None:
    query_mean = torch.zeros((1, 1, 2, 128), dtype=torch.float32)
    key_mean = torch.zeros((1, 1, 4, 128), dtype=torch.float32)

    routes = packed_mean_pool_routes_from_summaries(
        query_mean,
        key_mean,
        _layout((2,), 4),
    )

    assert routes.indices.tolist() == [[[0, 1], [0, 1]]]


def test_sequence_routes_match_explicit_compact_block_means() -> None:
    generator = torch.Generator().manual_seed(71)
    query = torch.randn((1, 2, 2 * 64 + 17, 128), dtype=torch.bfloat16, generator=generator)
    key = torch.randn((1, 2, 3 * 64, 128), dtype=torch.bfloat16, generator=generator)
    layout = _layout((1, 2), 3)

    actual = packed_mean_pool_routes_from_sequences(query, key, layout)
    query_blocks = query[:, :, : 2 * 64].unflatten(2, (2, 64)).float()
    key_blocks = key.unflatten(2, (3, 64)).float()
    query_mean = torch.cat(
        (query_blocks.mean(dim=3), query[:, :, 2 * 64 :].float().mean(dim=2, keepdim=True)),
        dim=2,
    )
    key_mean = key_blocks.mean(dim=3)
    expected = packed_mean_pool_routes_from_summaries(query_mean, key_mean, layout)

    assert torch.equal(actual.indices, expected.indices)


def test_ragged_query_mean_uses_only_logical_rows() -> None:
    query = torch.zeros((1, 1, 65, 128), dtype=torch.bfloat16)
    query[:, :, :64] = 2
    query[:, :, 64] = 7
    key = torch.zeros((1, 1, 64, 128), dtype=torch.bfloat16)

    query_mean = _sequence_block_means(query)
    routes = packed_mean_pool_routes_from_sequences(query, key, _layout((1,), 1))

    torch.testing.assert_close(query_mean[:, :, 0], torch.full_like(query_mean[:, :, 0], 2))
    torch.testing.assert_close(query_mean[:, :, 1], torch.full_like(query_mean[:, :, 1], 7))
    assert routes.indices.shape == (1, 2, 1)
