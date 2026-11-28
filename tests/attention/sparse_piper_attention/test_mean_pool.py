"""Mean-pool packed routing tests."""

import pytest
import torch

from piper_kernels import sparse_piper_coarse_residual
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention._routing import (
    PackedRouteAndCoarseBuilder,
    packed_routes_and_coarse_from_summaries,
    packed_routes_from_sequences,
    packed_routes_from_summaries,
    sequence_block_summaries,
)
from piper_kernels.attention.sparse_piper_attention.coarse import (
    _mean_pool_head_major_blocks,
)


def _sequence_block_means(sequence, block_lengths=None):
    return _mean_pool_head_major_blocks(sequence, block_lengths)


def packed_mean_pool_routes_from_sequences(query, key, layout, block_lengths=None):
    return packed_routes_from_sequences(query, key, layout, _MEAN_ROUTING, block_lengths)


def packed_mean_pool_routes_from_summaries(query_mean, key_mean, layout):
    return packed_routes_from_summaries(
        query_mean,
        key_mean,
        key_mean[:, :, :0],
        layout,
        _MEAN_ROUTING,
    )


def packed_mean_pool_routes_and_coarse_from_summaries(
    query_mean,
    key_mean,
    pooled_value,
    layout,
    *,
    sparse_key_blocks,
    coarse_scale,
):
    return packed_routes_and_coarse_from_summaries(
        query_mean,
        key_mean,
        key_mean[:, :, :0],
        pooled_value,
        layout,
        sparse_key_blocks=sparse_key_blocks,
        coarse_scale=coarse_scale,
        routing_mode=_MEAN_ROUTING,
    )


def _layout(
    head_keep_block_values: tuple[int, ...],
    sparse_key_blocks: int,
    device: torch.device | str = "cpu",
):
    ratios = tuple(value / sparse_key_blocks for value in head_keep_block_values)
    return _resolve_route_layout(
        _normalize_head_keep_ratios(ratios),
        sparse_key_blocks,
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

    assert routes.route_head_offsets.tolist() == [0, 1, 3]
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


def test_route_and_coarse_path_reuses_chunked_mean_scores() -> None:
    generator = torch.Generator().manual_seed(72)
    query_mean = torch.randn((1, 2, 385, 8), generator=generator)
    key_mean = torch.randn((1, 2, 5, 8), generator=generator)
    pooled_value = torch.randn((1, 2, 5, 6), generator=generator)
    sparse_key_blocks = 3
    layout = _layout((1, 2), sparse_key_blocks)
    coarse_scale = 8**-0.5

    actual = packed_mean_pool_routes_and_coarse_from_summaries(
        query_mean,
        key_mean,
        pooled_value,
        layout,
        sparse_key_blocks=sparse_key_blocks,
        coarse_scale=coarse_scale,
    )
    expected_routes = packed_mean_pool_routes_from_summaries(
        query_mean,
        key_mean[:, :, :sparse_key_blocks],
        layout,
    )
    expected_output = torch.softmax((query_mean @ key_mean.mT) * coarse_scale, dim=-1)
    expected_output = expected_output @ pooled_value

    assert torch.equal(actual.routes.indices, expected_routes.indices)
    torch.testing.assert_close(actual.coarse_output, expected_output)


def test_route_and_coarse_builder_places_out_of_order_query_chunks_by_offset() -> None:
    pooled_value = torch.tensor([[[[1.0], [3.0]]]])
    builder = PackedRouteAndCoarseBuilder(
        _layout((1,), 2),
        pooled_value,
        batch=1,
        heads=1,
        query_blocks=2,
        sparse_key_blocks=2,
        device=torch.device("cpu"),
    )
    first_scores = torch.tensor([[[[2.0, 0.0]]]])
    second_scores = torch.tensor([[[[0.0, 2.0]]]])

    builder.write(second_scores, query_block_offset=1)
    builder.write(first_scores, query_block_offset=0)
    result = builder.finish()

    expected = torch.softmax(torch.cat((first_scores, second_scores), dim=2), dim=-1)
    expected = expected @ pooled_value
    torch.testing.assert_close(result.coarse_output, expected)
    assert result.routes.indices.tolist() == [[[0], [1]]]


@pytest.mark.gpu
@pytest.mark.parametrize("head_major", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_padded_summaries_match_portable_valid_prefix_means(
    require_native_routing: None, head_major: bool, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(73)
    block_lengths = torch.tensor([64, 17, 51], dtype=torch.int32)
    if head_major:
        query = torch.randn((1, 2, 3 * 64, 128), dtype=dtype, generator=generator)
        key = torch.randn((1, 2, 2 * 64, 128), dtype=dtype, generator=generator)
    else:
        query = torch.randn((1, 3 * 64, 2, 128), dtype=dtype, generator=generator).transpose(1, 2)
        key = torch.randn((1, 2 * 64, 2, 128), dtype=dtype, generator=generator).transpose(1, 2)
    valid_query_rows = torch.arange(64)[None, :] < block_lengths[:, None]
    valid_key_rows = torch.arange(64)[None, :] < block_lengths[:2, None]
    query_blocks = query.unflatten(2, (3, 64))
    key_blocks = key.unflatten(2, (2, 64))
    query_blocks[:, :, ~valid_query_rows] = 10_000
    key_blocks[:, :, ~valid_key_rows] = -10_000

    gpu_query, gpu_key = query.cuda(), key.cuda()
    assert _backend.select_sequence_summaries(gpu_query, gpu_key) is not None
    expected = sequence_block_summaries(query, key, _MEAN_ROUTING, block_lengths)
    actual = sequence_block_summaries(
        gpu_query,
        gpu_key,
        _MEAN_ROUTING,
        block_lengths.cuda(),
    )

    for actual_summary, expected_summary in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_summary.cpu(), expected_summary, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("sequence_length", [1, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_native_short_sequence_mean_summaries_and_coarse_residual(
    require_native_routing: None, sequence_length: int, head_dim: int
) -> None:
    generator = torch.Generator().manual_seed(74 + sequence_length + head_dim)
    query, key, value, gate = (
        torch.randn((1, sequence_length, 2, head_dim), dtype=torch.bfloat16, generator=generator)
        for _ in range(4)
    )

    head_major_q, head_major_k = query.transpose(1, 2), key.transpose(1, 2)
    gpu_hm_q, gpu_hm_k = head_major_q.cuda(), head_major_k.cuda()
    assert _backend.select_sequence_summaries(gpu_hm_q, gpu_hm_k) is not None
    expected_summaries = sequence_block_summaries(head_major_q, head_major_k, _MEAN_ROUTING)
    actual_summaries = sequence_block_summaries(gpu_hm_q, gpu_hm_k, _MEAN_ROUTING)
    for actual_summary, expected_summary in zip(actual_summaries, expected_summaries, strict=True):
        torch.testing.assert_close(actual_summary.cpu(), expected_summary, atol=1e-6, rtol=1e-6)

    scale = head_dim**-0.5
    expected_residual = sparse_piper_coarse_residual(
        query, key, value, gate, routing="mean", coarse_scale=scale
    )
    actual_residual = sparse_piper_coarse_residual(
        query.cuda(), key.cuda(), value.cuda(), gate.cuda(), routing="mean", coarse_scale=scale
    )
    torch.testing.assert_close(actual_residual.cpu(), expected_residual, atol=1e-2, rtol=1e-2)
