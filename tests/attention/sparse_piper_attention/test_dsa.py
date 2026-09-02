"""Exact packed DSA routing tests."""

import pytest
import torch

import piper_kernels.attention.sparse_piper_attention.dsa as dsa_module
from piper_kernels import (
    coarse_attention_residual,
    dsa_coarse_residual,
    mean_pool_block_values,
)
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention.dsa import (
    _dsa_scores,
    _sequence_block_summaries,
    packed_dsa_routes_and_coarse_from_summaries,
    packed_dsa_routes_from_sequences,
    packed_dsa_routes_from_summaries,
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


def test_packed_routes_store_only_active_uint16_indices() -> None:
    generator = torch.Generator().manual_seed(53)
    query = torch.randn((1, 3, 2 * 64, 128), generator=generator)
    key = torch.randn((1, 3, 5 * 64, 128), generator=generator)
    layout = _layout((1, 3, 2), 5)

    routes = packed_dsa_routes_from_sequences(query, key, layout)

    assert routes.indices.shape == (1, 2, 6)
    assert routes.indices.dtype is torch.uint16
    assert routes.route_head_offsets.tolist() == [0, 1, 4, 6]
    assert bool((routes.indices.to(torch.int32) < 5).all())


def test_exact_score_ties_prefer_lower_key_index() -> None:
    query = torch.zeros((1, 1, 2 * 64, 128))
    key = torch.zeros((1, 1, 4 * 64, 128))
    layout = _layout((2,), 4)

    routes = packed_dsa_routes_from_sequences(query, key, layout)

    assert routes.indices.tolist() == [[[0, 1], [0, 1]]]


def test_existing_summaries_select_the_same_routes_as_sequence_inputs() -> None:
    generator = torch.Generator().manual_seed(55)
    query = torch.randn((1, 3, 4 * 64, 128), generator=generator)
    key = torch.randn((1, 3, 6 * 64, 128), generator=generator)
    layout = _layout((1, 4, 2), 6)
    query_float = query.unflatten(2, (4, 64)).float()
    key_float = key.unflatten(2, (6, 64)).float()
    query_summary = query_float.amax(dim=3) + query_float.amin(dim=3)
    key_max = key_float.amax(dim=3)
    key_min = key_float.amin(dim=3)

    expected = packed_dsa_routes_from_sequences(query, key, layout)
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


def test_ragged_query_sequence_produces_a_final_route_row() -> None:
    generator = torch.Generator().manual_seed(64)
    query = torch.randn((1, 2, 65, 128), generator=generator)
    key = torch.randn((1, 2, 3 * 64, 128), generator=generator)
    layout = _layout((1, 2), 3)

    routes = packed_dsa_routes_from_sequences(query, key, layout)

    assert routes.indices.shape == (1, 2, 3)
    assert bool((routes.indices.to(torch.int32) < 3).all())


def test_padded_summaries_and_routes_ignore_every_invalid_block_tail() -> None:
    generator = torch.Generator().manual_seed(641)
    block_lengths = torch.tensor([64, 17, 51], dtype=torch.int32)
    query = torch.randn((1, 2, 3 * 64, 8), generator=generator)
    key = torch.randn((1, 2, 2 * 64, 8), generator=generator)
    query_blocks = query.unflatten(2, (3, 64))
    key_blocks = key.unflatten(2, (2, 64))
    expected_query = torch.stack(
        [
            block[:, :, : int(length)].amax(dim=2) + block[:, :, : int(length)].amin(dim=2)
            for block, length in zip(query_blocks.unbind(2), block_lengths, strict=True)
        ],
        dim=2,
    )
    expected_key_max = torch.stack(
        [
            block[:, :, : int(length)].amax(dim=2)
            for block, length in zip(
                key_blocks.unbind(2),
                block_lengths[:2],
                strict=True,
            )
        ],
        dim=2,
    )
    expected_key_min = torch.stack(
        [
            block[:, :, : int(length)].amin(dim=2)
            for block, length in zip(
                key_blocks.unbind(2),
                block_lengths[:2],
                strict=True,
            )
        ],
        dim=2,
    )
    valid_query_rows = torch.arange(64)[None, :] < block_lengths[:, None]
    valid_key_rows = torch.arange(64)[None, :] < block_lengths[:2, None]
    layout = _layout((1, 2), 2)
    expected_routes = packed_dsa_routes_from_summaries(
        expected_query,
        expected_key_max,
        expected_key_min,
        layout,
    )

    for fill in (-10_000.0, 10_000.0):
        corrupted_query = query_blocks.clone()
        corrupted_key = key_blocks.clone()
        corrupted_query[:, :, ~valid_query_rows] = fill
        corrupted_key[:, :, ~valid_key_rows] = fill
        corrupted_query = corrupted_query.flatten(2, 3)
        corrupted_key = corrupted_key.flatten(2, 3)
        query_summary, key_max, key_min = _sequence_block_summaries(
            corrupted_query,
            corrupted_key,
            block_lengths,
        )
        routes = packed_dsa_routes_from_sequences(
            corrupted_query,
            corrupted_key,
            layout,
            block_lengths,
        )

        torch.testing.assert_close(query_summary, expected_query)
        torch.testing.assert_close(key_max, expected_key_max)
        torch.testing.assert_close(key_min, expected_key_min)
        assert torch.equal(routes.indices, expected_routes.indices)


def test_dsa_coarse_residual_matches_padded_sparse_prefix_composition(monkeypatch) -> None:
    monkeypatch.setattr(dsa_module, "_QUERY_CHUNK_BLOCKS", 2)
    generator = torch.Generator().manual_seed(642)
    block_lengths = torch.tensor([64, 17, 51], dtype=torch.int32)
    shape = (1, 3 * 64, 2, 8)
    fine_output, query, key, value, compression_gate = [
        torch.randn(shape, generator=generator) for _ in range(5)
    ]
    sparse_key_blocks = 2
    coarse_key_blocks = 3
    coarse_scale = shape[-1] ** -0.5
    query_summary, key_max, key_min = _sequence_block_summaries(
        query.transpose(1, 2),
        key.transpose(1, 2)[:, :, : coarse_key_blocks * 64],
        block_lengths,
    )
    pooled_value = mean_pool_block_values(value, block_lengths)[:, :, :coarse_key_blocks]
    expected = coarse_attention_residual(
        fine_output,
        _dsa_scores(query_summary, key_max, key_min) * coarse_scale,
        pooled_value,
        compression_gate,
    )
    actual = dsa_coarse_residual(
        fine_output,
        query,
        key,
        value,
        compression_gate,
        sparse_key_blocks=sparse_key_blocks,
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        block_lengths=block_lengths,
    )

    torch.testing.assert_close(actual, expected)


def test_dsa_coarse_residual_can_include_a_partial_block_after_sparse_prefix() -> None:
    generator = torch.Generator().manual_seed(645)
    shape = (1, 129, 1, 4)
    fine_output, query, key, value, compression_gate = [
        torch.randn(shape, generator=generator) for _ in range(5)
    ]
    coarse_scale = shape[-1] ** -0.5
    query_summary, key_max, key_min = _sequence_block_summaries(
        query.transpose(1, 2),
        key.transpose(1, 2),
    )
    expected = coarse_attention_residual(
        fine_output,
        _dsa_scores(query_summary, key_max, key_min) * coarse_scale,
        mean_pool_block_values(value),
        compression_gate,
    )
    actual = dsa_coarse_residual(
        fine_output,
        query,
        key,
        value,
        compression_gate,
        sparse_key_blocks=2,
        coarse_key_blocks=3,
        coarse_scale=coarse_scale,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("compile_function", [False, True])
def test_dsa_coarse_residual_preserves_composed_gradients(
    compile_function: bool,
) -> None:
    generator = torch.Generator().manual_seed(643)
    shape = (1, 129, 1, 4)
    tensors = [torch.randn(shape, generator=generator, requires_grad=True) for _ in range(5)]
    fine_output, query, key, value, compression_gate = tensors

    residual = (
        torch.compile(dsa_coarse_residual, backend="eager", fullgraph=True)
        if compile_function
        else dsa_coarse_residual
    )
    output = residual(
        fine_output,
        query,
        key,
        value,
        compression_gate,
        sparse_key_blocks=2,
        coarse_scale=shape[-1] ** -0.5,
    )
    output.square().sum().backward()

    for tensor in tensors:
        assert tensor.grad is not None
        assert bool(torch.all(torch.isfinite(tensor.grad)))


def test_dsa_coarse_residual_compiles_with_dynamic_block_lengths() -> None:
    generator = torch.Generator().manual_seed(644)
    shape = (1, 2 * 64, 1, 4)
    fine_output, query, key, value, compression_gate = [
        torch.randn(shape, generator=generator) for _ in range(5)
    ]
    captured_graphs = []

    def capture_backend(graph, _example_inputs):
        captured_graphs.append(graph)
        return graph.forward

    def run(candidate_block_lengths):
        return dsa_coarse_residual(
            fine_output,
            query,
            key,
            value,
            compression_gate,
            sparse_key_blocks=2,
            coarse_scale=shape[-1] ** -0.5,
            block_lengths=candidate_block_lengths,
        )

    compiled = torch.compile(run, backend=capture_backend, fullgraph=True)
    first = compiled(torch.tensor([64, 17], dtype=torch.int32))
    second = compiled(torch.tensor([63, 18], dtype=torch.int32))

    assert len(captured_graphs) == 1
    targets = [node.target for node in captured_graphs[0].graph.nodes if node.op == "call_function"]
    assert targets == [torch.ops.piper_kernels.sparse_piper_coarse_residual.default]
    assert not torch.equal(first, second)


def test_route_and_coarse_path_reuses_chunked_dsa_scores() -> None:
    generator = torch.Generator().manual_seed(66)
    query_summary = torch.randn((1, 2, 385, 8), generator=generator)
    key_max = torch.randn((1, 2, 5, 8), generator=generator)
    key_min = torch.randn((1, 2, 5, 8), generator=generator)
    pooled_value = torch.randn((1, 2, 5, 6), generator=generator)
    sparse_key_blocks = 3
    layout = _layout((1, 2), sparse_key_blocks)
    coarse_scale = 8**-0.5

    actual = packed_dsa_routes_and_coarse_from_summaries(
        query_summary,
        key_max,
        key_min,
        pooled_value,
        layout,
        sparse_key_blocks=sparse_key_blocks,
        coarse_scale=coarse_scale,
    )
    expected_routes = packed_dsa_routes_from_summaries(
        query_summary,
        key_max[:, :, :sparse_key_blocks],
        key_min[:, :, :sparse_key_blocks],
        layout,
    )
    expected_scores = _dsa_scores(query_summary, key_max, key_min) * coarse_scale
    expected_output = torch.softmax(expected_scores, dim=-1) @ pooled_value

    assert torch.equal(actual.routes.indices, expected_routes.indices)
    torch.testing.assert_close(actual.coarse_output, expected_output)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_packed_routes_match_the_portable_exact_policy() -> None:
    generator = torch.Generator().manual_seed(54)
    query = torch.randn((1, 3, 5 * 64, 128), dtype=torch.bfloat16, generator=generator)
    key = torch.randn((1, 3, 7 * 64, 128), dtype=torch.bfloat16, generator=generator)
    cpu_layout = _layout((1, 4, 6), 7)
    cuda_layout = _layout((1, 4, 6), 7, "cuda")

    expected = packed_dsa_routes_from_sequences(query, key, cpu_layout)
    actual = packed_dsa_routes_from_sequences(query.cuda(), key.cuda(), cuda_layout)

    torch.testing.assert_close(
        actual.indices.cpu().to(torch.int32),
        expected.indices.to(torch.int32),
        atol=0,
        rtol=0,
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_padded_summaries_match_portable_valid_prefix_extrema() -> None:
    generator = torch.Generator().manual_seed(545)
    block_lengths = torch.tensor([64, 17, 51], dtype=torch.int32)
    query = torch.randn((1, 2, 3 * 64, 128), dtype=torch.bfloat16, generator=generator)
    key = torch.randn((1, 2, 2 * 64, 128), dtype=torch.bfloat16, generator=generator)
    valid_query_rows = torch.arange(64)[None, :] < block_lengths[:, None]
    valid_key_rows = torch.arange(64)[None, :] < block_lengths[:2, None]
    query = query.unflatten(2, (3, 64))
    key = key.unflatten(2, (2, 64))
    query[:, :, ~valid_query_rows] = 10_000
    key[:, :, ~valid_key_rows] = -10_000
    query = query.flatten(2, 3)
    key = key.flatten(2, 3)

    expected = _sequence_block_summaries(query, key, block_lengths)
    actual = _sequence_block_summaries(
        query.cuda(),
        key.cuda(),
        block_lengths.cuda(),
    )

    for actual_summary, expected_summary in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_summary.cpu(), expected_summary, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_ragged_key_summaries_match_portable_extrema() -> None:
    generator = torch.Generator().manual_seed(546)
    query = torch.randn((1, 2, 129, 128), dtype=torch.bfloat16, generator=generator)
    key = torch.randn((1, 2, 193, 128), dtype=torch.bfloat16, generator=generator)

    expected = _sequence_block_summaries(query, key)
    actual = _sequence_block_summaries(query.cuda(), key.cuda())

    for actual_summary, expected_summary in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_summary.cpu(), expected_summary, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_ragged_routes_ignore_invalid_query_storage() -> None:
    generator = torch.Generator(device="cuda").manual_seed(65)
    query_storage = torch.randn(
        (1, 2, 128, 128),
        dtype=torch.bfloat16,
        generator=generator,
        device="cuda",
    )
    query = query_storage[:, :, :65]
    key = torch.randn(
        (1, 2, 4 * 64, 128),
        dtype=torch.bfloat16,
        generator=generator,
        device="cuda",
    )
    layout = _layout((1, 3), 4, "cuda")

    query_storage[:, :, 65:] = 10_000
    positive_summary, _key_max, _key_min = _sequence_block_summaries(query, key)
    positive_padding = packed_dsa_routes_from_sequences(query, key, layout)
    query_storage[:, :, 65:] = -10_000
    negative_summary, _key_max, _key_min = _sequence_block_summaries(query, key)
    negative_padding = packed_dsa_routes_from_sequences(query, key, layout)

    assert torch.equal(positive_summary, negative_summary)
    assert torch.equal(positive_padding.indices, negative_padding.indices)
