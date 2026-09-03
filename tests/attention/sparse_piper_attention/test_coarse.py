"""Policy-independent Sparse Piper coarse-attention residual tests."""

import pytest
import torch

import piper_kernels.attention.sparse_piper_attention._routing as routing_module
from piper_kernels import (
    apply_coarse_attention_residual,
    coarse_attention,
    coarse_attention_residual,
    mean_pool_block_values,
    sparse_piper_coarse_residual,
)
from piper_kernels.attention.sparse_piper_attention._routes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention._routing import routing_scores
from piper_kernels.attention.sparse_piper_attention._summaries import (
    sequence_block_summaries,
)


def test_compact_value_pooling_handles_a_partial_final_block_and_gradients() -> None:
    value = torch.arange(65 * 2, dtype=torch.float32).view(1, 65, 1, 2).requires_grad_()

    pooled = mean_pool_block_values(value)
    pooled.sum().backward()

    expected = torch.stack((value[:, :64].mean(dim=1), value[:, 64:].mean(dim=1)), dim=2)
    expected_gradient = torch.cat(
        (
            torch.full((1, 64, 1, 2), 1 / 64),
            torch.ones((1, 1, 1, 2)),
        ),
        dim=1,
    )
    assert pooled.shape == (1, 1, 2, 2)
    assert pooled.dtype is torch.float32
    torch.testing.assert_close(pooled, expected)
    torch.testing.assert_close(value.grad, expected_gradient)


def test_internal_block_lengths_exclude_every_padded_value_tail() -> None:
    generator = torch.Generator().manual_seed(801)
    block_lengths = torch.tensor([64, 17, 51], dtype=torch.int32)
    value = torch.randn((1, 3 * 64, 2, 5), generator=generator)
    valid_rows = torch.arange(64)[None, :] < block_lengths[:, None]
    corrupted = value.view(1, 3, 64, 2, 5).clone()
    corrupted[:, ~valid_rows] = 10_000
    corrupted = corrupted.view_as(value)

    actual = mean_pool_block_values(corrupted, block_lengths)
    expected = torch.stack(
        [
            value[:, block * 64 : block * 64 + int(length)].mean(dim=1)
            for block, length in enumerate(block_lengths)
        ],
        dim=2,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("invalid_length", [0, 65])
def test_value_pooling_rejects_lengths_outside_a_k64_block(invalid_length: int) -> None:
    value = torch.ones((1, 64, 1, 2))
    block_lengths = torch.tensor([invalid_length], dtype=torch.int32)

    with pytest.raises(RuntimeError, match=r"\[1, 64\]"):
        mean_pool_block_values(value, block_lengths)


def test_coarse_attention_uses_caller_supplied_block_logits() -> None:
    block_scores = torch.tensor([[[[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]]]])
    pooled_value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]]])

    actual = coarse_attention(block_scores, pooled_value)
    expected = torch.softmax(block_scores, dim=-1) @ pooled_value

    assert actual.shape == (1, 1, 2, 2)
    torch.testing.assert_close(actual, expected)


def test_sparse_piper_coarse_residual_matches_explicit_mean_composition(
    monkeypatch,
) -> None:
    monkeypatch.setattr(routing_module, "_QUERY_CHUNK_BLOCKS", 2)
    generator = torch.Generator().manual_seed(918)
    shape = (1, 129, 2, 8)
    query = torch.randn(shape, generator=generator)
    key = torch.randn(shape, generator=generator)
    value = torch.randn(shape, generator=generator)
    compression_gate = torch.randn(shape, generator=generator)
    coarse_key_blocks = 2
    coarse_scale = shape[-1] ** -0.5

    query_mean = mean_pool_block_values(query)
    key_mean = mean_pool_block_values(key)[:, :, :coarse_key_blocks]
    pooled_value = mean_pool_block_values(value)[:, :, :coarse_key_blocks]
    expected = coarse_attention_residual(
        (query_mean @ key_mean.mT) * coarse_scale,
        pooled_value,
        compression_gate,
    )
    actual = sparse_piper_coarse_residual(
        query,
        key,
        value,
        compression_gate,
        routing="mean",
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
    )

    torch.testing.assert_close(actual, expected)


def test_coarse_residual_rejects_non_rank_four_layout_before_accessing_dimensions() -> None:
    tensors = [torch.zeros(1) for _ in range(4)]

    with pytest.raises(ValueError, match=r"\[batch,sequence,heads,features\]"):
        sparse_piper_coarse_residual(
            *tensors,
            routing="mean",
            coarse_scale=1.0,
        )


def test_sparse_piper_coarse_residual_rejects_unknown_routing() -> None:
    tensors = [torch.zeros((1, 64, 1, 4)) for _ in range(4)]

    with pytest.raises(ValueError, match="routing must be 'mean' or 'minmax'"):
        sparse_piper_coarse_residual(
            *tensors,
            routing="median",
            coarse_scale=0.5,
        )


def test_sparse_piper_coarse_residual_supports_valid_front_padded_blocks() -> None:
    generator = torch.Generator().manual_seed(919)
    block_lengths = torch.tensor([64, 17, 51], dtype=torch.int32)
    shape = (1, 3 * 64, 2, 8)
    query, key, value, compression_gate = [
        torch.randn(shape, generator=generator) for _ in range(4)
    ]
    coarse_key_blocks = 3
    coarse_scale = shape[-1] ** -0.5

    query_mean = mean_pool_block_values(query, block_lengths)
    key_mean = mean_pool_block_values(key, block_lengths)
    pooled_value = mean_pool_block_values(value, block_lengths)
    expected = coarse_attention_residual(
        (query_mean @ key_mean.mT) * coarse_scale,
        pooled_value,
        compression_gate,
    )
    actual = sparse_piper_coarse_residual(
        query,
        key,
        value,
        compression_gate,
        routing="mean",
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        block_lengths=block_lengths,
    )

    torch.testing.assert_close(actual, expected)


def test_sparse_piper_coarse_residual_can_include_a_partial_block() -> None:
    generator = torch.Generator().manual_seed(922)
    shape = (1, 129, 1, 4)
    query, key, value, compression_gate = [
        torch.randn(shape, generator=generator) for _ in range(4)
    ]
    coarse_scale = shape[-1] ** -0.5

    query_mean = mean_pool_block_values(query)
    key_mean = mean_pool_block_values(key)
    pooled_value = mean_pool_block_values(value)
    expected = coarse_attention_residual(
        (query_mean @ key_mean.mT) * coarse_scale,
        pooled_value,
        compression_gate,
    )
    actual = sparse_piper_coarse_residual(
        query,
        key,
        value,
        compression_gate,
        routing="mean",
        coarse_key_blocks=3,
        coarse_scale=coarse_scale,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("coarse_key_blocks", [0, 4])
def test_sparse_piper_coarse_residual_rejects_invalid_coarse_scope(
    coarse_key_blocks: int,
) -> None:
    tensors = [torch.zeros((1, 3 * 64, 1, 4)) for _ in range(4)]

    with pytest.raises(ValueError, match="coarse_key_blocks"):
        sparse_piper_coarse_residual(
            *tensors,
            routing="mean",
            coarse_key_blocks=coarse_key_blocks,
            coarse_scale=0.5,
        )


@pytest.mark.parametrize("compile_function", [False, True])
def test_sparse_piper_coarse_residual_preserves_gradients(
    compile_function: bool,
) -> None:
    generator = torch.Generator().manual_seed(920)
    shape = (1, 65, 1, 4)
    tensors = [torch.randn(shape, generator=generator, requires_grad=True) for _ in range(4)]
    query, key, value, compression_gate = tensors

    residual = (
        torch.compile(sparse_piper_coarse_residual, backend="eager", fullgraph=True)
        if compile_function
        else sparse_piper_coarse_residual
    )
    output = residual(
        query,
        key,
        value,
        compression_gate,
        routing="mean",
        coarse_scale=shape[-1] ** -0.5,
    )
    output.square().sum().backward()

    for tensor in tensors:
        assert tensor.grad is not None
        assert bool(torch.all(torch.isfinite(tensor.grad)))


def test_sparse_piper_coarse_residual_compiles_with_dynamic_block_lengths() -> None:
    generator = torch.Generator().manual_seed(921)
    shape = (1, 2 * 64, 1, 4)
    query, key, value, compression_gate = [
        torch.randn(shape, generator=generator) for _ in range(4)
    ]
    captured_graphs = []

    def capture_backend(graph, _example_inputs):
        captured_graphs.append(graph)
        return graph.forward

    def run(candidate_block_lengths):
        return sparse_piper_coarse_residual(
            query,
            key,
            value,
            compression_gate,
            routing="mean",
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


def test_residual_expands_each_coarse_row_over_its_fine_query_block() -> None:
    compression_gate = torch.full((1, 65, 1, 2), 0.5)
    coarse_output = torch.tensor([[[[2.0, 4.0], [6.0, 10.0]]]])

    actual = apply_coarse_attention_residual(
        coarse_output,
        compression_gate,
    )

    torch.testing.assert_close(actual[:, :64], torch.tensor([1.0, 2.0]).expand(1, 64, 1, 2))
    torch.testing.assert_close(actual[:, 64:], torch.tensor([3.0, 5.0]).expand(1, 1, 1, 2))


def test_residual_canonicalizes_a_noncontiguous_gate_layout() -> None:
    gate_storage = torch.arange(1 * 2 * 65 * 4, dtype=torch.bfloat16).reshape(1, 2, 65, 4)
    compression_gate = gate_storage.transpose(1, 2)
    coarse_output = torch.randn((1, 2, 2, 4), generator=torch.Generator().manual_seed(801))
    assert not compression_gate.is_contiguous()

    actual = apply_coarse_attention_residual(coarse_output, compression_gate)

    expanded = coarse_output.permute(0, 2, 1, 3).repeat_interleave(64, dim=1)[:, :65]
    expected = (compression_gate.float() * expanded).to(compression_gate.dtype).contiguous()
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_zero_gate_produces_an_exact_bfloat16_zero_residual() -> None:
    generator = torch.Generator().manual_seed(802)
    compression_gate = torch.zeros((1, 65, 2, 8), dtype=torch.bfloat16)
    block_scores = torch.randn((1, 2, 2, 3), generator=generator)
    pooled_value = torch.randn((1, 2, 3, 8), generator=generator)

    actual = coarse_attention_residual(
        block_scores,
        pooled_value,
        compression_gate,
    )

    assert torch.count_nonzero(actual) == 0


def test_residual_is_differentiable_independently_of_the_score_policy() -> None:
    generator = torch.Generator().manual_seed(803)
    compression_gate = torch.randn((1, 65, 2, 4), generator=generator, requires_grad=True)
    block_scores = torch.randn((1, 2, 2, 3), generator=generator, requires_grad=True)
    pooled_value = torch.randn((1, 2, 3, 4), generator=generator, requires_grad=True)

    output = coarse_attention_residual(
        block_scores,
        pooled_value,
        compression_gate,
    )
    output.square().mean().backward()

    for tensor in (compression_gate, block_scores, pooled_value):
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())


def test_pooling_policies_feed_the_same_coarse_attention_contract() -> None:
    generator = torch.Generator().manual_seed(804)
    query = torch.randn((1, 2, 2 * 64, 8), generator=generator)
    key = torch.randn((1, 2, 3 * 64, 8), generator=generator)
    value = torch.randn((1, 3 * 64, 2, 6), generator=generator)
    query_mean, key_mean, mean_aux = sequence_block_summaries(
        query,
        key,
        _MEAN_ROUTING,
    )
    query_minmax, key_max, key_min = sequence_block_summaries(
        query,
        key,
        _MINMAX_ROUTING,
    )
    mean_scores = routing_scores(query_mean, key_mean, mean_aux, _MEAN_ROUTING)
    minmax_scores = routing_scores(
        query_minmax,
        key_max,
        key_min,
        _MINMAX_ROUTING,
    )
    pooled_value = mean_pool_block_values(value)

    mean_output = coarse_attention(mean_scores, pooled_value)
    minmax_output = coarse_attention(minmax_scores, pooled_value)

    assert mean_output.shape == minmax_output.shape == (1, 2, 2, 6)
    assert not torch.equal(mean_output, minmax_output)


def test_compiled_residual_accepts_changed_block_lengths_without_another_graph() -> None:
    generator = torch.Generator().manual_seed(805)
    value = torch.randn((1, 3 * 64, 2, 4), generator=generator)
    compression_gate = torch.randn_like(value)
    block_scores = torch.randn((1, 2, 3, 3), generator=generator)
    captured_graphs = []

    def capture_backend(graph, _example_inputs):
        captured_graphs.append(graph)
        return graph.forward

    def run(candidate_lengths):
        return coarse_attention_residual(
            block_scores,
            mean_pool_block_values(value, candidate_lengths),
            compression_gate,
        )

    compiled = torch.compile(run, backend=capture_backend, fullgraph=True)
    first = compiled(torch.tensor([64, 17, 51], dtype=torch.int32))
    second = compiled(torch.tensor([64, 18, 50], dtype=torch.int32))

    assert len(captured_graphs) == 1
    assert not torch.equal(first, second)
