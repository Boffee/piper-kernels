"""Numerical, chunk-boundary, and range checks for NVIDIA SM120 schedules."""

from dataclasses import replace

import pytest
import torch
from _accelerators import exact_sm120_available
from lib.sparse_piper import reference_prepared_query

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._backend import require_attention_backend
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._nvidia import policy
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
)
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING

requires_sm120 = pytest.mark.skipif(
    not exact_sm120_available(), reason="requires exact NVIDIA SM120"
)


def _prepared(
    sequence,
    ratios,
    *,
    head_dim=64,
    kv_heads=None,
    block_lengths=None,
    sparse_query_blocks=None,
):
    torch.manual_seed(331 + sequence + head_dim)
    resolved_kv_heads = len(ratios) if kv_heads is None else kv_heads
    query_operand = torch.randn(
        1, sequence, len(ratios), head_dim, device="cuda", dtype=torch.bfloat16
    )
    key_operand, value_operand = (
        torch.randn(1, sequence, resolved_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    operands = [query_operand, key_operand, value_operand]
    query, key, value = (x.transpose(1, 2) for x in operands)
    blocks = sequence // 64
    layout = _resolve_route_layout(_normalize_head_keep_ratios(ratios), blocks, query.device)
    routes = packed_routes_from_sequences(
        query, key[:, :, : blocks * 64], layout, _MINMAX_ROUTING, block_lengths
    )
    backend = require_attention_backend(query)
    prepared = backend.prepare(
        query,
        head_dim**-0.5,
        sparse_key_blocks=blocks,
        combined_key=key,
        combined_value=value,
        block_lengths=block_lengths,
        sparse_query_blocks=sparse_query_blocks,
    ).with_routes(
        routes.indices,
        routes.head_keep_blocks,
        routes.route_head_offsets,
        bind=backend.bind_context,
    )
    return operands, prepared


def _assert_sampled_d128_matches_fp64(prepared, output, blocks, *, coarse=None):
    """Verify sampled query blocks across every head against bounded FP64 math."""
    heads = prepared.query.data.shape[1]
    lengths = prepared.context.block_lengths
    for head in range(heads):
        for block in blocks:
            expected = reference_prepared_query(
                prepared, 0, head, block, output_dtype=torch.float64
            )
            rows = expected.shape[0]
            if coarse is not None:
                coarse_output, coarse_gate = coarse
                expected = expected + (
                    coarse_gate[0, block * 64 : block * 64 + rows, head].double()
                    * coarse_output[0, head, block].double()
                )
            actual = output[0, block * 64 : block * 64 + rows, head]
            if lengths is not None:
                valid = int(lengths[block])
                actual, expected = actual[:valid], expected[:valid]
            assert torch.isfinite(actual).all()
            error = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-30)
            assert error < 0.015, (head, block, error.item())


@pytest.mark.gpu
@requires_sm120
@pytest.mark.parametrize("sequence", [65, 193, 1797, 8193, 32833])
def test_skip_dense_routing_matches_materialized_routes(monkeypatch, sequence):
    operands, prepared = _prepared(sequence, [1.0, 1.0])
    expected = torch.empty_like(operands[0])
    with monkeypatch.context() as patch:
        patch.setattr(policy, "select_sm120_attention_schedule", lambda *args, **kwargs: (64, 4))
        prepared.launch(prepared.query, expected.transpose(1, 2))
    actual = SparsePiperAttention([1.0, 1.0])(*operands, sparse_key_blocks=sequence // 64)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    # Prepared operands also exercise replay without rebuilding CPU metadata.
    direct = replace(
        prepared,
        launch=require_attention_backend(prepared.context.key).bind_context(
            replace(prepared.context, routes_per_query=0)
        ),
        query=replace(prepared.query, routes=prepared.query.routes[:, :, :0].contiguous()),
    )
    direct.launch(direct.query, actual.transpose(1, 2))
    with torch.cuda.graph(graph):
        direct.launch(direct.query, actual.transpose(1, 2))
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@requires_sm120
@pytest.mark.parametrize("coarse", [False, True])
def test_two_warps_preserve_mixed_routes_dense_suffix_and_coarse(monkeypatch, coarse):
    operands, prepared = _prepared(8193, [0.25, 0.5, 1.0], sparse_query_blocks=80)
    expected, actual = torch.empty_like(operands[0]), torch.empty_like(operands[0])
    kwargs = {}
    if coarse:
        kwargs = {
            "coarse_output": torch.randn(1, 3, 129, 64, device="cuda"),
            "coarse_gate": torch.randn_like(operands[0]),
        }
    with monkeypatch.context() as patch:
        patch.setattr(policy, "select_sm120_attention_schedule", lambda *args, **kwargs: (64, 4))
        prepared.launch(prepared.query, expected.transpose(1, 2), **kwargs)
    prepared.launch(prepared.query, actual.transpose(1, 2), **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@requires_sm120
def test_large_tile_preserves_odd_query_ranges_and_output_guards(monkeypatch):
    sequence = 32833
    operands, prepared = _prepared(sequence, [1.0, 1.0])
    expected = torch.empty_like(operands[0])
    with monkeypatch.context() as patch:
        patch.setattr(policy, "select_sm120_attention_schedule", lambda *args, **kwargs: (64, 4))
        prepared.launch(prepared.query, expected.transpose(1, 2))
    context = replace(prepared.context, routes_per_query=0)
    routes = prepared.query.routes[:, :, :0].contiguous()
    direct = replace(
        prepared,
        launch=require_attention_backend(prepared.context.key).bind_context(context),
        query=replace(prepared.query, routes=routes),
    )
    rows = sequence - 64
    guarded = torch.full((1, rows + 2, 2, 64), 123.0, device="cuda", dtype=torch.bfloat16)
    direct.launch(direct.query, guarded[:, 1:-1].transpose(1, 2), query_block_offset=1)
    torch.testing.assert_close(guarded[:, 1:-1], expected[:, 64:], atol=0, rtol=0)
    assert torch.all(guarded[:, (0, -1)] == 123.0)

    local = replace(
        direct,
        query=_prepare_sparse_piper_query_from_quantized(
            prepared.query.data[:, :, 64:].contiguous(),
            prepared.query.scale[:, :, 2:].contiguous(),
            routes[:, 1:].contiguous(),
            context,
            global_block_offset=1,
        ),
    )
    local.launch(local.query, guarded[:, 1:-1].transpose(1, 2))
    torch.testing.assert_close(guarded[:, 1:-1], expected[:, 64:], atol=0, rtol=0)
    assert torch.all(guarded[:, (0, -1)] == 123.0)


@pytest.mark.gpu
@requires_sm120
def test_large_tile_preserves_internally_padded_blocks(monkeypatch):
    sequence = 32832
    lengths = torch.full((sequence // 64,), 64, device="cuda", dtype=torch.int32)
    lengths[::3] = 17
    lengths[-1] = 51
    operands, prepared = _prepared(sequence, [1.0, 1.0], block_lengths=lengths)
    expected = torch.empty_like(operands[0])
    with monkeypatch.context() as patch:
        patch.setattr(policy, "select_sm120_attention_schedule", lambda *args, **kwargs: (64, 4))
        prepared.launch(prepared.query, expected.transpose(1, 2))
    actual = SparsePiperAttention([1.0, 1.0])(
        *operands, sparse_key_blocks=sequence // 64, block_lengths=lengths
    )
    valid = torch.arange(sequence, device="cuda") % 64 < lengths.repeat_interleave(64)
    torch.testing.assert_close(actual[:, valid], expected[:, valid], atol=0, rtol=0)


@pytest.mark.gpu
@requires_sm120
@pytest.mark.parametrize("sequence", [2048, 2049, 3101])
@pytest.mark.parametrize("coarse", [False, True])
def test_d128_chunked_value_mean_and_dense_suffix_match_fp64(sequence, coarse):
    ratios = (0.25, 0.5, 0.75, 1.0)
    heads = len(ratios)
    query_blocks = (sequence + 63) // 64
    sparse_query_blocks = (sequence // 64) // 2
    operands, prepared = _prepared(
        sequence,
        ratios,
        head_dim=128,
        kv_heads=2,
        sparse_query_blocks=sparse_query_blocks,
    )
    output = torch.empty_like(operands[0])
    kwargs = {}
    if coarse:
        kwargs = {
            "coarse_output": torch.randn(1, heads, query_blocks, 128, device="cuda"),
            "coarse_gate": torch.randn_like(operands[0]),
        }
    prepared.launch(prepared.query, output.transpose(1, 2), **kwargs)
    coarse_terms = (kwargs["coarse_output"], kwargs["coarse_gate"]) if coarse else None
    sampled = sorted({0, sparse_query_blocks - 1, sparse_query_blocks, query_blocks - 1})
    _assert_sampled_d128_matches_fp64(prepared, output, sampled, coarse=coarse_terms)

    rows = sequence - 64
    guarded = torch.full((1, rows + 2, heads, 128), 123.0, device="cuda", dtype=torch.bfloat16)
    window_kwargs = (
        {"coarse_output": kwargs["coarse_output"], "coarse_gate": kwargs["coarse_gate"][:, 64:]}
        if coarse
        else {}
    )
    prepared.launch(
        prepared.query,
        guarded[:, 1:-1].transpose(1, 2),
        query_block_offset=1,
        **window_kwargs,
    )
    window_coarse = (kwargs["coarse_output"], kwargs["coarse_gate"]) if coarse else None
    _assert_sampled_d128_matches_fp64(
        prepared,
        torch.cat((output[:, :64], guarded[:, 1:-1]), dim=1),
        [block for block in sampled if block >= 1],
        coarse=window_coarse,
    )
    assert torch.all(guarded[:, (0, -1)] == 123.0)


@pytest.mark.gpu
@requires_sm120
def test_d128_padded_blocks_and_unselected_value_means_match_fp64():
    sequence = 2304
    tiles = sequence // 64
    lengths = torch.full((tiles,), 64, device="cuda", dtype=torch.int32)
    lengths[::3] = 17
    lengths[-1] = 51
    operands, prepared = _prepared(
        sequence, (0.25, 0.5, 0.75, 1.0), head_dim=128, kv_heads=2, block_lengths=lengths
    )
    output = torch.empty_like(operands[0])
    prepared.launch(prepared.query, output.transpose(1, 2))
    _assert_sampled_d128_matches_fp64(prepared, output, (0, 17, tiles - 1))

    generator = torch.Generator(device="cuda").manual_seed(911)
    for selected in ([3, 7, 15, 22, 31], list(range(1, 35))):
        route_count = len(selected)
        value_mean = torch.full(
            (1, 1, tiles, 128), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        value_mean[0, 0, selected] = torch.randn(
            (route_count, 128), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        context = _prepare_sparse_piper_context_from_quantized(
            torch.randint(
                -31, 32, (1, 1, sequence, 128), device="cuda", dtype=torch.int8, generator=generator
            ),
            torch.full((1, 1, tiles), 0.05, device="cuda"),
            torch.randint(
                -31, 32, (1, 1, tiles, 2, 128), device="cuda", dtype=torch.int8, generator=generator
            ),
            torch.tensor([0.04, 0.002], device="cuda").expand(1, 1, tiles, 2).contiguous(),
            torch.randint(
                -128,
                128,
                (1, 1, 128, sequence),
                device="cuda",
                dtype=torch.int8,
                generator=generator,
            ),
            torch.exp2(
                torch.empty((1, 1, tiles, 1), device="cuda").uniform_(-8, 8, generator=generator)
            ),
            value_mean,
            torch.tensor([route_count], device="cuda", dtype=torch.int32),
            torch.tensor([0, route_count], device="cuda", dtype=torch.int32),
            value_format=ValueFormat.TRANSPOSED,
            sparse_key_blocks=tiles,
            routes_per_query=route_count,
            logical_sequence_length=sequence,
        )
        routes = torch.tensor(selected, device="cuda", dtype=torch.uint16).expand(1, 1, route_count)
        query = _prepare_sparse_piper_query_from_quantized(
            torch.randint(
                -31, 32, (1, 1, 64, 128), device="cuda", dtype=torch.int8, generator=generator
            ),
            torch.full((1, 1, 2), 0.05, device="cuda"),
            routes.contiguous(),
            context,
        )
        manual_prepared = _PreparedSparsePiperAttention(
            require_attention_backend(context.key).bind_context(context), query
        )
        manual_output = torch.empty((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16)
        manual_prepared.launch(manual_prepared.query, manual_output)
        expected = reference_prepared_query(manual_prepared, 0, 0, 0, output_dtype=torch.float64)
        actual = manual_output[0, 0]
        assert torch.isfinite(actual).all()
        error = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-30)
        assert error < 0.015, (route_count, error.item())
