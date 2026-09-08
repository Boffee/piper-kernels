"""Numerical and range checks for size-dependent NVIDIA D64 schedules."""

from dataclasses import replace

import pytest
import torch

from piper_kernels import SparsePiperAttention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._nvidia import gluon as native
from piper_kernels.attention.sparse_piper_attention._nvidia import policy
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_query_from_quantized,
)
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING
from piper_kernels.attention.sparse_piper_attention.triton import _prepare_sparse_piper_attention

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not AcceleratorTarget.from_device(torch.device("cuda")).is_cuda_capability(12, 0),
    reason="requires exact NVIDIA SM120",
)


def _prepared(sequence, ratios, *, block_lengths=None, sparse_query_blocks=None):
    torch.manual_seed(331 + sequence)
    operands = [
        torch.randn(1, sequence, len(ratios), 64, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    query, key, value = (x.transpose(1, 2) for x in operands)
    blocks = sequence // 64
    layout = _resolve_route_layout(_normalize_head_keep_ratios(ratios), blocks, query.device)
    routes = packed_routes_from_sequences(
        query, key[:, :, : blocks * 64], layout, _MINMAX_ROUTING, block_lengths
    )
    prepared = _prepare_sparse_piper_attention(
        query,
        routes.indices,
        routes.head_keep_blocks,
        64**-0.5,
        sparse_key_blocks=blocks,
        route_head_offsets=routes.route_head_offsets,
        combined_key=key,
        combined_value=value,
        block_lengths=block_lengths,
        sparse_query_blocks=sparse_query_blocks,
    )
    return operands, prepared


@pytest.mark.gpu
@requires_sm120
@pytest.mark.parametrize("sequence", [65, 193, 1797, 8193, 32833])
def test_skip_dense_routing_matches_materialized_routes(monkeypatch, sequence):
    operands, prepared = _prepared(sequence, [1.0, 1.0])
    expected = torch.empty_like(operands[0])
    with monkeypatch.context() as patch:
        patch.setattr(policy, "select_attention_schedule", lambda *args, **kwargs: (64, 4))
        native._launch_sparse_piper_attention(prepared, expected.transpose(1, 2))
    actual = SparsePiperAttention([1.0, 1.0])(*operands, sparse_key_blocks=sequence // 64)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    # Prepared operands also exercise replay without rebuilding CPU metadata.
    direct = replace(
        prepared,
        context=replace(prepared.context, routes_per_query=0),
        query=replace(prepared.query, routes=prepared.query.routes[:, :, :0].contiguous()),
    )
    native._launch_sparse_piper_attention(direct, actual.transpose(1, 2))
    with torch.cuda.graph(graph):
        native._launch_sparse_piper_attention(direct, actual.transpose(1, 2))
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
        patch.setattr(policy, "select_attention_schedule", lambda *args, **kwargs: (64, 4))
        native._launch_sparse_piper_attention(prepared, expected.transpose(1, 2), **kwargs)
    native._launch_sparse_piper_attention(prepared, actual.transpose(1, 2), **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@requires_sm120
def test_large_tile_preserves_odd_query_ranges_and_output_guards(monkeypatch):
    sequence = 32833
    operands, prepared = _prepared(sequence, [1.0, 1.0])
    expected = torch.empty_like(operands[0])
    with monkeypatch.context() as patch:
        patch.setattr(policy, "select_attention_schedule", lambda *args, **kwargs: (64, 4))
        native._launch_sparse_piper_attention(prepared, expected.transpose(1, 2))
    context = replace(prepared.context, routes_per_query=0)
    routes = prepared.query.routes[:, :, :0].contiguous()
    direct = replace(prepared, context=context, query=replace(prepared.query, routes=routes))
    rows = sequence - 64
    guarded = torch.full((1, rows + 2, 2, 64), 123.0, device="cuda", dtype=torch.bfloat16)
    native._launch_sparse_piper_attention(
        direct, guarded[:, 1:-1].transpose(1, 2), query_block_offset=1
    )
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
    native._launch_sparse_piper_attention(local, guarded[:, 1:-1].transpose(1, 2))
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
        patch.setattr(policy, "select_attention_schedule", lambda *args, **kwargs: (64, 4))
        native._launch_sparse_piper_attention(prepared, expected.transpose(1, 2))
    actual = SparsePiperAttention([1.0, 1.0])(
        *operands, sparse_key_blocks=sequence // 64, block_lengths=lengths
    )
    valid = torch.arange(sequence, device="cuda") % 64 < lengths.repeat_interleave(64)
    torch.testing.assert_close(actual[:, valid], expected[:, valid], atol=0, rtol=0)
