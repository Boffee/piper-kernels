"""Fused AMD sparse attention correctness, including routes and storage tails."""

import pytest
import torch
from lib.sparse_piper import reference_prepared_query

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._amd.gluon import _launch_sparse_piper_attention
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
)
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention.reference import (
    reference_sparse_piper_attention,
)
from piper_kernels.attention.sparse_piper_attention.triton import _prepare_sparse_piper_attention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not AcceleratorTarget.from_device(torch.device("cuda")).is_amd_hip
        or not AcceleratorTarget.from_device(torch.device("cuda")).is_architecture(
            "gfx1200", "gfx1201"
        ),
        reason="requires an RDNA4 ROCm GPU",
    ),
]


@pytest.mark.parametrize("sequence", [64, 65, 127, 128, 193, 320])
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
def test_fused_amd_matches_reference(sequence, routing_mode):
    generator = torch.Generator(device="cuda").manual_seed(913 + sequence)
    q, k, v = [
        torch.randn((2, sequence, 2, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
        for _ in range(3)
    ]
    blocks = max(1, sequence // 64 - 1)
    layout = _resolve_route_layout(_normalize_head_keep_ratios((0.5, 1.0)), blocks, q.device)
    routes = packed_routes_from_sequences(
        q.transpose(1, 2), k.transpose(1, 2)[:, :, : blocks * 64], layout, routing_mode
    )
    prepared = _prepare_sparse_piper_attention(
        q.transpose(1, 2),
        routes.indices,
        routes.head_keep_blocks,
        128**-0.5,
        sparse_key_blocks=blocks,
        route_head_offsets=routes.route_head_offsets,
        combined_key=k.transpose(1, 2),
        combined_value=v.transpose(1, 2),
    )
    expected = reference_sparse_piper_attention(
        q, k, v, routes, sparse_key_blocks=blocks, scale=128**-0.5
    )
    storage = torch.full((q.numel() + 16,), -123, device=q.device, dtype=q.dtype)
    actual = storage[8:-8].view(q.shape)
    _launch_sparse_piper_attention(prepared, actual.transpose(1, 2))
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert torch.isfinite(actual).all()
    assert error < 0.015, error.item()
    assert (storage[:8] == -123).all()
    assert (storage[-8:] == -123).all()


@pytest.mark.parametrize("global_query_block", [0, 781, 1562])
def test_long_context_local_query_matches_fp64(global_query_block):
    # Exercise hundreds of online updates and the final 32-token tail without
    # allocating a complete 100k-token query/output tensor in the test suite.
    generator = torch.Generator(device="cuda").manual_seed(2137)
    sequence, padded, heads = 100000, 100032, 3
    keep = [391, 390, 1]
    key = torch.randint(
        -31, 32, (1, heads, padded, 128), dtype=torch.int8, device="cuda", generator=generator
    )
    value = torch.randint(
        -128, 128, (1, heads, 128, padded), dtype=torch.int8, device="cuda", generator=generator
    )
    multiplier = torch.exp2(
        torch.empty((1, heads, padded // 64, 1), device="cuda").uniform_(
            -12, 12, generator=generator
        )
    )
    context = _prepare_sparse_piper_context_from_quantized(
        key,
        torch.full((1, heads, padded // 64), 0.1, device="cuda"),
        value,
        multiplier,
        torch.randn((1, heads, 128), device="cuda", generator=generator),
        torch.tensor(keep, device="cuda", dtype=torch.int32),
        torch.tensor([0, 391, 781, 782], device="cuda", dtype=torch.int32),
        sparse_key_blocks=sequence // 64,
        routes_per_query=sum(keep),
        logical_sequence_length=sequence,
    )
    routes = torch.cat(
        [
            torch.randperm(sequence // 64, device="cuda", generator=generator)[:count]
            for count in keep
        ]
    ).to(torch.uint16)
    query = _prepare_sparse_piper_query_from_quantized(
        torch.randint(
            -31, 32, (1, heads, 64, 128), dtype=torch.int8, device="cuda", generator=generator
        ),
        torch.full((1, heads, 2), 0.001, device="cuda"),
        routes.view(1, 1, -1),
        context,
        global_block_offset=global_query_block,
    )
    prepared = _PreparedSparsePiperAttention(context, query)
    rows = min(64, sequence - global_query_block * 64)
    guarded = torch.full((heads * rows * 128 + 16,), -123, device="cuda", dtype=torch.bfloat16)
    output = guarded[8:-8].view(1, heads, rows, 128)
    _launch_sparse_piper_attention(prepared, output)
    for head in range(heads):
        expected = reference_prepared_query(prepared, 0, head, 0)
        actual = output[0, head]
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert torch.isfinite(actual).all()
        assert error < 0.015, (head, error.item())
    assert (guarded[:8] == -123).all()
    assert (guarded[-8:] == -123).all()
