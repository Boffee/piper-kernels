"""Fused AMD sparse attention correctness, including routes and storage tails."""

import pytest
import torch
from lib.sparse_piper import reference_prepared_query, reference_value_storage

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._backend import require_attention_backend
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
)
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention.reference import (
    reference_sparse_piper_attention,
)

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
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_amd_matches_reference(sequence, routing_mode, head_dim, dtype):
    generator = torch.Generator(device="cuda").manual_seed(913 + sequence)
    q, k, v = [
        torch.randn((2, sequence, 2, head_dim), dtype=dtype, device="cuda", generator=generator)
        for _ in range(3)
    ]
    blocks = max(1, sequence // 64 - 1)
    layout = _resolve_route_layout(_normalize_head_keep_ratios((0.5, 1.0)), blocks, q.device)
    routes = packed_routes_from_sequences(
        q.transpose(1, 2), k.transpose(1, 2)[:, :, : blocks * 64], layout, routing_mode
    )
    backend = require_attention_backend(q.transpose(1, 2))
    prepared = backend.prepare(
        q.transpose(1, 2),
        head_dim**-0.5,
        sparse_key_blocks=blocks,
        combined_key=k.transpose(1, 2),
        combined_value=v.transpose(1, 2),
    ).with_routes(
        routes.indices,
        routes.head_keep_blocks,
        routes.route_head_offsets,
        bind=backend.bind_context,
    )
    expected = reference_sparse_piper_attention(
        q, k, v, routes, sparse_key_blocks=blocks, scale=head_dim**-0.5
    )
    storage = torch.full((q.numel() + 16,), -123, device=q.device, dtype=q.dtype)
    actual = storage[8:-8].view(q.shape)
    prepared.launch(prepared.query, actual.transpose(1, 2))
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert torch.isfinite(actual).all()
    assert error < 0.015, error.item()
    assert (storage[:8] == -123).all()
    assert (storage[-8:] == -123).all()


@pytest.mark.parametrize("global_query_block", [0, 781, 1562])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_long_context_local_query_matches_fp64(global_query_block, head_dim):
    # Exercise hundreds of online updates and the final 32-token tail without
    # allocating a complete 100k-token query/output tensor in the test suite.
    generator = torch.Generator(device="cuda").manual_seed(2137)
    sequence, padded, heads = 100000, 100032, 3
    keep = [391, 390, 1]
    tiles = padded // 64
    key = torch.randint(
        -31, 32, (1, heads, padded, head_dim), dtype=torch.int8, device="cuda", generator=generator
    )
    key_center = torch.randint(
        -31,
        32,
        (1, heads, tiles, 2, head_dim),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    key_center_scale = (
        torch.tensor([0.05, 0.0005], device="cuda").expand(1, heads, tiles, 2).contiguous()
    )
    value = torch.randint(
        -128,
        128,
        (1, heads, head_dim, padded),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    multiplier = torch.exp2(
        torch.empty((1, heads, tiles, 1), device="cuda").uniform_(-12, 12, generator=generator)
    )
    value_mean = torch.randn(
        (1, heads, tiles, head_dim), device="cuda", generator=generator
    ).bfloat16()
    context = _prepare_sparse_piper_context_from_quantized(
        key,
        torch.full((1, heads, tiles), 0.1, device="cuda"),
        key_center,
        key_center_scale,
        reference_value_storage(value, ValueFormat.AMD_WMMA),
        multiplier,
        value_mean,
        torch.tensor(keep, device="cuda", dtype=torch.int32),
        torch.tensor([0, 391, 781, 782], device="cuda", dtype=torch.int32),
        sparse_key_blocks=sequence // 64,
        routes_per_query=sum(keep),
        logical_sequence_length=sequence,
        value_format=ValueFormat.AMD_WMMA,
    )
    routes = torch.cat(
        [
            torch.randperm(sequence // 64, device="cuda", generator=generator)[:count]
            for count in keep
        ]
    ).to(torch.uint16)
    query = _prepare_sparse_piper_query_from_quantized(
        torch.randint(
            -31, 32, (1, heads, 64, head_dim), dtype=torch.int8, device="cuda", generator=generator
        ),
        torch.full((1, heads, 2), 0.001, device="cuda"),
        routes.view(1, 1, -1),
        context,
        global_block_offset=global_query_block,
    )
    prepared = _PreparedSparsePiperAttention(
        require_attention_backend(context.key).bind_context(context), query
    )
    rows = min(64, sequence - global_query_block * 64)
    guarded = torch.full((heads * rows * head_dim + 16,), -123, device="cuda", dtype=torch.bfloat16)
    output = guarded[8:-8].view(1, heads, rows, head_dim)
    prepared.launch(prepared.query, output)
    for head in range(heads):
        expected = reference_prepared_query(prepared, 0, head, 0)
        actual = output[0, head]
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert torch.isfinite(actual).all()
        assert error < 0.015, (head, error.item())
    assert (guarded[:8] == -123).all()
    assert (guarded[-8:] == -123).all()


@pytest.mark.parametrize("kept", [1, 8, 9, 24, 25])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fused_amd_partial_chunk_bias_boundary_matches_reference(kept, head_dim):
    sequence, blocks, heads = 1600, 25, 2
    generator = torch.Generator(device="cuda").manual_seed(1703 + kept + head_dim)
    q, k, v = [
        torch.randn(
            (1, sequence, heads, head_dim), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        for _ in range(3)
    ]
    layout = _resolve_route_layout(
        _normalize_head_keep_ratios((kept / blocks,) * heads), blocks, q.device
    )
    assert layout.head_keep_blocks.tolist() == [kept] * heads
    routes = packed_routes_from_sequences(
        q.transpose(1, 2), k.transpose(1, 2), layout, _MINMAX_ROUTING
    )
    backend = require_attention_backend(q.transpose(1, 2))
    prepared = backend.prepare(
        q.transpose(1, 2),
        head_dim**-0.5,
        sparse_key_blocks=blocks,
        combined_key=k.transpose(1, 2),
        combined_value=v.transpose(1, 2),
    ).with_routes(
        routes.indices,
        routes.head_keep_blocks,
        routes.route_head_offsets,
        bind=backend.bind_context,
    )
    expected = reference_sparse_piper_attention(
        q, k, v, routes, sparse_key_blocks=blocks, scale=head_dim**-0.5
    )
    storage = torch.full((q.numel() + 16,), -123, device=q.device, dtype=q.dtype)
    actual = storage[8:-8].view(q.shape)
    prepared.launch(prepared.query, actual.transpose(1, 2))
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert torch.isfinite(actual).all()
    assert error < 0.015, error.item()
    assert (storage[:8] == -123).all()
    assert (storage[-8:] == -123).all()


def _expected_minmax_summaries(
    q_ref: torch.Tensor,
    k_ref: torch.Tensor,
    sparse_key_blocks: int,
    lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, sequence, head_dim = q_ref.shape
    kv_heads = k_ref.shape[1]
    storage_tiles = (sequence + 63) // 64
    if lengths is not None:
        mask = (torch.arange(64, device="cuda")[None, :] < lengths[:, None]).view(
            1, 1, storage_tiles, 64, 1
        )
        q_tiles = q_ref.float().reshape(batch, heads, storage_tiles, 64, head_dim)
        k_tiles = k_ref.float()[:, :, : sparse_key_blocks * 64].reshape(
            batch, kv_heads, sparse_key_blocks, 64, head_dim
        )
        k_mask = mask[:, :, :sparse_key_blocks]
        return (
            torch.where(mask, q_tiles, float("-inf")).amax(dim=3)
            + torch.where(mask, q_tiles, float("inf")).amin(dim=3),
            torch.where(k_mask, k_tiles, float("-inf")).amax(dim=3),
            torch.where(k_mask, k_tiles, float("inf")).amin(dim=3),
        )
    q_padded = torch.zeros(
        (batch, heads, storage_tiles * 64, head_dim), device="cuda", dtype=torch.float32
    )
    q_padded[:, :, :sequence] = q_ref.float()
    q_valid = (torch.arange(storage_tiles * 64, device="cuda") < sequence).view(
        1, 1, storage_tiles, 64, 1
    )
    q_tiles = q_padded.reshape(batch, heads, storage_tiles, 64, head_dim)
    k_tiles = k_ref.float()[:, :, : sparse_key_blocks * 64].reshape(
        batch, kv_heads, sparse_key_blocks, 64, head_dim
    )
    return (
        torch.where(q_valid, q_tiles, float("-inf")).amax(dim=3)
        + torch.where(q_valid, q_tiles, float("inf")).amin(dim=3),
        k_tiles.amax(dim=3),
        k_tiles.amin(dim=3),
    )


def _make_sparse_operand(
    batch: int,
    head_count: int,
    sequence: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    layout: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if layout == "bhsd":
        return torch.randn(
            (batch, head_count, sequence, head_dim),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
    if layout == "bshd":
        return torch.randn(
            (batch, sequence, head_count, head_dim),
            device="cuda",
            dtype=dtype,
            generator=generator,
        ).transpose(1, 2)
    stride_elem = head_dim + 1
    flat = torch.randn(
        (1 + batch * sequence * head_count * stride_elem,),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    operand = (
        flat[1:].view(batch, sequence, head_count, stride_elem)[..., :head_dim].transpose(1, 2)
    )
    assert (operand.data_ptr() % 16 != 0) or any(
        (stride * operand.element_size()) % 16 != 0 for stride in operand.stride()[:3]
    )
    return operand


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("layout", ["bhsd", "bshd", "unaligned"])
def test_amd_prepare_operands_edge_shapes_summaries_and_attention(
    head_dim,
    dtype,
    padded,
    layout,
):
    generator = torch.Generator(device="cuda").manual_seed(3101 + head_dim + int(padded))
    sequence = 16 * 64 if padded else 16 * 64 + 17
    storage_tiles = (sequence + 63) // 64
    sparse_key_blocks = 8 if padded else 9
    # Cover multi-batch GQA/MQA, odd head counts, and partial last tiles.
    batch, heads, kv_heads = (2, 3, 1) if head_dim == 64 else (1, 6, 3)
    q = _make_sparse_operand(
        batch, heads, sequence, head_dim, dtype=dtype, layout=layout, generator=generator
    )
    k = _make_sparse_operand(
        batch, kv_heads, sequence, head_dim, dtype=dtype, layout=layout, generator=generator
    )
    v = _make_sparse_operand(
        batch, kv_heads, sequence, head_dim, dtype=dtype, layout=layout, generator=generator
    )
    k.add_(3.0)
    lengths = None
    q_ref, k_ref, v_ref = q.clone(), k.clone(), v.clone()
    if padded:
        lengths = torch.tensor(
            ([64, 17, 1, 32] * 5)[:storage_tiles], device="cuda", dtype=torch.int32
        )
        valid = (torch.arange(64, device="cuda")[None, :] < lengths[:, None]).reshape(
            1, 1, sequence, 1
        )
        for operand in (q, k, v):
            operand.masked_fill_(~valid, float("nan"))
        q_ref, k_ref, v_ref = (torch.where(valid, t, 0) for t in (q_ref, k_ref, v_ref))

    backend = require_attention_backend(q)
    opts = {
        "sparse_key_blocks": sparse_key_blocks,
        "combined_key": k,
        "combined_value": v,
        "block_lengths": lengths,
    }
    ops_sum = backend.prepare(q, head_dim**-0.5, emit_summaries=True, **opts)
    ops_no_sum = backend.prepare(q, head_dim**-0.5, emit_summaries=False, **opts)
    assert ops_sum.value_format is ValueFormat.AMD_WMMA
    assert ops_no_sum.query_summary is None
    assert ops_no_sum.key_summary is None
    assert ops_no_sum.key_aux is None

    for field in (
        "query",
        "query_scale",
        "key",
        "key_scale",
        "key_center",
        "key_center_scale",
        "value",
        "value_scale_multiplier",
        "value_mean",
    ):
        torch.testing.assert_close(
            getattr(ops_sum, field), getattr(ops_no_sum, field), atol=0, rtol=0, msg=field
        )

    assert ops_sum.key_summary is not None
    assert ops_sum.key_summary.shape == (batch, kv_heads, sparse_key_blocks, head_dim)
    if not padded:
        assert torch.count_nonzero(ops_sum.query[:, :, sequence:]) == 0
        assert torch.count_nonzero(ops_sum.key[:, :, sequence:]) == 0
        assert (ops_sum.query_scale[:, :, -1] == 0).all()

    exp_q_sum, exp_k_sum, exp_k_aux = _expected_minmax_summaries(
        q_ref, k_ref, sparse_key_blocks, lengths
    )
    torch.testing.assert_close(ops_sum.query_summary, exp_q_sum, atol=0, rtol=0)
    torch.testing.assert_close(ops_sum.key_summary, exp_k_sum, atol=0, rtol=0)
    torch.testing.assert_close(ops_sum.key_aux, exp_k_aux, atol=0, rtol=0)

    keep_ratios = tuple((0.25, 0.5, 1.0)[i % 3] for i in range(heads))
    route_layout = _resolve_route_layout(
        _normalize_head_keep_ratios(keep_ratios), sparse_key_blocks, q.device
    )
    routes = packed_routes_from_sequences(
        q_ref, k_ref[:, :, : sparse_key_blocks * 64], route_layout, _MINMAX_ROUTING, lengths
    )
    prepared = ops_sum.with_routes(
        routes.indices,
        routes.head_keep_blocks,
        routes.route_head_offsets,
        bind=backend.bind_context,
    )
    output = torch.empty(
        (batch, sequence, heads, head_dim), device=q.device, dtype=dtype
    ).transpose(1, 2)
    prepared.launch(prepared.query, output)
    expected = reference_sparse_piper_attention(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        routes,
        sparse_key_blocks=sparse_key_blocks,
        scale=head_dim**-0.5,
        block_lengths=lengths,
    )
    actual = output.transpose(1, 2)
    if padded:
        valid_out = (torch.arange(64, device="cuda")[None, :] < lengths[:, None]).reshape(
            1, sequence, 1, 1
        )
        actual, expected = torch.where(valid_out, actual, 0), torch.where(valid_out, expected, 0)
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < 0.015, error.item()


def test_amd_prepare_operands_empty_batch_and_single_row_tail():
    backend = require_attention_backend(
        torch.empty((1, 2, 64, 64), device="cuda", dtype=torch.bfloat16)
    )
    empty_q = torch.empty((0, 3, 65, 64), device="cuda", dtype=torch.bfloat16)
    empty_kv = torch.empty((0, 1, 65, 64), device="cuda", dtype=torch.bfloat16)
    empty_ops = backend.prepare(
        empty_q,
        64**-0.5,
        sparse_key_blocks=1,
        combined_key=empty_kv,
        combined_value=empty_kv,
        emit_summaries=True,
    )
    assert empty_ops.query.shape == (0, 3, 128, 64)
    assert empty_ops.key_summary is not None
    assert empty_ops.key_summary.shape == (0, 1, 1, 64)

    generator = torch.Generator(device="cuda").manual_seed(6565)
    q = torch.randn((1, 3, 65, 64), device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn((1, 1, 65, 64), device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn((1, 1, 65, 64), device="cuda", dtype=torch.bfloat16, generator=generator)
    ops = backend.prepare(
        q, 64**-0.5, sparse_key_blocks=1, combined_key=k, combined_value=v, emit_summaries=True
    )
    assert torch.count_nonzero(ops.query[:, :, 65:]) == 0
    assert torch.count_nonzero(ops.key[:, :, 65:]) == 0
    assert (ops.query_scale[:, :, -1] == 0).all()
    assert torch.isfinite(ops.key_scale).all()
    assert torch.isfinite(ops.key_center_scale).all()
    assert torch.isfinite(ops.value_mean).all()

    # Also verify a partial tile crossing into the second Q32 group (seq=1057 -> 33 remainder rows).
    q_long = torch.randn(
        (1, 3, 1057, 128), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k_long = torch.randn(
        (1, 1, 1057, 128), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    v_long = torch.randn(
        (1, 1, 1057, 128), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    ops_long = backend.prepare(
        q_long,
        128**-0.5,
        sparse_key_blocks=16,
        combined_key=k_long,
        combined_value=v_long,
        emit_summaries=True,
    )
    assert torch.count_nonzero(ops_long.query[:, :, 1057:]) == 0
    assert torch.count_nonzero(ops_long.key[:, :, 1057:]) == 0
    assert (ops_long.query_scale[:, :, -1] > 0).all()
