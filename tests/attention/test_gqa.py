"""Shared K/V heads must preserve per-query-head attention and routing."""

from dataclasses import replace

import pytest
import torch
from _compile_capture import TargetCapturePass
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels import SparsePiperAttention, piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._nvidia.triton import (
    _default_piper_attention_execution_plan,
    _prepare_piper_attention,
    _run_piper_attention,
)
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._budget import _resolve_route_layout
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention._scores_triton import minmax_scores


def _sm120_available():
    # ROCm can also report capability (12, 0); include the accelerator backend.
    return torch.cuda.is_available() and AcceleratorTarget.from_device(
        torch.device("cuda")
    ).is_cuda_capability(12, 0)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    return request.param


@pytest.mark.parametrize(("kv_heads", "groups"), [(1, 4), (2, 3), (2, 1)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_dense_gqa_matches_repeated_kv(device, kv_heads, groups, head_dim, causal):
    torch.manual_seed(814)
    query = torch.randn(2, kv_heads * groups, 65, head_dim, device=device, dtype=torch.float16)
    key = torch.randn(
        2, kv_heads, 65 if causal else 131, head_dim, device=device, dtype=query.dtype
    )
    value = torch.randn_like(key)
    actual = piper_attention(query, key, value, is_causal=causal)
    expected = piper_attention(
        query,
        key.repeat_interleave(groups, 1),
        value.repeat_interleave(groups, 1),
        is_causal=causal,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("kv_heads", "groups"), [(1, 4), (2, 3)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("routing", ["mean", "minmax"])
@pytest.mark.parametrize("layout", ["ragged", "padded", "full_keep"])
def test_sparse_gqa_matches_repeated_kv(device, kv_heads, groups, head_dim, routing, layout):
    torch.manual_seed(815)
    heads = kv_heads * groups
    sequence = 256 if layout == "padded" else 227
    query = torch.randn(2, sequence, heads, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(2, sequence, kv_heads, head_dim, device=device, dtype=query.dtype)
    value = torch.randn_like(key)
    # Different query heads sharing one KV head must retain independent routing budgets.
    ratios = [1.0] * heads if layout == "full_keep" else [0.34, 0.67, 1.0] * heads
    attention = SparsePiperAttention(ratios[:heads], routing=routing)
    lengths = (
        torch.tensor([43, 64, 17, 35], dtype=torch.int32, device=device)
        if layout == "padded"
        else None
    )
    kwargs = {"sparse_key_blocks": 3, "sparse_query_blocks": 2, "block_lengths": lengths}
    actual = attention(query, key, value, **kwargs)
    expected = attention(
        query, key.repeat_interleave(groups, 2), value.repeat_interleave(groups, 2), **kwargs
    )
    if lengths is not None:
        valid = (torch.arange(64, device=device)[None] < lengths[:, None]).flatten()
        actual, expected = actual[:, valid], expected[:, valid]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("query_heads", "key_heads", "value_heads"),
    [(3, 2, 2), (2, 4, 4), (4, 2, 1), (4, 0, 0), (0, 2, 2)],
)
@pytest.mark.parametrize("sparse", [False, True])
def test_gqa_invalid_heads_rejected_without_tensor_contents(
    query_heads, key_heads, value_heads, sparse
):
    with FakeTensorMode():
        tensors = [
            torch.empty(1, heads, 128, 64, dtype=torch.bfloat16)
            for heads in (query_heads, key_heads, value_heads)
        ]
        if sparse:
            attention = SparsePiperAttention([0.5] * max(query_heads, 1))
            with pytest.raises(ValueError, match="head"):
                attention(*(x.transpose(1, 2) for x in tensors), sparse_key_blocks=2)
        else:
            with pytest.raises(ValueError, match="head"):
                piper_attention(*tensors)


@pytest.mark.gpu
@pytest.mark.skipif(not _sm120_available(), reason="requires NVIDIA SM120")
@pytest.mark.parametrize("query_length", [256, 257])
def test_dense_gqa_compile_and_prepared_storage(query_length):
    # Keep semantic scalars inside the graph, as in an architecture processor.
    def dense_attention(query, key, value):
        return piper_attention(query, key, value, scale=128**-0.5)

    query = torch.randn(2, 8, query_length, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, 256, 128, device="cuda", dtype=query.dtype)
    value = torch.randn_like(key)
    dense = _prepare_piper_attention(
        query,
        key,
        value,
        128**-0.5,
        False,
        execution_plan=_default_piper_attention_execution_plan(query, False),
    )
    assert dense.query.shape[1] == 8
    for tensor in (dense.key_scale, dense.value_scale_multiplier, dense.value_mean):
        assert tensor.shape[1] == 2
    torch.testing.assert_close(
        torch.compile(dense_attention, fullgraph=True, dynamic=True)(query, key, value),
        piper_attention(query, key, value),
        rtol=0,
        atol=0,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(("kv_heads", "groups"), [(1, 4), (2, 3)])
@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_sparse_gqa_compile_and_prepared_storage(head_dim, kv_heads, groups, routing):
    torch.manual_seed(816)
    torch._dynamo.reset()
    heads = kv_heads * groups
    query = torch.randn(2, heads, 256, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, kv_heads, 256, head_dim, device="cuda", dtype=query.dtype)
    value = torch.randn_like(key)
    backend = _backend.select_attention_backend(query)
    if backend is None:
        pytest.skip("requires a native sparse-attention backend")
    scale = head_dim**-0.5
    attention = SparsePiperAttention([0.5] * heads, routing=routing)
    layout = _resolve_route_layout(attention._head_keep_ratio_units, 4, query.device)
    routing_mode = _MEAN_ROUTING if routing == "mean" else _MINMAX_ROUTING
    routes = packed_routes_from_sequences(query, key, layout, routing_mode)
    prepared = backend.prepare(
        query,
        routes.indices,
        routes.head_keep_blocks,
        scale,
        sparse_key_blocks=4,
        route_head_offsets=routes.route_head_offsets,
        combined_key=key,
        combined_value=value,
    )
    assert prepared.query.data.shape[1] == heads
    assert prepared.context.head_keep_blocks.numel() == heads
    for tensor in (
        prepared.context.key,
        prepared.context.value,
        prepared.context.key_scale,
        prepared.context.value_scale_multiplier,
        prepared.context.value_mean,
    ):
        assert tensor.shape[1] == kv_heads
    local_output = torch.empty((2, heads, 64, head_dim), device="cuda", dtype=query.dtype)
    backend.launch(prepared, local_output, query_block_offset=1, query_block_count=1)
    expected = attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), sparse_key_blocks=4
    )
    torch.testing.assert_close(local_output, expected[:, 64:128].transpose(1, 2), rtol=0, atol=0)

    def sparse_attention(query, key, value, sparse_key_blocks):
        # Keep the structural scale constant inside the graph, as in the dense test.
        return attention(
            query,
            key,
            value,
            sparse_key_blocks=sparse_key_blocks,
            scale=64**-0.5 if head_dim == 64 else 128**-0.5,
        )

    capture = TargetCapturePass()
    compiled = torch.compile(
        sparse_attention,
        fullgraph=True,
        dynamic=True,
        options={"post_grad_custom_pre_pass": capture},
    )
    with torch.inference_mode():
        for sequence, sparse_key_blocks in ((193, 2), (257, 3), (320, 4)):
            query = torch.randn(2, sequence, heads, head_dim, device="cuda", dtype=query.dtype)
            key = torch.randn(2, sequence, kv_heads, head_dim, device="cuda", dtype=query.dtype)
            value = torch.randn_like(key)
            torch.testing.assert_close(
                compiled(query, key, value, sparse_key_blocks),
                attention(query, key, value, sparse_key_blocks=sparse_key_blocks),
                rtol=0,
                atol=0,
            )
    assert capture.calls == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _sm120_available(), reason="requires NVIDIA SM120")
@pytest.mark.parametrize("causal", [False, True])
def test_dense_gqa_per_thread_preparation(causal):
    query = torch.randn(2, 6, 67, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, 67, 128, device="cuda", dtype=query.dtype)
    value = torch.randn_like(key)
    plan = replace(
        _default_piper_attention_execution_plan(query, causal),
        grouped_qk=False,
        use_tensor_descriptors=False,
    )
    actual = _run_piper_attention(query, key, value, 128**-0.5, causal, execution_plan=plan)
    expected = _run_piper_attention(
        query,
        key.repeat_interleave(3, 1),
        value.repeat_interleave(3, 1),
        128**-0.5,
        causal,
        execution_plan=plan,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_gqa_minmax_score_kernel_preserves_query_heads():
    query = torch.randn(2, 6, 17, 128, device="cuda", dtype=torch.float32)
    key_min = torch.randn(2, 2, 19, 128, device="cuda", dtype=torch.float32)
    key_max = torch.randn_like(key_min)
    actual = minmax_scores(query, key_min, key_max, score_scale=0.3)
    expected = minmax_scores(
        query, key_min.repeat_interleave(3, 1), key_max.repeat_interleave(3, 1), score_scale=0.3
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
