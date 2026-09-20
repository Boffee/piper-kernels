"""RDNA4 dense attention semantics, preparation, and compiler integration."""

import pytest
import torch
from _compile_capture import TargetCapturePass

from piper_kernels import piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import dispatch
from piper_kernels.attention.piper_attention._amd import gluon as backend
from piper_kernels.attention.piper_attention._amd import triton as preparation
from piper_kernels.attention.piper_attention._amd.policy import supports_target
from piper_kernels.attention.piper_attention.reference import reference_piper_attention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not supports_target(AcceleratorTarget.from_device(torch.device("cuda"))),
        reason="requires Linux RDNA4 ROCm",
    ),
]


@pytest.fixture(autouse=True)
def require_native(monkeypatch):
    def no_fallback(*args, **kwargs):
        pytest.fail("RDNA4 dense attention unexpectedly used the portable fallback")

    monkeypatch.setattr(dispatch, "reference_piper_attention", no_fallback)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("length", [1, 31, 63, 64, 65, 127, 193, 257])
def test_matches_quantized_reference(head_dim, dtype, causal, length):
    torch.manual_seed(991 + length)
    query = torch.randn(2, 3, length, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(2, 1, length if causal else length + 38, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    actual = piper_attention(query, key, value, is_causal=causal, scale=0.17)
    expected = reference_piper_attention(
        query,
        key,
        value,
        0.17,
        causal,
        qk_quantization="per_warp",
    )
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.006)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kind", ["zero", "constant", "varying_scale", "strided"])
def test_value_semantics_and_strides(head_dim, causal, kind):
    torch.manual_seed(992)
    inputs = [
        torch.randn(2, 131, 3, head_dim, device="cuda", dtype=torch.float16).transpose(1, 2)
        for _ in range(3)
    ]
    query, key, value = inputs
    if kind == "zero":
        query.zero_()
        key.zero_()
        value.zero_()
    elif kind == "constant":
        value.copy_(value[:, :, :1].clone())
    elif kind == "varying_scale":
        value *= torch.exp2(torch.linspace(-8, 8, 131, device="cuda"))[None, None, :, None]
    actual = piper_attention(query, key, value, is_causal=causal)
    expected = reference_piper_attention(
        query,
        key,
        value,
        head_dim**-0.5,
        causal,
        qk_quantization="per_warp",
    )
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)
    assert torch.isfinite(actual).all()
    assert error < 0.005, error.item()
    if kind == "zero":
        assert torch.count_nonzero(actual) == 0
    if kind == "constant" and not causal:
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("zero_operand", ["query", "key"])
@pytest.mark.parametrize("mixed", [False, True])
def test_zero_scores_with_nonzero_values_at_tile_boundaries(
    head_dim, dtype, causal, zero_operand, mixed
):
    """Masked tails must not evaluate zero QK scales times negative infinity."""
    torch.manual_seed(995)
    query = torch.randn(1, 6, 193, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(1, 2, 193 if causal else 231, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    if zero_operand == "query":
        # A whole Q32 scale group exercises zero and nonzero scales in one Q64 tile.
        (query[:, 0, :32] if mixed else query).zero_()
    else:
        (key[:, 0] if mixed else key).zero_()
    actual = piper_attention(query, key, value, is_causal=causal)
    expected = reference_piper_attention(
        query,
        key,
        value,
        head_dim**-0.5,
        causal,
        qk_quantization="per_warp",
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.006)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_preparation_retains_per_token_scales_and_compact_gqa_storage(head_dim, causal):
    torch.manual_seed(993)
    query = torch.randn(2, 6, 65, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, 65, head_dim, device="cuda", dtype=query.dtype)
    value = torch.randn_like(key)
    prepared = backend.prepare_attention(query, key, value, head_dim**-0.5, causal)
    assert prepared.query.shape == (2, 6, 128, head_dim)
    for tensor in (
        prepared.key,
        prepared.value,
        prepared.key_scale,
        prepared.multiplier,
        prepared.log_scale,
    ):
        assert tensor.shape[1] == 2
    centered = value.float() if causal else value.float() - value.float().mean(2, keepdim=True)
    scale = centered.abs().amax(-1) / 127 + 1e-7
    torch.testing.assert_close(prepared.multiplier[..., :65], scale * 255, rtol=2e-6, atol=1e-7)
    assert prepared.multiplier.shape == (2, 2, 128)
    tokens = torch.arange(64, device="cuda")
    packed_tokens = (tokens & ~24) | ((tokens & 8) << 1) | ((tokens & 16) >> 1)
    unpacked = prepared.value[..., packed_tokens].transpose(-1, -2).reshape(2, 2, 128, head_dim)
    normalized = centered / scale[..., None]
    expected = (
        (normalized + torch.where(normalized >= 0, 0.5, -0.5)).clamp(-127, 127).to(torch.int8)
    )
    torch.testing.assert_close(unpacked[..., :65, :], expected, atol=0, rtol=0)
    assert torch.count_nonzero(unpacked[..., 65:, :]) == 0
    torch.testing.assert_close(
        backend.launch_attention(prepared),
        piper_attention(query, key, value, is_causal=causal),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_causal_outputs_do_not_depend_on_future_values(head_dim):
    torch.manual_seed(994)
    query, key, value = [
        torch.randn(1, 2, 129, head_dim, device="cuda", dtype=torch.float16) for _ in range(3)
    ]
    expected = piper_attention(query, key, value, is_causal=True)
    value[:, :, 47:] *= 16
    actual = piper_attention(query, key, value, is_causal=True)
    torch.testing.assert_close(actual[:, :, :47], expected[:, :, :47], atol=0, rtol=0)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_dynamic_fullgraph_and_live_graph_capture(head_dim, causal):
    torch._dynamo.reset()
    capture = TargetCapturePass()

    def attention(query, key, value):
        # Keep the semantic scale constant, as in the existing dense GQA tests.
        return piper_attention(
            query,
            key,
            value,
            is_causal=causal,
            scale=64**-0.5 if head_dim == 64 else 128**-0.5,
        )

    compiled = torch.compile(
        attention, fullgraph=True, dynamic=True, options={"post_grad_custom_pre_pass": capture}
    )
    with torch.inference_mode():
        for length in (65, 193, 257):
            query = torch.randn(2, 6, length, head_dim, device="cuda", dtype=torch.bfloat16)
            key = torch.randn(
                2, 2, length if causal else length + 12, head_dim, device="cuda", dtype=query.dtype
            )
            value = torch.randn_like(key)
            torch.testing.assert_close(
                compiled(query, key, value), attention(query, key, value), atol=0, rtol=0
            )
        assert capture.calls == 1
        # Capture the complete public operation, including fresh preparation.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = attention(query, key, value)
        value.mul_(0.25).add_(3)
        graph.replay()
        torch.testing.assert_close(output, attention(query, key, value), atol=0, rtol=0)


def test_dynamic_lengths_and_equivalent_gqa_shapes_reuse_kernel_specializations():
    backend._dense_piper_kernel.device_caches.clear()
    preparation._prepare_value_kernel.device_caches.clear()

    with torch.inference_mode():
        for query_length, key_length, heads, kv_heads in (
            (65, 77, 6, 2),
            (193, 205, 6, 2),
            (257, 269, 6, 2),
            (193, 205, 3, 1),
            (1, 1, 3, 1),
        ):
            query = torch.randn(1, heads, query_length, 64, device="cuda", dtype=torch.bfloat16)
            key = torch.randn(1, kv_heads, key_length, 64, device="cuda", dtype=query.dtype)
            piper_attention(query, key, torch.randn_like(key))
        torch.cuda.synchronize()

    assert sum(len(cache[0]) for cache in backend._dense_piper_kernel.device_caches.values()) == 1
    assert (
        sum(len(cache[0]) for cache in preparation._prepare_value_kernel.device_caches.values())
        == 1
    )


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quality_against_sdpa(head_dim, causal, dtype):
    torch.manual_seed(995)
    inputs = [torch.randn(1, 2, 1025, head_dim, device="cuda", dtype=dtype) for _ in range(3)]
    actual = piper_attention(*inputs, is_causal=causal).float()
    expected = torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=causal).float()
    relative_error = (actual - expected).norm() / expected.norm()
    assert relative_error < 0.03, relative_error.item()


def test_empty_batch():
    query = torch.empty(0, 2, 3, 64, device="cuda", dtype=torch.float16)
    output = piper_attention(query, query, query)
    assert output.shape == query.shape
    assert output.is_contiguous()
