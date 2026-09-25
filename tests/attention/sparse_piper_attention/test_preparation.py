"""Shared preparation preserves quantization, routing, and padding semantics."""

import sys
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import _backend, _routing_modes


@pytest.mark.parametrize(
    ("target", "head_dim", "rows", "fused"),
    [
        (AcceleratorTarget("cuda", "sm120"), 64, 1024, False),
        (AcceleratorTarget("cuda", "sm120"), 64, 2048, True),
        (AcceleratorTarget("cuda", "sm120"), 128, 512, False),
        (AcceleratorTarget("cuda", "sm120"), 128, 1024, True),
        (AcceleratorTarget("cuda", "sm120"), 128, 100000, True),
        (AcceleratorTarget("hip", "gfx1200"), 64, 256, True),
        (AcceleratorTarget("hip", "gfx1201"), 128, 256, True),
        (AcceleratorTarget("hip", "gfx942"), 128, 4096, False),
        (AcceleratorTarget("cpu"), 128, 4096, False),
    ],
)
def test_fused_preparation_selection(monkeypatch, target, head_dim, rows, fused):
    if _backend.preparation is None:
        pytest.skip("requires Triton")
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    query = torch.empty(1, 4, rows, head_dim, device="meta", dtype=torch.bfloat16)
    key = torch.empty(1, 2, rows, head_dim, device="meta", dtype=torch.bfloat16)
    selected = _backend.select_fused_operand_preparation(query, key, _routing_modes._MINMAX_ROUTING)
    assert selected is (_backend.preparation._prepare_sparse_piper_operands if fused else None)
    assert (
        _backend.select_fused_operand_preparation(query, key, _routing_modes._MEAN_ROUTING) is None
    )


def test_fused_preparation_rejects_unsupported_metadata_before_probing(monkeypatch):
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    query = torch.empty(1, 2, 2048, 128, device="meta", dtype=torch.bfloat16)
    for key in (query.float(), query[..., ::2], torch.empty_like(query, device="cpu")):
        assert (
            _backend.select_fused_operand_preparation(query, key, _routing_modes._MINMAX_ROUTING)
            is None
        )


@pytest.fixture
def preparation():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from piper_kernels.attention.sparse_piper_attention import (  # noqa: PLC0415
        triton as preparation,
    )

    return preparation


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("padded", [False, True])
def test_fused_preparation_matches_separate_passes(preparation, head_dim, dtype, padded):
    from piper_kernels.attention.kernels.qk_quantization.int8.sage import (  # noqa: PLC0415
        triton as qk_quantization,
    )
    from piper_kernels.attention.piper_attention import _quantization  # noqa: PLC0415
    from piper_kernels.attention.sparse_piper_attention import _summaries_triton  # noqa: PLC0415
    from piper_kernels.attention.sparse_piper_attention._block_layout import (  # noqa: PLC0415
        valid_block_rows,
    )

    # Cross a mean-reduction chunk boundary, with GQA and non-contiguous head strides.
    rows = 17 * 64 if padded else 17 * 64 + 5
    storage = (rows + 63) // 64 * 64
    generator = torch.Generator(device="cuda").manual_seed(606)
    query = (
        torch.randn(2, rows, 4, head_dim, device="cuda", generator=generator)
        .to(dtype)
        .transpose(1, 2)
    )
    key = (
        (torch.randn(2, rows, 2, head_dim, device="cuda", generator=generator) + 5)
        .to(dtype)
        .transpose(1, 2)
    )
    value = torch.randn_like(key)
    lengths = None
    masked_query, masked_key, masked_value = query, key, value
    if padded:
        lengths = torch.tensor(([64, 17, 1] * 6)[:17], device="cuda", dtype=torch.int32)
        valid = valid_block_rows(lengths).reshape(1, 1, rows, 1)
        # Invalid storage must never affect statistics, even with non-finite padding.
        query, key, value = (
            torch.where(valid, tensor, float("nan")) for tensor in (query, key, value)
        )
        masked_query, masked_key, masked_value = (
            torch.where(valid, tensor, 0) for tensor in (query, key, value)
        )
    key_mean, value_mean = _quantization.compute_kv_means(masked_key, masked_value, is_causal=False)
    actual_means = _quantization.compute_kv_means(
        key, value, is_causal=False, block_lengths=lengths
    )
    for actual, expected in zip(actual_means, (key_mean, value_mean), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_qk = qk_quantization.prepare_query_key(
        masked_query,
        masked_key,
        key_mean,
        head_dim**-0.5,
        grouped=True,
        storage_key_length=storage,
        storage_query_length=storage,
    )
    expected_summaries = _summaries_triton.sequence_block_summaries(
        masked_query,
        masked_key,
        _routing_modes._MINMAX_ROUTING,
        lengths,
    )
    options = {
        "sparse_key_blocks": rows // 64,
        "combined_key": key,
        "combined_value": value,
        "block_lengths": lengths,
    }
    fused = preparation._prepare_sparse_piper_operands(
        query, head_dim**-0.5, emit_summaries=True, **options
    )
    separate = preparation._prepare_sparse_piper_operands(query, head_dim**-0.5, **options)
    for name in ("query", "key", "query_scale", "key_scale"):
        torch.testing.assert_close(
            getattr(fused, name), getattr(expected_qk, name), atol=0, rtol=0, msg=name
        )
    for name in ("value", "value_mean", "value_scale_multiplier"):
        torch.testing.assert_close(
            getattr(fused, name), getattr(separate, name), atol=0, rtol=0, msg=name
        )
    for name, expected in zip(
        ("query_summary", "key_summary", "key_aux"), expected_summaries, strict=True
    ):
        torch.testing.assert_close(getattr(fused, name), expected, atol=0, rtol=0, msg=name)
    torch.testing.assert_close(fused.value_mean, value_mean, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("padded", [False, True])
def test_public_fused_preparation_matches_separate_and_graph(
    preparation, monkeypatch, head_dim, padded
):
    from piper_kernels import SparsePiperAttention  # noqa: PLC0415

    rows = 2048 if padded else 2051
    query = torch.randn(1, rows, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, rows, 2, head_dim, device="cuda", dtype=torch.bfloat16) + 4
    value = torch.randn_like(key)
    if _backend.select_attention_backend(query) is None:
        pytest.skip("requires native sparse attention")
    assert (
        _backend.select_fused_operand_preparation(
            query.transpose(1, 2), key.transpose(1, 2), _routing_modes._MINMAX_ROUTING
        )
        is not None
    )
    lengths = (
        torch.tensor([64, 17, 1, 64] * 8, device="cuda", dtype=torch.int32) if padded else None
    )
    # Mix per-head budgets and include a dense suffix.
    attention = SparsePiperAttention((0.25, 0.5, 1.0, 0.25))
    options = {"sparse_key_blocks": 24, "block_lengths": lengths, "sparse_query_blocks": 16}
    fused = attention(query, key, value, **options)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attention(query, key, value, **options)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = attention(query, key, value, **options)
    graph.replay()
    torch.testing.assert_close(captured, fused, atol=0, rtol=0)
    monkeypatch.setattr(_backend, "select_fused_operand_preparation", lambda *args: None)
    separate = attention(query, key, value, **options)
    torch.testing.assert_close(fused, separate, atol=0, rtol=0)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.skipif(sys.platform != "linux", reason="ROCm compilation is Linux-only")
def test_fused_preparation_compiles_for_rdna4(head_dim, architecture, padded):
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget  # noqa: PLC0415
    from triton.compiler import ASTSource  # noqa: PLC0415

    from piper_kernels.attention.sparse_piper_attention import (  # noqa: PLC0415
        triton as preparation,
    )

    for kernel in (
        preparation._quantize_query_with_summary_kernel,
        preparation._quantize_key_with_summary_kernel,
    ):
        constants = {"mask_block_lengths": padded, "head_dim": head_dim}
        if kernel is preparation._quantize_query_with_summary_kernel:
            constants.update(
                softmax_scale=head_dim**-0.5, block_m=64, scale_rows=preparation.QUERY_SCALE_ROWS
            )
        else:
            constants.update(block_n=64)
        signature = {name: "i32" for name in kernel.arg_names if name not in constants}
        signature.update({name: "*fp32" for name in signature if name.endswith("_ptr")})
        signature.update({name: "*i8" for name in signature if name.endswith("_int8_ptr")})
        signature["block_lengths_ptr"] = "*i32"
        signature["query_ptr" if "query_ptr" in signature else "key_ptr"] = "*bf16"
        compiled = triton.compile(
            ASTSource(kernel, signature, constexprs=constants),
            target=GPUTarget("hip", architecture, 32),
            options={"num_warps": preparation._FUSED_QK_WARPS[head_dim]},
        )
        assert compiled.asm["hsaco"]


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fused_qk_compilation_is_reused_across_sequence_shapes(preparation, head_dim):
    kernels = (
        preparation._quantize_query_with_summary_kernel,
        preparation._quantize_key_with_summary_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()
    for batch, heads, rows in ((1, 4, 1024), (2, 8, 2048), (1, 4, 4096)):
        query = torch.zeros(
            batch, rows, heads, head_dim, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
        key = torch.zeros(
            batch, rows, heads // 2, head_dim, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
        key_mean = torch.zeros(batch, heads // 2, head_dim, device="cuda")
        prepared, *summaries = preparation._prepare_query_key_with_summaries(
            query,
            key,
            key_mean,
            head_dim**-0.5,
            storage_sequence_length=rows,
            block_lengths=None,
        )
        # The new producer must retain the all-zero quantization guards.
        assert torch.count_nonzero(prepared.query) == 0
        assert torch.count_nonzero(prepared.key) == 0
        assert torch.isfinite(prepared.query_scale).all()
        assert torch.isfinite(prepared.key_scale).all()
        assert all(torch.count_nonzero(summary) == 0 for summary in summaries)
    assert all(
        sum(len(cache[0]) for cache in kernel.device_caches.values()) == 1 for kernel in kernels
    )
