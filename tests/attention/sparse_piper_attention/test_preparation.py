"""Shared preparation preserves quantization, routing, and padding semantics."""

import re
import sys
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels._grid import select_block_grid
from piper_kernels.attention.sparse_piper_attention import _backend, _routing_modes


def test_select_block_grid_respects_strides_and_cuda_grid_y_limit():
    sequence_major = torch.empty((2, 128, 8, 64), device="meta").transpose(1, 2)
    head_major = torch.empty((2, 8, 128, 64), device="meta")
    single_head = torch.empty((2, 128, 1, 64), device="meta").transpose(1, 2)

    assert select_block_grid(sequence_major, 65_535) == ((8, 65_535, 2), 1)
    assert select_block_grid(sequence_major, 65_536) == ((65_536, 8, 2), 0)
    assert select_block_grid(head_major, 64) == ((64, 8, 2), 0)
    assert select_block_grid(single_head, 64) == ((64, 1, 2), 0)


@pytest.mark.parametrize(
    ("target", "head_dim", "rows", "fused"),
    [
        (AcceleratorTarget("cuda", "sm120"), 64, 1024, False),
        (AcceleratorTarget("cuda", "sm120"), 64, 2048, True),
        (AcceleratorTarget("cuda", "sm120"), 128, 512, False),
        (AcceleratorTarget("cuda", "sm120"), 128, 1024, True),
        (AcceleratorTarget("cuda", "sm120"), 128, 100000, True),
        (AcceleratorTarget("cuda", "sm89"), 64, 256, True),
        (AcceleratorTarget("cuda", "sm89"), 128, 256, True),
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
    selected = _backend.use_fused_operand_preparation(query, key, _routing_modes._MINMAX_ROUTING)
    assert selected is fused
    assert _backend.use_fused_operand_preparation(query, key, _routing_modes._MEAN_ROUTING) is False


def test_fused_preparation_rejects_unsupported_metadata_before_probing(monkeypatch):
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    query = torch.empty(1, 2, 2048, 128, device="meta", dtype=torch.bfloat16)
    for key in (query.float(), query[..., ::2], torch.empty_like(query, device="cpu")):
        assert (
            _backend.use_fused_operand_preparation(query, key, _routing_modes._MINMAX_ROUTING)
            is False
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
@pytest.mark.parametrize("sparse_key_blocks", [1, None])
@pytest.mark.parametrize("head_major", [False, True])
def test_fused_preparation_matches_separate_passes(
    preparation, head_dim, dtype, padded, sparse_key_blocks, head_major
):
    from piper_kernels.attention.kernels.qk_quantization.int8.sage import (  # noqa: PLC0415
        triton as qk_quantization,
    )
    from piper_kernels.attention.sparse_piper_attention import _summaries_triton  # noqa: PLC0415
    from piper_kernels.attention.sparse_piper_attention._block_layout import (  # noqa: PLC0415
        valid_block_rows,
    )
    from piper_kernels.attention.sparse_piper_attention.coarse import (  # noqa: PLC0415
        mean_pool_block_values,
    )

    # Cross a K64 tile boundary, with GQA and both sequence-major and head-major strides.
    rows = 17 * 64 if padded else 17 * 64 + 5
    storage = (rows + 63) // 64 * 64
    sparse_key_blocks = rows // 64 if sparse_key_blocks is None else sparse_key_blocks
    generator = torch.Generator(device="cuda").manual_seed(606)

    def make_operand(heads: int) -> torch.Tensor:
        if head_major:
            return torch.randn(2, heads, rows, head_dim, device="cuda", generator=generator).to(
                dtype
            )
        return (
            torch.randn(2, rows, heads, head_dim, device="cuda", generator=generator)
            .to(dtype)
            .transpose(1, 2)
        )

    query = make_operand(4)
    key = make_operand(2) + 5
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
    expected_query, expected_query_scale = qk_quantization.prepare_query(
        masked_query,
        head_dim**-0.5,
        grouped=True,
        storage_query_length=storage,
    )
    expected_key, expected_key_scale, expected_key_center, expected_key_center_scale = (
        qk_quantization.prepare_tile_centered_key(
            masked_key,
            storage_key_length=storage,
            block_lengths=lengths,
        )
    )
    expected_summaries = _summaries_triton.sequence_block_summaries(
        masked_query,
        masked_key[:, :, : sparse_key_blocks * 64],
        _routing_modes._MINMAX_ROUTING,
        lengths,
    )
    options = {
        "sparse_key_blocks": sparse_key_blocks,
        "combined_key": key,
        "combined_value": value,
        "block_lengths": lengths,
    }
    fused = preparation._prepare_sparse_piper_operands(
        query, head_dim**-0.5, emit_summaries=True, **options
    )
    separate = preparation._prepare_sparse_piper_operands(query, head_dim**-0.5, **options)
    for name, expected in (
        ("query", expected_query),
        ("key", expected_key),
        ("query_scale", expected_query_scale),
        ("key_scale", expected_key_scale),
        ("key_center", expected_key_center),
        ("key_center_scale", expected_key_center_scale),
    ):
        torch.testing.assert_close(getattr(fused, name), expected, atol=0, rtol=0, msg=name)
    for name in (
        "value",
        "value_mean",
        "value_scale_multiplier",
        "key_center",
        "key_center_scale",
    ):
        torch.testing.assert_close(
            getattr(fused, name), getattr(separate, name), atol=0, rtol=0, msg=name
        )
    for name, expected in zip(
        ("query_summary", "key_summary", "key_aux"), expected_summaries, strict=True
    ):
        torch.testing.assert_close(getattr(fused, name), expected, atol=0, rtol=0, msg=name)
    for summary in (fused.key_summary, fused.key_aux):
        assert summary.shape == (2, 2, sparse_key_blocks, head_dim)
        # A prefix view of a full-sequence allocation would retain the wasted storage.
        assert summary.untyped_storage().nbytes() == summary.numel() * summary.element_size()
    expected_v_mean = mean_pool_block_values(masked_value.transpose(1, 2), lengths).to(
        torch.bfloat16
    )
    torch.testing.assert_close(fused.value_mean, expected_v_mean, atol=1e-2, rtol=2e-2)


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
        _backend.use_fused_operand_preparation(
            query.transpose(1, 2), key.transpose(1, 2), _routing_modes._MINMAX_ROUTING
        )
        is True
    )
    lengths = (
        torch.tensor([64, 17, 1, 64] * 8, device="cuda", dtype=torch.int32) if padded else None
    )
    # Mix per-head budgets and route a small prefix before a long dense suffix.
    attention = SparsePiperAttention((0.25, 0.5, 1.0, 0.25))
    options = {"sparse_key_blocks": 4, "block_lengths": lengths, "sparse_query_blocks": 16}
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
    monkeypatch.setattr(_backend, "use_fused_operand_preparation", lambda *args: False)
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

    from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat  # noqa: PLC0415
    from piper_kernels.attention.sparse_piper_attention import (  # noqa: PLC0415
        triton as preparation,
    )

    for kernel in (
        preparation._quantize_query_with_summary_kernel,
        preparation._quantize_key_with_summary_kernel,
        preparation._quantize_value_per_tile_kernel,
    ):
        constants = {"mask_block_lengths": padded, "head_dim": head_dim}
        num_warps = preparation._FUSED_QK_WARPS[head_dim]
        if kernel is preparation._quantize_query_with_summary_kernel:
            constants.update(
                softmax_scale=head_dim**-0.5, block_m=64, scale_rows=preparation.QUERY_SCALE_ROWS
            )
        elif kernel is preparation._quantize_key_with_summary_kernel:
            constants.update(block_n=64)
        else:
            constants.update(block_n=64, value_format=int(ValueFormat.AMD_WMMA))
            num_warps = 4
        signature = {name: "i32" for name in kernel.arg_names if name not in constants}
        signature.update({name: "*fp32" for name in signature if name.endswith("_ptr")})
        signature.update({name: "*i8" for name in signature if name.endswith("_int8_ptr")})
        signature["block_lengths_ptr"] = "*i32"
        if kernel is preparation._quantize_value_per_tile_kernel:
            signature.update(
                value_ptr="*bf16",
                value_mean_ptr="*bf16",
                value_scale_ptr="*fp32",
                output_ptr="*i8",
            )
        else:
            signature["query_ptr" if "query_ptr" in signature else "key_ptr"] = "*bf16"
        compiled = triton.compile(
            ASTSource(kernel, signature, constexprs=constants),
            target=GPUTarget("hip", architecture, 32),
            options={"num_warps": num_warps},
        )
        assert compiled.asm["hsaco"]
        scratch = re.search(
            r"\.amdhsa_private_segment_fixed_size\s+(\d+)\b", compiled.asm["amdgcn"]
        )
        assert scratch is not None
        if kernel is preparation._quantize_value_per_tile_kernel:
            assert int(scratch.group(1)) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fused_qk_compilation_is_reused_across_sequence_shapes(preparation, head_dim):
    from piper_kernels.attention.kernels.qk_quantization.int8.sage import (  # noqa: PLC0415
        triton as qk_quantization,
    )
    from piper_kernels.attention.sparse_piper_attention import _summaries_triton  # noqa: PLC0415

    single_cache_kernels = (
        preparation._quantize_query_with_summary_kernel,
        preparation._quantize_key_with_summary_kernel,
        preparation._quantize_value_per_tile_kernel,
        qk_quantization.quantize_query_per_warp_kernel,
        qk_quantization.quantize_key_tile_centered_per_block_kernel,
    )
    summary_kernel = _summaries_triton._block_summary_kernel
    for kernel in (*single_cache_kernels, summary_kernel):
        kernel.device_caches.clear()
    for batch, heads, rows, sparse_key_blocks, head_major in (
        (1, 4, 1024, 16, False),
        (1, 6, 1024, 1, True),
        (1, 4, 1024, 3, False),
        (2, 8, 2048, 32, True),
        (1, 4, 4096, 5, False),
    ):
        if head_major:
            query = torch.zeros(batch, heads, rows, head_dim, device="cuda", dtype=torch.bfloat16)
            key = torch.zeros(
                batch, heads // 2, rows, head_dim, device="cuda", dtype=torch.bfloat16
            )
        else:
            query = torch.zeros(
                batch, rows, heads, head_dim, device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
            key = torch.zeros(
                batch, rows, heads // 2, head_dim, device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
        value = torch.zeros_like(key)
        prepared_ops = preparation._prepare_sparse_piper_operands(
            query,
            head_dim**-0.5,
            sparse_key_blocks=sparse_key_blocks,
            combined_key=key,
            combined_value=value,
            emit_summaries=True,
        )
        # Also exercise the separate Q/K and summary paths across layouts/head counts.
        qk_quantization.prepare_query(
            query, head_dim**-0.5, grouped=True, storage_query_length=rows
        )
        qk_quantization.prepare_tile_centered_key(key, storage_key_length=rows)
        _summaries_triton.sequence_block_summaries(query, key, _routing_modes._MINMAX_ROUTING)
        # The fused producer must retain all-zero quantization and summary guards.
        assert torch.count_nonzero(prepared_ops.query) == 0
        assert torch.count_nonzero(prepared_ops.key) == 0
        assert torch.count_nonzero(prepared_ops.key_center) == 0
        assert torch.isfinite(prepared_ops.query_scale).all()
        assert torch.isfinite(prepared_ops.key_scale).all()
        assert torch.isfinite(prepared_ops.key_center_scale).all()
        assert torch.count_nonzero(prepared_ops.query_summary) == 0
        assert torch.count_nonzero(prepared_ops.key_summary) == 0
        assert torch.count_nonzero(prepared_ops.key_aux) == 0
    assert all(
        sum(len(cache[0]) for cache in kernel.device_caches.values()) == 1
        for kernel in single_cache_kernels
    )
    assert sum(len(cache[0]) for cache in summary_kernel.device_caches.values()) == 2


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize(
    ("query_summary", "mean_pool_summary"),
    [(True, False), (False, False), (True, True), (False, True)],
)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.skipif(sys.platform != "linux", reason="ROCm compilation is Linux-only")
def test_block_summary_compiles_for_rdna4(
    head_dim, architecture, query_summary, mean_pool_summary, padded
):
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget  # noqa: PLC0415
    from triton.compiler import ASTSource  # noqa: PLC0415

    from piper_kernels.attention.sparse_piper_attention import (  # noqa: PLC0415
        _summaries_triton,
    )

    kernel = _summaries_triton._block_summary_kernel
    constants = {
        "block_rows": 64,
        "head_dim": head_dim,
        "query_summary": query_summary,
        "mean_pool_summary": mean_pool_summary,
        "mask_block_lengths": padded,
        "mask_ragged_tail": not padded,
    }
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        input_ptr="*bf16",
        output_primary_ptr="*fp32",
        output_aux_ptr="*fp32",
        block_lengths_ptr="*i32",
    )
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs=constants),
        target=GPUTarget("hip", architecture, 32),
        options={"num_warps": 4, "num_stages": 1},
    )
    assert compiled.asm["hsaco"]
    scratch = re.search(r"\.amdhsa_private_segment_fixed_size\s+(\d+)\b", compiled.asm["amdgcn"])
    assert scratch is not None
    assert int(scratch.group(1)) == 0
