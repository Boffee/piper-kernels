"""Operand stores must address head-major buffers beyond signed 32-bit offsets."""

import pytest
import torch
import triton
import triton.language as tl

from piper_kernels.attention.kernels.sparse_piper import triton as primitives


@triton.jit
def _store_last_head(data, scales, summary, means, length: tl.constexpr, kind: tl.constexpr):
    head = tl.full((1,), 55, tl.int32)
    rows = tl.arange(0, 64)
    values = tl.full((64, 1, 128), 1.0, tl.float32)
    if kind == "key":
        primitives.store_key_tile(
            values,
            data,
            scales,
            summary,
            summary,
            scales,
            0,
            56,
            head,
            rows,
            length,
            length,
            0,
            True,
            False,
            1,
            128,
            64,
            64,
        )
    elif kind == "value":
        primitives.store_value_tile(
            values,
            means,
            data,
            scales,
            summary,
            scales,
            0,
            56,
            head,
            rows,
            length,
            length,
            0,
            False,
            True,
            1,
            128,
            64,
            64,
        )
    else:
        primitives.store_query_tile(
            values,
            data,
            scales,
            summary,
            scales,
            0,
            56,
            head,
            rows,
            rows,
            length,
            length,
            0,
            0,
            1.0,
            True,
            False,
            False,
            1,
            128,
            64,
            16,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ["query", "key", "value"])
def test_store_last_head_uses_full_address(kind):
    length = 400_000
    required = 56 * length * 128 + 256 * 1024**2
    if torch.cuda.mem_get_info()[0] < required:
        pytest.skip("Not enough free device memory for the large-address regression")
    data = torch.full((56, length * 128), -7, device="cuda", dtype=torch.int8)
    scales = torch.empty((56, length // 16), device="cuda", dtype=torch.float32)
    summary = torch.empty((56, length // 64, 128), device="cuda")
    means = torch.zeros((56, 128), device="cuda")
    _store_last_head[(1,)](data, scales, summary, means, length, kind)
    torch.cuda.synchronize()
    if kind == "value":
        written = data[55].view(128, length)[:, :64]
        untouched = data[55].view(128, length)[:, 64:128]
    else:
        written = data[55, : 64 * 128]
        untouched = data[55, 64 * 128 : 128 * 128]
    reference = torch.full((56, 128 * 128), -7, device="cuda", dtype=torch.int8)
    _store_last_head[(1,)](reference, scales, summary, means, 128, kind)
    expected = (
        reference[55].view(128, 128)[:, :64] if kind == "value" else reference[55, : 64 * 128]
    )
    torch.testing.assert_close(written, expected, rtol=0, atol=0)
    assert (untouched == -7).all().item()
    assert (data[54, -128:] == -7).all().item()
