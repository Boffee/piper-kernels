"""Shared FP32 RMSNorm/RoPE has a working portable device-math default."""

import pytest
import torch
import triton
import triton.language as tl

from piper_kernels.fusions.projected_qk.triton import rmsnorm_rope_tile


@triton.jit
def _transform(
    input_ptr,
    norm_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    rows: tl.constexpr,
    rotary_dim: tl.constexpr,
    head_dim: tl.constexpr,
):
    row = tl.arange(0, 64)
    column = tl.arange(0, 2 * head_dim)
    offsets = row[:, None] * (2 * head_dim) + column[None, :]
    projection = tl.reshape(tl.load(input_ptr + offsets, row[:, None] < rows, 0), (64, 2, head_dim))
    transformed = rmsnorm_rope_tile(
        projection,
        norm_ptr,
        cos_ptr,
        sin_ptr,
        row,
        rows,
        2,
        head_dim,
        rotary_dim,
        1e-6,
        True,
        64,
    )
    tl.store(output_ptr + offsets, tl.reshape(transformed, (64, 2 * head_dim)), row[:, None] < rows)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("rows", [1, 63, 64])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("rotary_fraction", [0, 0.75, 1])
@pytest.mark.parametrize("affine", [True, False])
def test_portable_rmsnorm_rope_stays_fp32_and_matches_fp64(rows, head_dim, rotary_fraction, affine):
    rotary_dim = max(2, int(head_dim * rotary_fraction))
    generator = torch.Generator(device="cuda").manual_seed(336)
    projection = torch.randn((rows, 2, head_dim), device="cuda", generator=generator)
    projection *= torch.logspace(-5, 4, rows, device="cuda")[:, None, None]
    projection[0, 0] = 0
    norm = torch.rand(head_dim, dtype=torch.bfloat16, device="cuda", generator=generator) + 0.5
    if not affine:
        norm = None
    angle = torch.randn((rows, rotary_dim), device="cuda", generator=generator)
    cos, sin = angle.cos(), angle.sin()
    actual = torch.empty_like(projection)
    _transform[(1,)](projection, norm, cos, sin, actual, rows, rotary_dim, head_dim, num_warps=8)
    normalized = projection.double() * torch.rsqrt(
        projection.double().square().mean(-1, keepdim=True) + 1e-6
    )
    if norm is not None:
        normalized *= norm.double()
    half = rotary_dim // 2
    rotated = torch.cat((-normalized[..., half:rotary_dim], normalized[..., :half]), dim=-1)
    expected = normalized.clone()
    expected[..., :rotary_dim] = (
        normalized[..., :rotary_dim] * cos.double()[:, None, :] + rotated * sin.double()[:, None, :]
    )
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=3e-6, atol=2e-6)
