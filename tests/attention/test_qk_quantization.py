"""Tests for shared Sage-style Q/K reference preparation."""

import pytest
import torch

from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import (
    SIGNED_HADAMARD_MASK,
    rotate_signed_hadamard_heads,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    quantize_query_key,
)


def _signed_hadamard(head_dim: int) -> torch.Tensor:
    signs = torch.tensor(
        [
            1.0 if SIGNED_HADAMARD_MASK[index // 32] & (1 << (index % 32)) else -1.0
            for index in range(head_dim)
        ]
    )
    hadamard = torch.tensor([[1.0]])
    while hadamard.shape[0] < head_dim:
        hadamard = torch.cat(
            (
                torch.cat((hadamard, hadamard), dim=1),
                torch.cat((hadamard, -hadamard), dim=1),
            ),
            dim=0,
        )
    return signs[:, None] * hadamard / head_dim**0.5


@pytest.mark.parametrize("head_dim", [64, 128])
def test_signed_hadamard_heads_match_fixed_orthogonal_matrix(head_dim: int) -> None:
    torch.manual_seed(71)
    value = torch.randn(2, 3, 5, head_dim)

    actual = rotate_signed_hadamard_heads(value)
    expected = value @ _signed_hadamard(head_dim)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual @ actual.transpose(-1, -2),
        value @ value.transpose(-1, -2),
        atol=3e-5,
        rtol=3e-5,
    )


def test_signed_hadamard_heads_reject_other_widths() -> None:
    with pytest.raises(ValueError, match="must be one of 64, 128"):
        rotate_signed_hadamard_heads(torch.empty(2, 32))


@pytest.mark.parametrize("granularity", ["per_thread", "per_warp"])
def test_quantize_query_key_centers_and_smooths_before_quantization(
    granularity: str,
) -> None:
    torch.manual_seed(70)
    query = torch.randn(2, 3, 35, 64, dtype=torch.float16)
    key = torch.randn(2, 3, 67, 64, dtype=torch.float16)

    query_int8, key_int8, query_scale, key_scale = quantize_query_key(
        query,
        key,
        granularity=granularity,  # type: ignore[arg-type]
    )

    query_rotated = rotate_signed_hadamard_heads(query.float())
    key_centered = key.float() - key.float().mean(dim=2, keepdim=True)
    key_centered = rotate_signed_hadamard_heads(key_centered)
    query_error = (query_int8.float() * query_scale[..., None] - query_rotated).abs()
    key_error = (key_int8.float() * key_scale[..., None] - key_centered).abs()
    assert query_int8.shape == query.shape
    assert key_int8.shape == key.shape
    assert query_scale.shape == query.shape[:3]
    assert key_scale.shape == key.shape[:3]
    assert bool((query_error <= query_scale[..., None] / 2 + 1e-6).all())
    assert bool((key_error <= key_scale[..., None] / 2 + 1e-6).all())


def test_quantize_query_key_rejects_unknown_granularity() -> None:
    value = torch.empty((1, 1, 8, 64), dtype=torch.float16)

    with pytest.raises(ValueError, match="unknown Q/K quantization granularity"):
        quantize_query_key(value, value, granularity="row")  # type: ignore[arg-type]
