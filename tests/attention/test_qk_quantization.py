"""Tests for shared Sage-style Q/K reference preparation."""

import pytest
import torch

from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    quantize_query_key,
)


@pytest.mark.parametrize("granularity", ["per_thread", "per_warp"])
def test_quantize_query_key_centers_key_and_returns_per_row_scales(
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

    key_centered = key.float() - key.float().mean(dim=2, keepdim=True)
    query_error = (query_int8.float() * query_scale[..., None] - query.float()).abs()
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
