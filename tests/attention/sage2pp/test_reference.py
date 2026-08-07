"""Tests for the portable SageAttention2++ reference."""

import pytest
import torch

from piper_kernels.attention._sage2pp.reference import reference_sage_attention_2pp


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_is_close_to_exact_attention(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
) -> None:
    torch.manual_seed(41)
    query = torch.randn(1, 2, 65, head_dim, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual = reference_sage_attention_2pp(query, key, value, head_dim**-0.5, is_causal)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=is_causal,
    )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.01
    assert error.max().item() < 0.2


def test_reference_supports_rectangular_noncausal_attention() -> None:
    torch.manual_seed(42)
    query = torch.randn(2, 1, 17, 64, dtype=torch.bfloat16)
    key = torch.randn(2, 1, 29, 64, dtype=torch.bfloat16)
    value = torch.randn_like(key)

    actual = reference_sage_attention_2pp(query, key, value, 0.2, False)

    assert actual.shape == query.shape
    assert actual.dtype is torch.bfloat16
    assert torch.isfinite(actual).all()
