"""Tests for the portable Piper Attention reference."""

import pytest
import torch

from piper_kernels.attention.piper_attention.reference import reference_piper_attention


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_reference_is_close_to_exact_attention(
    head_dim: int,
    is_causal: bool,
) -> None:
    torch.manual_seed(51)
    query = torch.randn(1, 2, 65, head_dim, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual = reference_piper_attention(
        query,
        key,
        value,
        head_dim**-0.5,
        is_causal,
    )
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


def test_reference_centering_restores_constant_value_exactly() -> None:
    torch.manual_seed(52)
    query = torch.randn(1, 2, 65, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value_row = torch.randn(1, 2, 1, 64, dtype=torch.float16)
    value = value_row.expand_as(query).contiguous()

    actual = reference_piper_attention(
        query,
        key,
        value,
        64**-0.5,
        False,
    )

    torch.testing.assert_close(actual, value, atol=0.0, rtol=0.0)


def test_causal_reference_is_independent_of_future_value_rows() -> None:
    torch.manual_seed(53)
    query = torch.randn(1, 1, 65, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    changed_value = value.clone()
    changed_value[:, :, 32:] = torch.randn_like(changed_value[:, :, 32:]) * 32

    original = reference_piper_attention(query, key, value, 64**-0.5, True)
    changed = reference_piper_attention(query, key, changed_value, 64**-0.5, True)

    torch.testing.assert_close(
        original[:, :, :32],
        changed[:, :, :32],
        atol=0.0,
        rtol=0.0,
    )
