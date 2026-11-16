"""Tests for the portable Piper Attention reference."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

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
    # The math oracle avoids native BF16 CPU SDPA crashes on Windows runners.
    with sdpa_kernel(SDPBackend.MATH):
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


@pytest.mark.parametrize("is_causal", [False, True])
def test_reference_centering_restores_constant_value_exactly(is_causal: bool) -> None:
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
        is_causal,
    )

    torch.testing.assert_close(actual, value, atol=0.0, rtol=0.0)


def test_causal_reference_is_independent_of_future_1024_value_tiles() -> None:
    torch.manual_seed(53)
    query = torch.randn(1, 1, 1057, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    changed_value = value.clone()
    changed_value[:, :, 1024:] = torch.randn_like(changed_value[:, :, 1024:]) * 32

    original = reference_piper_attention(query, key, value, 64**-0.5, True)
    changed = reference_piper_attention(query, key, changed_value, 64**-0.5, True)

    torch.testing.assert_close(
        original[:, :, :1024],
        changed[:, :, :1024],
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("constant_value", [2.0**100, -(2.0**100)])
def test_reference_restores_large_constant_value(
    is_causal: bool,
    constant_value: float,
) -> None:
    query = torch.zeros(1, 2, 2048, 64, dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.full_like(query, constant_value)

    actual = reference_piper_attention(query, key, value, 64**-0.5, is_causal)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, value, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("is_causal", [False, True])
def test_reference_large_key_center_bias_stays_finite_in_fp32(is_causal: bool) -> None:
    query = torch.ones(1, 1, 1025, 64, dtype=torch.bfloat16)
    key = torch.full_like(query, 256.0)
    value = torch.ones_like(query)

    actual = reference_piper_attention(query, key, value, 1000.0, is_causal)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, value, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("is_causal", [False, True])
def test_reference_preserves_score_precision_under_large_k_bias(is_causal: bool) -> None:
    # A huge shared key offset (k_bias ~ 2e8) must not wash out small intra-chunk
    # score differences before softmax exponentiation.
    query = torch.zeros(1, 1, 64, 64, dtype=torch.bfloat16)
    query[:, :, :, :2] = 2.0
    key = torch.zeros_like(query)
    key[:, :, :, 0] = 1.0e9
    key[:, :, 0, 1] = 64.0
    value = torch.zeros_like(query)
    value[:, :, 0, 0] = 1.0

    actual = reference_piper_attention(query, key, value, 0.1, is_causal)

    assert torch.isfinite(actual).all()
    assert actual[0, 0, -1, 0].item() > 0.9


@pytest.mark.parametrize("is_causal", [False, True])
def test_reference_scales_multi_chunk_key_bias(is_causal: bool) -> None:
    torch.manual_seed(54)
    query = torch.zeros(1, 1, 1025, 64, dtype=torch.bfloat16)
    query[:, :, :, 0] = 1.0
    key = torch.zeros_like(query)
    key[:, :, 1024:, 0] = 8.0
    value = torch.zeros_like(query)
    value[:, :, 1024:, 0] = 1.0

    actual = reference_piper_attention(query, key, value, 0.125, is_causal)
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=0.125,
            is_causal=is_causal,
        )

    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)


@pytest.mark.parametrize("is_causal", [False, True])
def test_reference_preserves_intra_chunk_weights_under_large_k_bias(is_causal: bool) -> None:
    query = torch.zeros(1, 1, 128, 64, dtype=torch.bfloat16)
    query[:, :, :, :2] = 1.0
    key = torch.zeros_like(query)
    key[:, :, :, 0] = 1.0e9
    key[:, :, :64, 1] = 8.0
    key[:, :, 64:, 1] = -8.0
    value = torch.zeros_like(query)
    value[:, :, :64, 0] = 1.0

    actual = reference_piper_attention(query, key, value, 1.0, is_causal)

    assert torch.isfinite(actual).all()
    assert actual[0, 0, -1, 0].item() > 0.9


def test_causal_reference_is_independent_of_future_1024_key_chunks() -> None:
    query = torch.zeros(1, 1, 1027, 64, dtype=torch.bfloat16)
    query[:, :, :, 0] = 1.0
    key = torch.zeros_like(query)
    key[:, :, :512, 0] = 1.0
    key[:, :, 512:1024, 0] = -1.0
    value = torch.zeros_like(query)
    value[:, :, :512, 0] = 1.0
    changed_key = key.clone()
    changed_key[:, :, 1024, 0] = 2.0**60

    original = reference_piper_attention(query, key, value, 1.0, True)
    changed = reference_piper_attention(query, changed_key, value, 1.0, True)

    torch.testing.assert_close(
        original[:, :, :1024],
        changed[:, :, :1024],
        atol=0.0,
        rtol=0.0,
    )
